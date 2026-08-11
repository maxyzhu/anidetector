# AniDetector 结构重构方案：为视频流水线做准备

> 目标：让图片和视频两条流水线共享推理层，但互不依赖。
> 原则：**只搬移，不改逻辑。** 每一步做完必须验证现有图片流程行为不变，再进行下一步。

---

## 0. 当前状态与核心问题

```
detections/
├── management/commands/
│   ├── ingest.py            ← MegaDetector 逻辑在这里
│   ├── classify_species.py  ← SpeciesNet 逻辑在这里
│   ├── rollup.py            ← 分类树 rollup，图片和视频都要用
│   ├── cluster.py
│   └── profile.py
├── inference.py             ← 已存在，抽取的种子
├── models.py
├── tasks.py
└── ...
```

**核心问题：业务逻辑住在 management command 里。**

management command 是 CLI 入口点。`video` 想复用 MegaDetector，就必须 `from detections.management.commands.ingest import ...` —— 这既丑陋又会拖入 Django 的命令框架依赖，而且无法在没有 Django 的环境下测试。

所以重构的实质不是"新建文件夹"，是**把逻辑从命令里挖出来，放进可被任何地方 import 的模块**。

---

## 1. 目标结构

```
config/                      不变
│
core/                        【Django app，新建】
├── __init__.py
├── apps.py
├── models.py                Deployment, Media, Taxon
├── taxonomy.py              ← 从 commands/rollup.py 抽出
└── migrations/
│
inference/                   【纯 Python 包，不是 Django app】
├── __init__.py
├── types.py                 Frame, Detection, Crop, SpeciesVote
├── registry.py              模型加载 / 缓存 / 变体选择 / 许可校验
├── detector.py              MegaDetector 封装
├── classifier.py            SpeciesNet 封装
└── sources.py               FrameSource 协议 + 图片目录实现
│
detections/                  【保留，收敛为"图片工作流"】
├── management/commands/
│   ├── ingest.py            → 瘦身为 CLI 壳
│   ├── classify_species.py  → 瘦身为 CLI 壳
│   ├── cluster.py
│   └── profile.py
├── services.py              【新增】原本在 command 里的编排逻辑
├── models.py
├── tasks.py
└── ...
│
video/                       【Django app，新建】
├── __init__.py
├── apps.py
├── models.py                Track, MotionSignal
├── decode.py                PyAV 帧来源
├── sampling.py              关键帧 / 步长 / 门控
├── tracking.py              SORT → ByteTrack
├── motion.py                位移 + 形变
├── bouts.py                 查询时计算，无表
├── services.py              流水线编排
├── tasks.py                 Celery
└── migrations/
```

### 为什么 `inference` 是纯 Python 包而不是 Django app

它没有 model、没有 view、没有 migration。做成纯包的好处：

- 不依赖 Django，可以脱离框架单独测试
- 不需要写进 `INSTALLED_APPS`
- 将来想抽成独立库或者给边缘设备用，直接搬走即可

`core` 和 `video` 有 model，必须是 Django app。

### 依赖方向（铁律）

```
core  ←  inference  ←  detections
  ↑          ↑
  └──────────┴──────  video
```

**`detections` 和 `video` 永远不互相 import。** 破了这条，两周内就会缠成一团。

---

## 2. 执行顺序

风险从低到高。**不要跳步，不要并行做。**

| 步骤 | 内容 | 风险 | 可回滚 |
|---|---|---|---|
| Step 1 | 抽 `inference` 包 | 低 | ✅ |
| Step 2 | 命令瘦身 → `services.py` | 低 | ✅ |
| Step 3 | 建 `core`（仅新模型） | 低 | ✅ |
| Step 4 | 建 `video` | 低 | ✅ |
| Step 5 | Celery 队列分离 | 低 | ✅ |
| Step 6 | 图片模型迁移到 `core` | **高** | ⚠️ 需数据迁移 |

**Step 6 现在不要做。** 等视频流水线跑通、结构验证过之后再说。

---

## 3. Step 1 — 抽 `inference` 包

### 3.1 先读 `detections/inference.py`

它大概率已经包含了模型加载和推理调用。先判断里面有什么，再决定怎么切。典型内容会是：模型路径解析、权重加载、单张/批量推理、结果后处理。

### 3.2 建立包骨架

```
inference/
├── __init__.py
├── types.py
├── registry.py
├── detector.py
├── classifier.py
└── sources.py
```

**`inference` 不进 `INSTALLED_APPS`。**

### 3.3 `types.py` — 先定契约

```python
from dataclasses import dataclass
from typing import Protocol, Iterator
import numpy as np

@dataclass
class Frame:
    array: np.ndarray          # HWC, RGB
    timestamp: float           # 秒，相对 media 起点
    frame_index: int
    media_id: str

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # xyxy, 归一化
    confidence: float
    category: str                              # animal / person / vehicle

@dataclass
class SpeciesVote:
    label: str
    score: float
    taxon_path: str            # 纲;目;科;属;种;俗名

class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...
```

`Frame` 这个抽象是整个重构的支点：`detections` 的实现是遍历图片目录，`video` 的实现是 PyAV 解码，**推理层完全不知道区别**。

将来加边缘实时流，也只是第三个 `FrameSource` 实现。

### 3.4 `registry.py` — 集中模型加载

把散落在 `ingest.py` / `classify_species.py` / `inference.py` 里的模型加载收拢到一处。必须处理的事：

- 变体选择：只允许 `MDV6-apa-rtdetr-e`（Apache-2.0）或 `MDV6-mit-yolov9-e`（MIT）
- 加载类必须是 `MegaDetectorV6Apache` / `MegaDetectorV6MIT`，**不是通用的 `MegaDetectorV6()`**（默认变体走 ultralytics，AGPL 会传染整个项目）
- 进程内缓存：模型只加载一次，Celery worker 复用
- **SpeciesNet 默认 ensemble 会自动下载 MegaDetector v5a（AGPL）作为检测器，必须显式替换掉**

建议加一个启动期断言，防止将来手滑引入 AGPL 依赖。

### 3.5 `detector.py` / `classifier.py` — 只做搬移

从 `ingest.py` 挖出的应该只是**纯推理部分**：图像数组进，`Detection` 列表出。

不要带进去的东西：文件遍历、进度条、数据库写入、Django ORM 调用。那些留给 `services.py`。

`classifier.py` 同理：裁切进，`SpeciesVote` 列表出。

### 3.6 验证

`scripts/try_models.py` 和 `scripts/try_speciesnet.py` 正好是现成的冒烟测试。改成 import 新包，跑一遍，输出与重构前一致即通过。

---

## 4. Step 2 — 命令瘦身

### 目标形态

management command 只负责三件事：解析参数、调用 service、打印进度。**一行业务逻辑都不留。**

```python
# detections/management/commands/ingest.py
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("path")
        parser.add_argument("--batch-size", type=int, default=8)

    def handle(self, *args, **opts):
        from detections.services import ingest_directory
        for progress in ingest_directory(opts["path"], opts["batch_size"]):
            self.stdout.write(f"{progress.done}/{progress.total}")
```

### `detections/services.py`

装原本在命令里的编排：遍历目录 → 构造 `FrameSource` → 调 `inference` → 写库 → 产出进度。

用 generator 产出进度，这样 CLI 和 Celery task 可以共用同一个函数，只是消费进度的方式不同。

### 注意

`rollup.py` 里的分类树逻辑**不要**放进 `services.py` —— 它是图片和视频共享的，Step 3 会把它挪到 `core/taxonomy.py`。这一步先原地不动。

---

## 5. Step 3 — 建 `core`

### 只放新模型，不碰旧的

```python
# core/models.py
class Deployment(models.Model):
    """相机 × 地点 × 时间段。不是"一个视频文件"。"""
    camera_id = models.CharField(max_length=64)
    location = models.CharField(max_length=255)
    lat = models.FloatField(null=True)
    lon = models.FloatField(null=True)
    start_ts = models.DateTimeField()
    end_ts = models.DateTimeField(null=True)
    modality = models.CharField(max_length=16)   # rgb | thermal

class Media(models.Model):
    """一个媒体资源。图片和视频共用这张表。"""
    deployment = models.ForeignKey(Deployment, on_delete=models.CASCADE)
    path = models.CharField(max_length=1024)
    kind = models.CharField(max_length=16)       # image | video
    start_offset_ts = models.FloatField(default=0)
    fps = models.FloatField(null=True)
    gop_size = models.IntegerField(null=True)
    duration = models.FloatField(null=True)
```

`Media` 同时覆盖图片和视频不是设计偏好 —— **Camtrap DP 标准就是 deployments / media / observations 这个结构**。将来导出和发布 GBIF 时不用重构数据模型。

### `core/taxonomy.py`

把 `commands/rollup.py` 的逻辑搬过来，改成可 import 的函数。图片和视频都要用它做分类层级聚合。

原命令改成薄壳调用它。

### 关键：不迁移 detections 的现有模型

`detections/models.py` 原样保留，继续服务图片流程。

**新写的 `video` 从第一天就用 `core` 的模型。** 这样图片那边什么时候迁移都行，不阻塞视频开发。

这是 strangler fig 模式：新代码用新结构，旧代码原地不动，等新结构验证过再迁移旧的。

---

## 6. Step 4 — 建 `video`

```python
# video/models.py
from core.models import Media

class Track(models.Model):
    media = models.ForeignKey(Media, on_delete=models.CASCADE)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    start_ts = models.FloatField()
    end_ts = models.FloatField()
    species_label = models.CharField(max_length=255, null=True)
    species_votes = models.JSONField(null=True)      # 完整投票分布
    species_agreement = models.FloatField(null=True) # 一致率，低=需复核
    taxon_path = models.CharField(max_length=512, null=True)

class MotionSignal(models.Model):
    track = models.ForeignKey(Track, related_name="signals", on_delete=models.CASCADE)
    ts = models.FloatField()
    displacement_bl_per_s = models.FloatField()   # 体长/秒，尺度不变
    deformation_score = models.FloatField()
    bbox = models.JSONField()

    class Meta:
        indexes = [models.Index(fields=["track", "ts"])]
```

**没有 `Bout` 表。** bout 由 `bouts.py` 在查询时从 `MotionSignal` 计算，阈值是查询参数。这样改阈值不用重跑视频。

### 骨架文件

先建空文件 + 函数签名，不实现：`decode.py` / `sampling.py` / `tracking.py` / `motion.py` / `bouts.py` / `services.py` / `tasks.py`。

先让结构立起来，再往里填。

---

## 7. Step 5 — Celery 队列分离

这是运维硬需求，不是洁癖。

| | 图片 | 视频 |
|---|---|---|
| 单任务时长 | 秒～分钟 | **小时** |
| 并发度 | 高 | 低（GPU / 内存受限） |
| 断点续跑 | 不太需要 | **必需** |

同队列的后果：一个视频任务把 worker 堵死几小时，图片任务全部饿死。

```python
# config/settings.py
CELERY_TASK_ROUTES = {
    "detections.tasks.*": {"queue": "images"},
    "video.tasks.*":      {"queue": "video"},
}
```

`docker-compose.yml` 里起两组 worker：

```yaml
worker-images:
  command: celery -A config worker -Q images -c 4
worker-video:
  command: celery -A config worker -Q video -c 1
```

视频 worker 并发设 1（或按 GPU 数），避免显存打爆。

---

## 8. 验证清单

每步做完必须过：

- [ ] **Step 1** — `scripts/try_models.py` 和 `try_speciesnet.py` 输出与重构前逐字一致
- [ ] **Step 1** — 全局 grep 确认没有 `import ultralytics`
- [ ] **Step 2** — `python manage.py ingest <已有目录>` 结果与重构前一致
- [ ] **Step 2** — 命令文件里没有任何模型调用或 ORM 写入
- [ ] **Step 3** — `makemigrations` 只对 `core` 生成新表，**不对 `detections` 产生任何变更**
- [ ] **Step 4** — `video` 的 import 里没有 `detections`
- [ ] **Step 4** — `detections` 的 import 里没有 `video`
- [ ] **Step 5** — 长视频任务跑起来时，图片任务仍能正常调度

---

## 9. 明确的禁止事项

- ❌ **不要在重构过程中改逻辑。** 只搬移。想优化的地方记 TODO，重构完再动。
- ❌ **不要现在迁移 `detections` 的模型到 `core`。** 那需要数据迁移，是唯一高风险的一步，等视频跑通再说。
- ❌ **不要让 `inference` 依赖 Django。** 它一旦 import 了 `django.db`，就再也脱不掉了。
- ❌ **不要按"图片 vs 视频"切分推理代码。** 那会导致两份模型加载逻辑在三个月内漂移 —— 这是最经典的死法。
- ❌ **不要一次做完所有步骤再测。** 每步单独验证。

---

## 10. 顺带处理的小事

- `commands/profile.py` 目前未提交（U）。**提交它。** `inference` 抽出来之后，它可以独立 profile 推理层，而这正是 M0 基线测量要用的工具。
- `visualize.py` 暂时留在 `detections`。视频的可视化需求（画轨迹、时间轴）差异较大，等真正写到那里再决定是否抽公共部分。调色板定义（R/B 通道预交换那段）如果 video 也要用，单独抽一个 `core/palette.py`。
- `scripts/try_*.py` 保留，它们是 `inference` 包最方便的冒烟测试入口。
