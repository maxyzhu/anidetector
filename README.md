# AniDetector

A wildlife detection pipeline to pre-process camera-trap **image** and **video**, helping
researchers filter out empty frames and roughly cluster behaviors (now with active/reset, 
future with VLM named clusters).


<table>
  <tr>
    <td align="center">
      <img src="./asset/home.jpeg" width="400"/><br/>
      <sub>home</sub>
    </td>
    <td align="center">
      <img src="./asset/image.jpeg" width="400"/><br/>
      <sub>image</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./asset/video.jpg" width="400"/><br/>
      <sub>video</sub>
    </td>
    <td align="center">
      <img src="./asset/track_detail.jpg" width="400"/><br/>
      <sub>track detail</sub>
    </td>
  </tr>
</table>


## Feature
1. Animal Detection via MegaDetector — *completed in 2026.07.28*
2. Species identification via SpeciesNet — *completed in 2026.08.06*
3. Image processing — *completed in 2026.08.12*
4. Video decoding and frame extraction, Animal Tracking with **SORT and Kalman filter**,
   Activity bouts and representative frame (active/rest) — *completed in 2026.09.04*
5. Throughput measurement: 12 h video per 5.5 h. Decode and sampling run at ~325 fps,
   the detector forward pass is 88 ms/frame — *completed in 2026.09.07*
6. Replace SORT with **ByteTrack**: keep animals to partial occlusion — *completed in 2026.09.07*
7. Pet-set validation run end to end. — *completed in 2026.09.07*

## Roadmap
11. **Enhance Detector throughput.** Real batching means first fixing upstream's
   `batch_image_detection`, which divides x by the image *height* and y by the
   *width*. That leaves the model: quantisation, ONNX Runtime or CoreML, or a
   smaller variant.
12. **TW-FINCH** cluster wildlife behaviors and naming with local VLM.
   Refer to: https://arxiv.org/abs/2103.11264
13. Change-point detection over tracks, and event clustering built on it.


## First-time Use
**Clone the project to your computer**

**Mac OS Only**
*Install Docker Desktop and sign in* https://www.docker.com/products/docker-desktop/
open terminal, direct to your anidetector root folder, then
```bash
cd <your anidetector root folder>
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Linux OS Only**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
cd <your anidetector root folder>
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows Only**
*Install Docker Desktop and sign in* https://www.docker.com/products/docker-desktop/
```powershell
cd <your anidetector root folder>
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Close the terminal and open a new one**

**Copy .env.example and change the Django key with yours** paste the key to DJANGO_SECRET_KEY
```bash
uv sync                          # creates .venv, installs deps
cp .env.example .env
uv run python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

**Register a Django superadmin**

**Start services (Postgres + Redis)**
```bash
docker compose up -d          # starts anidetector-pg + anidetector-redis
uv run python manage.py migrate
uv run python manage.py runserver
```

Success criterion: `docker compose up -d` brings both containers up, `migrate`
completes, and `runserver` starts. The schema lives in `core/models.py`
(see the diagram at the bottom).


## Validation

**Validate the detector in isolation:**
```bash
uv run python golden_value/try_models.py example_images/01.webp
```

**Validate the SpeciesNet classifier in isolation:**
```bash
uv run python golden_value/try_speciesnet.py example_images/01.webp --threshold 0.5
```

**(Optional) Measure the decode → detect throughput**
The recorded run is `golden_value/baseline_mps.json`; `--check` re-runs and fails if
throughput drops more than 20%, and refuses to compare across devices:
```bash
uv run python golden_value/benchmark_hot_path.py <video> --device mps
uv run python golden_value/benchmark_hot_path.py <video> --device mps --check golden_value/baseline_mps.json
```


## Daily Use

**Step 0 — start infra + web:**
```bash
docker compose up -d                       # Postgres + Redis
uv run python manage.py runserver          # Django on http://127.0.0.1:8000
```

**Step 1 — create a Deployment in admin**
http://127.0.0.1:8000/admin
Sign in with your superadmin account (see First-time Use if you don't have one)
Click "+Add" button behind Deployment
Or:
```bash
uv run python manage.py shell -c "
from django.utils import timezone
from core.models import Deployment
print(Deployment.objects.create(
    camera_id='cam1', location='backyard', country='USA',
    start_ts=timezone.now(),
).id)"
```

**Step 2 — open the home page**
http://127.0.0.1:8000
Follow the instruction to process videos or image


## Custom Use (Hack)

**Step 2a — images:**
```bash
uv run python manage.py ingest <folder> --deployment <id>   # detect + persist
uv run python manage.py classify_species                    # SpeciesNet on animal boxes
```
`ingest` takes `--retry-failed` to redo failures, `--limit` to cap new rows.

**Step 2b — video:**
Quick start:
```bash
uv run python manage.py quickstart <folder> --deployment <id>
```
Custom:
```bash
uv run python manage.py register_video <folder> --deployment <id>
uv run python manage.py process_video                       # decode -> tracks + signals
uv run python manage.py activity_report                     # bouts and activity budgets
```

Add `--contiguous` to `register_video` when the files are one recording the camera
split up, in filename order; that sets each `start_offset_ts` from the accumulated
durations. It is off by default because inferring contiguity that is not there
produces a silently wrong timeline.

`process_video` writes one representative frame per track and enqueues it for
species classification automatically — no second command needed, as long as a
Celery worker is running.

`activity_report` takes every threshold as an argument rather than a setting
(they are query parameters, not a property of the data), plus `--sensitivity` to
sweep them instead of reporting one setting:
```bash
uv run python manage.py activity_report --displacement-enter 0.5 --min-duration 1.0
uv run python manage.py activity_report --sensitivity --json out.json
```

**Async classification via Celery** — start a worker, then enqueue with `--async`:
```bash
uv run celery -A config worker -l info --pool=solo   # see note below
uv run python manage.py classify_species --async     # enqueue instead of running inline
```

> **macOS / GPU note:** the default `prefork` pool `fork()`s worker processes,
> and PyTorch **MPS (Metal)** and **CUDA** are not fork-safe — the worker crashes
> with `SIGABRT` the moment it touches the GPU. Use `--pool=solo` (single process,
> no fork) or `-P threads`. If a worker dies mid-batch, its detections are left in
> `processing`; re-run with `classify_species --retry-failed` to requeue them.

**Step 3 — view results:**
- `http://127.0.0.1:8000/results` — rendered gallery (stills only)
- `http://127.0.0.1:8000/results/<id>/annotated.png` — boxes + species drawn on one image
- `http://127.0.0.1:8000/admin` — Django admin
- `http://127.0.0.1:8000/api/images/` — image + detection + species JSON
- `http://127.0.0.1:8000/api/video/tracks/<id>/` — track detail: species, bout
  timeline and per-behaviour images
- `http://127.0.0.1:8000/api/video/tracks/<id>/behaviour/active.jpg` — a
  representative frame for the moving (or `rest.jpg`, resting) behaviour, extracted
  on demand for the thresholds in the query string and cached against them


## Architecture

**Images** -> ingest a folder -> MegaDetectorV6 (via PytorchWildlife) detect -> filter blank
frames -> SpeciesNet classification.

**Video** -> register a folder -> decode with PyAV and sample -> MegaDetectorV6 detect 
-> ByteTrack animal tracking and per-track motion signals  -> SpeciesNet classification
-> activity bouts.

Both persist to PostgreSQL and are exposed over a DRF API plus a minimal results page.

Required Dependencies: Django + DRF · PostgreSQL · PytorchWildlife/MegaDetectorV6 · SpeciesNet ·
PyAV · supervision · Pillow · NumPy · Celery + Redis · uv · Docker.

```mermaid
flowchart LR
   subgraph Inputs["📷 Camera‑Trap Inputs"]
       IMG[Image Folder]
       VID[Video Folder]
   end
   %% Image path
   IMG -->|Ingest| PRE_IMG[Filter blank frames]
   %% Video path
   VID -->|PyAV Decode & Sample| PRE_VID[ByteTrack + Motion signals]
   %% Shared detection step: single MegaDetectorV6
   PRE_IMG & PRE_VID --> MD[MegaDetectorV6<br/>Detection]
   %% Single unified Celery task queue
   MD -->|Enqueue animal boxes / track representative frames| CELERY[Celery Task]
   CELERY --> SN[SpeciesNet<br/>Species Classification]
   %% Storage
   SN --> DB[(PostgreSQL)]
   DB --> DRF[DRF API]
   %% Image result path
   DRF --> WEB_IMG[Web Image Result]
   %% Video: activity bouts computed on‑the‑fly at query time
   DRF -->|On‑demand calculation<br/>Activity bouts| WEB_VID[Web Video Result]
   subgraph Backend["⚙️ Backend Infrastructure"]
       REDIS[Redis • Celery Broker]
   end
   CELERY <--> REDIS
   classDef input fill:#e8f4f8,stroke:#2385bb
   classDef ai fill:#eaf8ea,stroke:#34a853
   classDef store fill:#fff6e6,stroke:#f29900
   classDef infra fill:#f3e8fc,stroke:#9c27b0
   class IMG,VID input
   class MD,PRE_IMG,PRE_VID,SN ai
   class DB store
   class REDIS,DRF,WEB_IMG,WEB_VID infra
```

## Package layout

```txt
core       ← models, taxonomy, box drawing. Imports nothing of ours.
inference  ← model loading and inference. No Django import anywhere.
image      ← the stills workflow. May import core + inference.
video      ← the video workflow. May import core + inference.
```

`image` and `video` never import each other; they meet at the `core.Detection`
table. `tests/test_boundries.py` enforces all of this as an executable rule.

## Data Structure

```txt
                        ┌───────────────────────────┐
                        │        Deployment         │
                        │───────────────────────────│
                        │ camera_id / location      │
                        │ country / modality        │
                        │ start_ts / end_ts         │
                        └─────┬───────────────┬─────┘
                         1:N  │               │  1:N
              ┌───────────────┘               └───────────────┐
              ▼                                               ▼
┌──────────────────────────┐                    ┌───────────────────────────┐
│          Image           │                    │           Media           │
│──────────────────────────│                    │───────────────────────────│
│ path (unique)            │                    │ path (unique) / kind      │
│ source: ingest |         │                    │ uploaded_at               │
│         video_frame      │                    │ fps / duration / gop_size │
│ captured_at              │                    │ start_offset_ts           │
│ width / height           │                    │ width / height / status   │
│ status / is_blank        │                    └─────────────┬─────────────┘
└────────────┬─────────────┘                              1:N │
         1:N │                                                ▼
             ▼                                  ┌───────────────────────────┐
┌──────────────────────────┐                    │           Track           │
│        Detection         │                    │───────────────────────────│
│──────────────────────────│                    │ media (FK→Media)          │
│ image (FK→Image)         │◄───rep_detection───┤ start_frame / end_frame   │
│ category                 │                    │ start_ts / end_ts / hits  │
│  (person/vehicle/animal) │                    │ rep_ts                    │
│ bbox_x1..y2              │                    └─────────────┬─────────────┘
│ confidence               │                              1:N │
│ status (species queue)   │                                  ▼
└────────────┬─────────────┘                    ┌───────────────────────────┐
         1:N │                                  │       MotionSignal        │
             ▼                                  │───────────────────────────│
┌──────────────────────────┐                    │ track (FK→Track)          │
│  SpeciesClassification   │                    │ ts                        │
│──────────────────────────│                    │ displacement_bl_per_s     │
│ detection (FK→Detection) │                    │ deformation_score         │
│ source (speciesnet/human)│                    │ confidence                │
│ category (species label) │                    │ bbox_x1..y2               │
│ confidence / top_k       │                    └───────────────────────────┘
│ model_version            │
│ annotator (FK→User)      │
└──────────────────────────┘
```

Two things in that diagram carry most of the design:

**A track's representative frame is an `Image` with one `Detection` on it.** That
is why a track gets classified by the stills species queue with no video-specific
code, and why `annotated.png` draws boxes on it for free. `Image.source` keeps
those frames out of the photo gallery.

**There is no Bout table.** Bouts are computed from `MotionSignal` at query time,
because thresholds are argued about and storing the verdict would mean re-decoding
terabytes every time one changes. `SpeciesClassification` is append-only for the
same reason: re-running a model version leaves history rather than overwriting it.
