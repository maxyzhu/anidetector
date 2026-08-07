# AniDetector

A wildlife detection image processing pipeline.  
-> Ingests a folder of images  
-> runs MegaDetectorV6 (via PytorchWildlife)  
-> filters blank frames  
-> persists results to PostgreSQL, and exposes them over a DRF API + a minimal results page.  

Stack: Django + DRF · PostgreSQL · PytorchWildlife/MegaDetectorV6 · SpeciesNet · supervision · Pillow · Celery + Redis · uv · Docker.  

<img width="1512" height="856" alt="interface" src="https://github.com/user-attachments/assets/bcd97b14-22f7-4edd-8854-9d157dd2eef3" />

## Roadmap
1. Add species detection using "SpeciesNet"
2. Event Clustering


## First-time Use

Prereqs: `uv`, Docker Desktop.

**Install Docker Desktop** (then sign in):
```bash
brew install --cask docker
```

**Start services (Postgres + Redis) with Docker Compose.** Compose and Django
both read the same `.env`:
```bash
cp .env.example .env
docker compose up -d          # starts anidetector-pg + anidetector-redis
```

**Install deps and set up the database:**
```bash
uv sync                          # creates .venv, installs deps
uv run python manage.py migrate
uv run python manage.py runserver
```

Success criterion: `docker compose up -d` brings both containers up, `migrate`
completes, and `runserver` starts. The schema lives in `detections/models.py`
(see the diagram at the bottom).

**Validate the detector in isolation (no Django):**
```bash
uv run python scripts/try_model.py example_images/01.webp
```

**Validate the SpeciesNet classifier in isolation:**
```bash
uv run python scripts/try_speciesnet.py example_images/01.webp --threshold 0.5
```

## Daily Use

**Step 0 — start infra + web:**
```bash
docker compose up -d                       # Postgres + Redis
uv run python manage.py runserver          # Django on http://127.0.0.1:8000
```

**Step 1 — run the pipeline** (in another console):
```bash
uv run python manage.py ingest <folder>    # detect + persist; add --retry-failed to redo failures
uv run python manage.py cluster            # group images into events by capture time
uv run python manage.py classify_species   # SpeciesNet on animal boxes (sync)
uv run python manage.py rollup             # aggregate species per event
```

Run species classification asynchronously via Celery instead — start a worker,
then enqueue with `--async`:
```bash
uv run celery -A config worker -l info --pool=solo   # see note below
uv run python manage.py classify_species --async     # enqueue instead of running inline
```

> **macOS / GPU note:** the default `prefork` pool `fork()`s worker processes,
> and PyTorch **MPS (Metal)** and **CUDA** are not fork-safe — the worker crashes
> with `SIGABRT` the moment it touches the GPU. Use `--pool=solo` (single process,
> no fork) or `-P threads`. If a worker dies mid-batch, its detections are left in
> `processing`; re-run with `classify_species --retry-failed` to requeue them.

**Step 2 — view results:**
- `http://127.0.0.1:8000/results` — rendered gallery
- `http://127.0.0.1:8000/admin` — Django admin
- `http://127.0.0.1:8000/api/images` — image + detection + species JSON
- `http://127.0.0.1:8000/api/events` — event + species JSON


## Data Structure
```txt
┌─────────────────────────┐
│         Event           │
│─────────────────────────│
│ camera_site             │
│ start_time / end_time   │
│ image_count             │
└─────────────┬───────────┘
              │ 1
              │
      ┌───────┴────────┐
      │ N              │ N
      ▼                ▼
┌──────────────┐  ┌───────────────────┐
│    Image     │  │   EventSpecies    │
│──────────────│  │───────────────────│
│ path         │  │ category (species)│
│ status       │  │ confidence        │
│ captured_at  │  │ detection_count   │
│ camera_site  │  │ representative_   │
│ event ───────┼──┘  image (FK→Image) │
│ is_blank     │  └──────┼─────┬──────┘
└──────┬───────┘◄────────┘     │ N
       │ 1                     │
       │                       │
       │ N                     │
       ▼                       │
┌─────────────────────┐        │
│     Detection       │        │
│─────────────────────│        │
│ image (FK→Image)    │        │
│ category            │        │
│ (person/vehicle/    │        │
│  animal)            │        │
│ bbox_x1..y2         │        │
│ confidence          │        │
│ status (async queue)│        │
└──────────┬──────────┘        │
           │ 1                 │
           │                   │
           │ N                 │
           ▼                   │
┌────────────────────────────┐ │
│   SpeciesClassification    │ │
│────────────────────────────│ │
│ detection (FK→Detection)   │ │
│ source (speciesnet/human)  │ │
│ category (species)         │ │
│ confidence                 │ │
│ top_k (JSON)               │ │
│ annotator (FK→User) ────┐  │ │
└─────────────────────────┼──┘ │
                          │    │
                          ▼    │
                    ┌────────────┐
                    │    User    │
                    │ (settings. │
                    │AUTH_USER_  │
                    │  MODEL)    │
                    └────────────┘
```
