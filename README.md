# Wildlife Detector

MegaDetector-based image detection pipeline. Ingests a folder of images, runs
MegaDetectorV6 (via PytorchWildlife), filters blank frames, persists results to
PostgreSQL, and exposes them over a DRF API + a minimal results page.

Stack: Django + DRF · PostgreSQL · PytorchWildlife/MegaDetectorV6 · Pillow ·
uv · Docker.

More to go:
Celery + Redis (in-developement)

## First-time Use

Prereqs: `uv`, `docker desktop`, a running PostgreSQL. 

**Download Docker Desktop:**
```bash
brew install --casj docker
```

Then: sign-in to Docker Desktop.

**Create Postgres:**
```bash
docker run --name anidector-pg -e POSTGRES_DB=anidector \
  -e POSTGRES_USER=anidector -e POSTGRES_PASSWORD=anidector \
  -p 5432:5432 -d postgres:16
```

Then:
```bash
cp .env.example .env
uv sync                      # creates .venv, installs deps
uv run python manage.py migrate
uv run python manage.py runserver
```

Success criterion: `runserver` starts and `migrate` completes against Postgres.
The model is not touched yet.

The schema lives in `detections/models.py` (`Image`, `Detection`). Migrations
were generated with `makemigrations`; `migrate` creates the tables.

**Validate the model in isolation (no Django):**
```bash
uv run python scripts/try_model.py example_images/01.webp
```

## Daily Use
Read the printed structure of the result, then wire it into the pipeline in
**Step 0:**
`docker start` - start docker
`source .venv\bin\activate` - activate uv environment
`uv run python manage.py runserver` - start Django

**Step 1:**
start another console
`uv run manage.py ingest <folder>` - Inference all valid images in the folder.
`uv run manage.py ingest <folder> --retry-failed` - Retry inferencing failed images in the folder.

**Setp 2:**
open `http://127.0.0.1/results` to see inference results
open `http://127.0.0.1/admin` to manage Postgres
open `http://127.0.0.1/api/images` to see JSON results