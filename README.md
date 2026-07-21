# Wildlife Detector

MegaDetector-based image detection pipeline. Ingests a folder of images, runs
MegaDetectorV6 (via PytorchWildlife), filters blank frames, persists results to
PostgreSQL, and exposes them over a DRF API + a minimal results page.

Stack: Django + DRF · PostgreSQL · PytorchWildlife/MegaDetectorV6 · Pillow ·
uv · Docker.

## Step 0 — foundation

Prereqs: `uv`, a running PostgreSQL. Easiest Postgres for now:

```bash
docker run --name wildlife-pg -e POSTGRES_DB=wildlife \
  -e POSTGRES_USER=wildlife -e POSTGRES_PASSWORD=wildlife \
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

## Step 1 — schema + single-image inference

The schema lives in `detections/models.py` (`Image`, `Detection`). Migrations
were generated with `makemigrations`; `migrate` creates the tables.

Validate the model in isolation (no Django):

```bash
uv run python scripts/try_model.py sample_data/<some.jpg>
```

Read the printed structure of the result, then wire it into the pipeline in
Step 2 (`manage.py ingest <folder>`).