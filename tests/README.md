## Testing
```bash
uv run pytest
```
Tests inject a fake classifier (no model weights) and run Celery eagerly (no
broker), but still need Postgres up (`docker compose up -d`).
