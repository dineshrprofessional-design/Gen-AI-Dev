# Gen-AI-Dev API

A FastAPI service scaffold.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell/cmd)
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API root: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/health

## Test

```bash
pytest -q
```

## Layout

```
app/
  main.py            # app factory, CORS, router wiring
  models.py          # Pydantic schemas
  core/config.py     # settings via pydantic-settings (.env)
  api/routes/
    health.py        # GET /health
    items.py         # CRUD under /api/v1/items
tests/test_api.py
```

Items are held in an in-memory dict — swap `app/api/routes/items.py` for a real
database layer when you need persistence.
