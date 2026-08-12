# Gen-AI-Dev API

A FastAPI service for asking grounded questions over a documentation corpus.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell/cmd)
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Retrieval works
without a key; written answers do not.

## Run

**There is no separate frontend server.** The UI is one static file
(`app/web/index.html`) served by the backend itself, so a single command runs
both halves:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Leave that running and open <http://127.0.0.1:8000/ui>. `/` redirects there.

| What | Where |
| --- | --- |
| Frontend — ask-a-question page | http://127.0.0.1:8000/ui |
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |
| Health | http://127.0.0.1:8000/health |
| What generation can do right now | http://127.0.0.1:8000/api/v1/chat/status |

The root `web/` directory is unused scaffolding — no `package.json`, no
sources. Ignore it; there is nothing to `npm run` there.

### The index

The page loads either way, but asking a question returns 503 until the vector
index exists. The repo already ships a built `.rag_index/`, so you only need to
rebuild after changing the corpus or a chunker:

```bash
python -m app.rag.index --all
```

### Two terminals, if you want them

Nothing requires a second terminal, but it is convenient to keep the server in
one and drive the API from another.

Terminal 1 — the server (backend + UI):

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — exercise it. PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/chat/ask `
  -ContentType application/json `
  -Body '{"question":"How do I authenticate?","k":5}'
```

bash / Git Bash:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I authenticate?","k":5}'
```

## Other entrypoints

Each runs standalone, no server needed:

```bash
python -m app.rag.ingest              # load and normalise the corpus
python -m app.rag.chunk               # chunk it
python -m app.rag.index --all         # embed and index every strategy
python -m app.rag.index --search "how do I authenticate?" -k 5
python -m app.rag.evaluate            # hit-in-top-k over eval/questions.yaml
```

## Layout

```
app/
  main.py            # app factory, CORS, router wiring, /ui and / routes
  models.py          # Pydantic schemas
  core/config.py     # settings via pydantic-settings (.env)
  web/index.html     # the frontend — one file, no build step
  api/routes/
    health.py        # GET /health
    ingest.py        # corpus endpoints under /api/v1/ingest
    chat.py          # /api/v1/chat/ask and /api/v1/chat/status
  rag/               # ingest, chunkers, embeddings, index, generate, evaluate
eval/questions.yaml  # evaluation set
.rag_index/          # persisted Chroma index (built by app.rag.index)
```
