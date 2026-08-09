from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api.routes import chunking, health, ingest
from app.core.config import get_settings

WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        debug=settings.debug,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(chunking.router, prefix="/api/v1")

    @app.get("/ui", include_in_schema=False)
    def chunk_viewer() -> FileResponse:
        """The chunk viewer — one static page, no build step."""
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    return app


app = create_app()
