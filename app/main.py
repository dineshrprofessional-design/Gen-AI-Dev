from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, items
from app.core.config import get_settings


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
    app.include_router(items.router, prefix="/api/v1")

    @app.get("/", tags=["root"])
    def root() -> dict[str, str]:
        return {"message": f"{settings.app_name} is running", "docs": "/docs"}

    return app


app = create_app()
