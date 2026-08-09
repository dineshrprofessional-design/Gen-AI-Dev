from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version)
