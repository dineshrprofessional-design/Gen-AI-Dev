from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Gen-AI-Dev API"
    version: str = "0.1.0"
    debug: bool = False
    cors_origins: list[str] = ["*"]

    # Where the documentation corpus lives. Overridable per request so a client
    # can point at a different corpus without editing config.
    docs_root: str = "docs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
