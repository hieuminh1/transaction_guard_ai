from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    model_path: Path = Path("artifacts/model.joblib")
    model_metadata_path: Path = Path("artifacts/metadata.json")
    database_url: str | None = None
    log_level: str = "INFO"
    max_batch_size: int = Field(default=100, ge=1, le=100)
