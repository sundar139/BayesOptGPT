from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        validation_alias="MLFLOW_TRACKING_URI",
    )
    enable_mlflow: bool = Field(default=True, validation_alias="ENABLE_MLFLOW")
    model_dir: Path = Field(
        default=Path("artifacts/model"),
        validation_alias="MODEL_DIR",
    )
    tokenizer_dir: Path = Field(
        default=Path("artifacts/tokenizer"),
        validation_alias="TOKENIZER_DIR",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=7860, validation_alias="PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
