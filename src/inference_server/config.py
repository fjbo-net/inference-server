"""Application settings, parsed from environment variables and an optional .env file.

When running as a frozen executable (PyInstaller), external files such as the
models directory and the .env file are resolved relative to the executable's
directory rather than the bundle's extraction directory.
"""

import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_base_dir() -> Path:
    """Return the directory external files (models, .env) are resolved against."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path.cwd()


def _default_models_dir() -> Path:
    return get_base_dir() / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=get_base_dir() / ".env",
        env_file_encoding="utf-8",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    models_dir: Path = Field(default_factory=_default_models_dir)


def get_settings() -> Settings:
    return Settings()
