from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"

    database_url: str = Field(default="postgresql+asyncpg://archstyle:archstyle@postgres:5432/archstyle")
    database_url_sync: str = Field(default="postgresql+psycopg://archstyle:archstyle@postgres:5432/archstyle")
    redis_url: str = "redis://redis:6379/0"

    hf_token: str = ""
    hf_org: str = "kkkaredaw"
    hf_model_repo: str = "kkkaredaw/archstyle55-backbones"
    hf_dataset_repo: str = "kkkaredaw/archstyle55-scraped"

    wandb_api_key: str = ""
    wandb_project: str = "archstyle-vkr"

    prefect_api_url: str = ""
    prefect_api_key: str = ""

    public_backend_url: str = "http://localhost:8000"
    streamlit_theme_base: str = "light"

    model_cache_dir: Path = Path("/data/models")
    model_lru_slots: int = 3
    inference_device: str = "cpu"
    torch_num_threads: int = 4

    rate_limit_predict: str = "30/minute"
    allowed_origins: str = "*"

    scraper_user_agent: str = "archstyle55-research-bot/1.0"
    scraper_max_per_job: int = 100

    runs_dir: Path = Path("/runs_res")
    repo_dir: Path = Path("/repo")


@lru_cache
def get_settings() -> Settings:
    return Settings()
