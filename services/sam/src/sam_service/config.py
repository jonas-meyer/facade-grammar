"""Service configuration.

All settings come from env vars (``SAM_`` prefix) or ``.env``. LitServe's
``accelerator="auto"`` picks CUDA/ROCm first, then MPS, else CPU — the
``device`` override here is only for pinning in tests.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_id: str = "facebook/sam3"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    max_batch_size: int = 8
    batch_timeout: float = 0.05
    request_timeout: float = 600.0
