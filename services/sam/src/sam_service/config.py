"""Service configuration.

All settings come from env vars (``SAM_`` prefix) or ``.env``. ``accelerator``
is passed straight to ``LitServer``; the default ``auto`` works for MPS /
local dev, but ROCm deployments must set ``SAM_ACCELERATOR=cuda`` because
LitServe's auto-probe calls ``nvidia-smi`` which ROCm doesn't ship.
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
    accelerator: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    # LitServer's ``devices="auto"`` probe returns 0 on ROCm-built torch (the
    # device introspection path they take doesn't recognise HIP devices), which
    # then dies with ``num_api_servers must be greater than 0``. Pin 1 worker
    # in that case via ``SAM_DEVICES=1``.
    devices: int | Literal["auto"] = "auto"
    # Multiple model replicas on one GPU — each worker is its own process with
    # its own model + CUDA context. Throughput gain on a single consumer-grade
    # ROCm GPU is modest (~15% measured on RX 6800, both workers contend on
    # the same compute units — no NVIDIA-MPS equivalent on HIP), so only
    # bump this if you've actually measured a win on your hardware.
    workers_per_device: int = 1
    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout: float = 600.0
