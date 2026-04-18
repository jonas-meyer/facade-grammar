"""Application configuration.

``AppConfig()`` layers sources in decreasing precedence: init kwargs → env
vars (``FG_`` prefix, nested via double-underscore, e.g.
``FG_AREA__MIN_LON=4.88``) → ``.env`` file → YAML at
``config/default.yaml`` → model defaults.
"""

from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Bbox(NamedTuple):
    """WGS84 bounding box. NamedTuple so Hamilton's input type-check can isinstance-validate it."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class CacheConfig(BaseModel):
    """Hamilton cache-related settings."""

    output_dir: Path = Path("data/outputs")
    recompute_nodes: list[str] = Field(default_factory=list)


class AppConfig(BaseSettings):
    """Top-level configuration for the facade-grammar pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="FG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/default.yaml",
        extra="ignore",
    )

    area: Bbox
    cache: CacheConfig = Field(default_factory=CacheConfig)
    mapillary_token: SecretStr

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
