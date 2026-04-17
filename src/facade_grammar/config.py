"""Application configuration.

``AppConfig()`` layers sources in decreasing precedence: init kwargs → env
vars (``FG_`` prefix, nested via double-underscore, e.g.
``FG_TEST_AREA__MIN_LON=4.88``) → ``.env`` file → YAML at
``config/default.yaml`` → model defaults.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class BboxConfig(BaseModel):
    """Geographic bounding box in WGS84 degrees."""

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

    test_area: BboxConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)

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
