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

    cache_dir: Path = Field(
        default=Path("data/.hamilton_cache"),
        description="On-disk location of Hamilton's cache stores.",
    )
    output_dir: Path = Field(
        default=Path("data/outputs"),
        description="Where final artifacts (grammar JSON, features CSV) are written.",
    )
    recompute_nodes: list[str] = Field(
        default_factory=list,
        description=(
            "Extra node names forced to RECOMPUTE alongside the decorator-level cache policy."
        ),
    )


class IngestionConfig(BaseModel):
    """Settings for the raw_* dataloader nodes."""

    photo_fetch_buffer_m: float = Field(
        default=50.0,
        description=(
            "Expand the Mapillary bbox outward by this many metres so edge buildings "
            "find photos captured just outside the area. 3D BAG and OSM already "
            "overfetch (Overpass returns whole ways), so only Mapillary needs it."
        ),
    )


class SpatialConfig(BaseModel):
    """Thresholds for edge classification in ``pipeline.spatial``."""

    canal_near_m: float = Field(
        default=15.0,
        description="Max distance (m) from edge midpoint to waterway for canal classification.",
    )
    street_near_m: float = Field(
        default=10.0,
        description="Max distance (m) from an edge midpoint to a street for street classification.",
    )
    forward_cos_min: float = Field(
        default=0.5,
        description=(
            "Outward normal must point within this cosine of the nearest feature "
            "(cos 60deg = 0.5) for an edge to classify. Blocks side walls of narrow "
            "canal houses from inheriting their neighbour's canal proximity."
        ),
    )


class SelectionConfig(BaseModel):
    """Thresholds for photo selection in ``pipeline.selection``."""

    photo_min_dist_m: float = Field(
        default=3.0,
        description="Reject photos closer than this to the facade line (noise / lens distortion).",
    )
    photo_max_dist_m: float = Field(
        default=60.0,
        description="Reject photos farther than this to the facade line.",
    )
    bearing_tol_deg: float = Field(
        default=35.0,
        description="Photo bearing must be within this many degrees of (facade-normal + 180).",
    )
    top_k: int = Field(
        default=3,
        description=(
            "Candidates carried per facade. Across-canal photos sort ahead of same-side; "
            "downstream (SAM tree-occlusion filtering, etc.) picks among them."
        ),
    )


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
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
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
