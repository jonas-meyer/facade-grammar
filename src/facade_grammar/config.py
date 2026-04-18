"""Application configuration.

``AppConfig()`` layers sources in decreasing precedence: init kwargs → env
vars (``FG_`` prefix, nested via double-underscore, e.g.
``FG_AREA__MIN_LON=4.88``) → ``.env`` file → YAML at
``config/default.yaml`` → model defaults.
"""

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, HttpUrl, SecretStr
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


class SamConfig(BaseModel):
    """SAM 3 inference service settings and per-facade selection tunables."""

    service_url: HttpUrl = Field(
        default=HttpUrl("http://127.0.0.1:8000"),
        description="Base URL of the running sam-service (LitServe).",
    )
    facade_prompt: str = Field(
        default="building",
        description="SAM 3 phrase prompt for the target structure.",
    )
    occluder_prompts: list[str] = Field(
        default_factory=lambda: ["tree", "car"],
        description=(
            "Phrase prompts for things that occlude a canal facade from across-canal views. "
            "Masks are unioned before the bbox-occlusion metric is computed."
        ),
    )
    min_facade_score: float = Field(
        default=0.3,
        description="Reject a photo if SAM's top facade instance scores below this.",
    )
    http_timeout_s: float = Field(
        default=600.0,
        description=(
            "Per-request timeout to sam-service. fp32 MPS inference is slow and queues "
            "serialize on a single GPU; see reference_mps_sam3_fp32 memory for why fp32 "
            "is unavoidable on Apple Silicon."
        ),
    )
    mapillary_image_timeout_s: float = Field(
        default=60.0,
        description="Per-request timeout when downloading Mapillary thumb bytes.",
    )
    mask_dir: Path = Field(
        default=Path("data/masks"),
        description="On-disk location for selected facade + tree masks.",
    )
    pano_thumb_field: Literal["thumb_1024_url", "thumb_2048_url", "thumb_original_url"] = Field(
        default="thumb_2048_url",
        description="Mapillary Graph API thumb field fetched per-pano at SAM time.",
    )
    pano_view_size: tuple[int, int] = Field(
        default=(1280, 960),
        description="(W, H) pixels of the rectilinear view fed to SAM.",
    )
    pano_view_fov_deg: float = Field(
        default=90.0,
        description="Horizontal field of view of the reprojected pano view.",
    )
    bbox_margin_frac: float = Field(
        default=0.08,
        description="Safety margin expanding the projected footprint bbox on each side.",
    )
    max_concurrency: int = Field(
        default=2,
        description="Parallel in-flight HTTP calls to sam-service.",
    )
    max_buildings: int | None = Field(
        default=None,
        description="Cap on canal-front buildings segmented per run (dev aid; None = all).",
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
    sam: SamConfig = Field(default_factory=SamConfig)
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
