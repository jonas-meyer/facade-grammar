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

from facade_grammar.schemas.buildings import FeatureClass


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
        default_factory=lambda: [
            "tree",
            "car",
            "lamppost",
            "boat",
            "scaffolding",
        ],
        description=(
            "Phrase prompts for things that occlude a canal facade from across-canal views. "
            "Masks are unioned before the bbox-occlusion metric is computed. Amsterdam-"
            "specific additions: lampposts on canal quays, moored boats in the canal, and "
            "scaffolding during renovation cycles."
        ),
    )
    feature_prompts: list[FeatureClass] = Field(
        default=["window", "door", "chimney", "balcony", "hoist_beam"],
        description=(
            "Per-facade sub-feature SAM prompts. Only classes SAM 3 can "
            "plausibly text-ground on 2D facade views — 'gable'/'roof peak'/"
            "'pediment' empirically return ~zero masks at the default score "
            "threshold on canal-house photos (gable prominence is derived "
            "from BAG roof heights instead). 'hoist_beam' is the uniquely "
            "Dutch hijsbalk; text-grounding quality untested at scale."
        ),
    )
    feature_min_score: float = Field(
        default=0.3,
        description="Reject individual feature instances below this SAM score.",
    )
    features_dir: Path = Field(
        default=Path("data/features"),
        description="Per-building directory for class-union feature mask PNGs.",
    )
    min_facade_score: float = Field(
        default=0.3,
        description="Reject a photo if SAM's top facade instance scores below this.",
    )
    min_mask_bbox_overlap: float = Field(
        default=0.5,
        description=(
            "Fraction of SAM's facade-mask pixels that must fall inside the "
            "projected footprint bbox (computed per-pano from BAG coords). "
            "Below this threshold the candidate is rejected — SAM locked "
            "onto a neighbouring building instead of our target. Only applies "
            "to panos (perspective photos have no projected_bbox; they rely "
            "on perspective_max_dist_m being tight enough that the target "
            "dominates the frame)."
        ),
    )
    max_occluder_ratio: float = Field(
        default=0.4,
        description=(
            "Occluder mask fraction of the facade bbox above which a candidate "
            "is excluded from winning. Among candidates below this threshold "
            "the best-ranked (most head-on, winter-leafless, etc.) wins; if "
            "every candidate exceeds it we still pick the best-ranked one "
            "rather than drop the facade. So occlusion is a hard filter and "
            "perpendicularity is the fine-grained selector among survivors."
        ),
    )
    http_timeout_s: float = Field(
        default=600.0,
        description=(
            "Per-request timeout to sam-service. fp32 MPS inference is slow and queues "
            "serialize on a single GPU; see reference_mps_sam3_fp32 memory for why fp32 "
            "is unavoidable on Apple Silicon."
        ),
    )
    http_retries: int = Field(
        default=3,
        description=(
            "Retry transient sam-service failures (transport errors / 5xx) up to this "
            "many times per request. 4xx responses are not retried."
        ),
    )
    http_retry_base_delay_s: float = Field(
        default=1.0,
        description="Exponential backoff base (1s, 2s, 4s, …) for sam-service retries.",
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
        default="thumb_original_url",
        description=(
            "Mapillary Graph API thumb field fetched per-pano at SAM time. "
            "Original gives the cleanest downsample for reprojection — the "
            "~50% extra bytes over thumb_2048 is worth it at 45° FoV."
        ),
    )
    pano_view_size: tuple[int, int] = Field(
        default=(1280, 960),
        description="(W, H) pixels of the rectilinear view fed to SAM.",
    )
    pano_view_fov_deg: float = Field(
        default=45.0,
        description=(
            "Horizontal FoV of the reprojected pano view. SAM 3 resizes inputs "
            "to 1008² internally, so effective facade pixel density is set by "
            "FoV x source pano resolution, not by pano_view_size."
        ),
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
    perspective_max_dist_m: float = Field(
        default=25.0,
        description=(
            "Stricter distance cap for non-pano (perspective) photos. Panos "
            "get a projected_bbox prompt that locks SAM onto the specific "
            "target building, so they can afford to sit 40-60m away; "
            "perspective photos have no such anchor and SAM picks whatever "
            "'building' is largest in frame — so the target has to dominate "
            "the view, which means close. 25m ≈ across-a-canal-plus-sidewalk "
            "and keeps cross-canal panos untouched."
        ),
    )
    bearing_tol_deg: float = Field(
        default=25.0,
        description=(
            "Perspective-photo only: photo bearing must be within this many "
            "degrees of (facade-normal + 180). Panos bypass this because they "
            "get reprojected at vision-time — use max_perpendicularity_deg to "
            "filter oblique panos."
        ),
    )
    max_perpendicularity_deg: float = Field(
        default=20.0,
        description=(
            "Reject photos whose position is more than this many degrees off "
            "the facade-normal axis (0 = dead head-on, 90 = looking along the "
            "facade). Applies to both panos and perspective photos, so it's "
            "the only filter that protects against wildly oblique pano "
            "captures driven along the canal. 30° is empirically the last "
            "angle a homography rectification could plausibly recover."
        ),
    )
    top_k: int = Field(
        default=5,
        description=(
            "Candidates carried per facade. Across-canal photos sort ahead of same-side; "
            "downstream (SAM tree-occlusion filtering, etc.) picks among them. Bumped "
            "from 3 to 5 so candidates have fallback room when the top couple are "
            "heavily occluded."
        ),
    )
    winter_months: set[int] = Field(
        default={12, 1, 2, 3},
        description=(
            "Months (1-12) treated as 'leafless' for facade visibility. "
            "Amsterdam canal-side deciduous trees are bare roughly Dec-Mar. "
            "Photos captured in these months rank ahead of summer ones."
        ),
    )
    winter_only: bool = Field(
        default=False,
        description=(
            "When True, hard-reject photos captured outside winter_months. "
            "Trades yield for clean (leafless) views. Default False keeps "
            "winter as a tiebreaker only; enable via FG_SELECTION__WINTER_ONLY=true "
            "for runs where heavy summer foliage is blocking too many facades."
        ),
    )
    pano_only: bool = Field(
        default=False,
        description=(
            "When True, hard-reject perspective photos and keep only panos. "
            "Panos are reprojected toward the facade midpoint so the view is "
            "guaranteed to be centred on the target; perspective photos have "
            "a fixed FoV and can pass our position/bearing filters while the "
            "target sits mostly outside the frame (dashcam driving-past "
            "case). Enable via FG_SELECTION__PANO_ONLY=true."
        ),
    )
    min_quality_score: float | None = Field(
        default=None,
        description=(
            "Hard-reject photos below this Mapillary quality_score (0-1). "
            "None = no threshold; photos without a score are never rejected "
            "on this basis."
        ),
    )


class GrammarConfig(BaseModel):
    """Grammar-extraction thresholds.

    Floor/column clustering is a 1D peak-find on a gaussian-smoothed histogram
    of window centres — robust to overlapping detections and adjacent rows
    whose window heights are similar to the inter-row gap (the failure mode
    of a pure gap-based splitter on Dutch canal houses).
    """

    histogram_n_bins: int = Field(
        default=40,
        description=(
            "Bin count over the normalised [0,1] facade axis. 40 → 0.025 "
            "resolution; windows span a few bins, floor/column bands dozens."
        ),
    )
    smooth_sigma_bins: float = Field(
        default=1.5,
        description=(
            "Gaussian smoothing sigma in bin units applied to the density "
            "histogram before peak-finding."
        ),
    )
    min_peak_prominence: float = Field(
        default=0.2,
        description=(
            "Ignore histogram peaks below this fraction of the max bin - "
            "filters isolated noise detections."
        ),
    )
    min_windows: int = Field(
        default=2,
        description="Skip grammar fitting if fewer windows were detected — too little signal.",
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
    grammar: GrammarConfig = Field(default_factory=GrammarConfig)
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
