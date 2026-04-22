"""Building and facade record types."""

from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from facade_grammar.schemas.base import FrozenModel

FacadeClass = Literal["canal", "street", "other"]


class Building(FrozenModel):
    """A single building from 3D BAG with its ground-plane footprint.

    ``height_m`` is roof ridge above ground; ``eaves_height_m`` is the 70th-
    percentile roof point above ground (close to the eaves). Their difference
    is the gable prominence — a direct roof-shape signal independent of SAM.
    """

    building_id: str
    footprint: list[tuple[float, float]]
    height_m: float = Field(ge=0)
    eaves_height_m: float | None = Field(default=None, ge=0)
    ground_altitude_m: float
    floor_count: int | None = Field(default=None, ge=1)
    roof_type: str | None = None
    construction_year: int | None = Field(default=None, ge=1000, le=2100)
    footprint_area_m2: float | None = Field(default=None, ge=0)


class Facade(FrozenModel):
    """One edge of a building footprint, classified by what it faces.

    ``normal_deg`` is the compass azimuth of the outward-pointing facade normal
    (0 = north, 90 = east). Edge coords are in WGS84 to match ``Building.footprint``;
    the classifier computes geometry internally in EPSG:28992 and reprojects back.
    """

    building_id: str
    edge_start: tuple[float, float]
    edge_end: tuple[float, float]
    classification: FacadeClass
    normal_deg: float = Field(ge=0, lt=360)

    @property
    def midpoint(self) -> tuple[float, float]:
        """WGS84 midpoint of the facade edge."""
        return (
            (self.edge_start[0] + self.edge_end[0]) / 2,
            (self.edge_start[1] + self.edge_end[1]) / 2,
        )


class FacadeMask(FrozenModel):
    """SAM 3 segmentation result attached to the chosen photo for a canal facade."""

    building_id: str
    photo_id: str
    view_path: Path
    mask_path: Path
    occluder_mask_path: Path
    facade_score: float = Field(ge=0, le=1)
    occluder_ratio: float = Field(ge=0)
    projected_bbox: tuple[int, int, int, int] | None = None
    facade_bbox_px: tuple[int, int, int, int]


FeatureClass = Literal["window", "door", "gable", "floor", "chimney", "balcony", "hoist_beam"]

# Multi-word concepts need a human phrase SAM 3 can text-ground.
FEATURE_CLASS_PROMPT: dict[FeatureClass, str] = {
    "window": "window",
    "door": "door",
    "gable": "gable",
    "floor": "floor",
    "chimney": "chimney",
    "balcony": "balcony",
    "hoist_beam": "hoist beam",
}


class FeatureInstance(FrozenModel):
    cls: FeatureClass
    score: float = Field(ge=0, le=1)
    bbox: tuple[int, int, int, int]


class FacadeFeatures(FrozenModel):
    """Per-class SAM 3 segmentation results for sub-features of a chosen facade."""

    building_id: str
    photo_id: str
    view_path: Path
    class_mask_paths: dict[FeatureClass, Path]
    instances: list[FeatureInstance]


class LatticeCell(FrozenModel):
    """One cell of a regularised facade lattice.

    ``cls`` is the dominant feature class whose bbox centre landed in this
    cell, or ``"wall"`` if nothing did. ``source_bbox_norm`` is that winning
    detection's bbox in facade-normalised coords, or None for wall-fills.
    """

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    cls: FeatureClass | Literal["wall"]
    score: float = Field(ge=0, le=1)
    source_bbox_norm: tuple[float, float, float, float] | None = None


class FacadeLattice(FrozenModel):
    """Regularised rectangle-grid representation of one facade.

    Snaps the per-facade feature detections onto the row/column structure
    found by window-centre clustering and emits a dense grid
    (``n_rows x n_cols`` cells, wall-filled where nothing was detected).
    This is the representation
    any future MDL grammar induction or FaçAID-style parser will consume.
    Coordinates are normalised to the facade bbox (0 = top/left, 1 =
    bottom/right).
    """

    building_id: str
    n_rows: int = Field(ge=1)
    n_cols: int = Field(ge=1)
    row_boundaries: list[float]  # length n_rows + 1
    col_boundaries: list[float]  # length n_cols + 1
    cells: list[LatticeCell]  # exactly n_rows * n_cols entries

    @model_validator(mode="after")
    def _check_dense_grid(self) -> Self:
        if len(self.row_boundaries) != self.n_rows + 1:
            raise ValueError(
                f"row_boundaries has {len(self.row_boundaries)} entries, "
                f"expected n_rows+1 = {self.n_rows + 1}"
            )
        if len(self.col_boundaries) != self.n_cols + 1:
            raise ValueError(
                f"col_boundaries has {len(self.col_boundaries)} entries, "
                f"expected n_cols+1 = {self.n_cols + 1}"
            )
        expected = self.n_rows * self.n_cols
        if len(self.cells) != expected:
            raise ValueError(
                f"cells has {len(self.cells)} entries, expected n_rows*n_cols = {expected}"
            )
        return self


AuditStatus = Literal["ok", "no_candidates", "error"]


class AuditRecord(FrozenModel):
    """Per-facade outcome record, emitted by every parallel worker.

    One of these rides alongside every ``FacadeWork`` so the full audit log
    can be reconstructed deterministically from cached per-facade results —
    no thread-locked file appends from within workers. ``ts`` is set when
    the record is first created, so cache-hit entries keep their original
    timestamp rather than picking up the resumed-run's clock.

    ``perpendicularity_deg`` and ``occluder_ratio`` are joined post-hoc
    (see ``pipeline.selection.audit_records``) so they do NOT invalidate
    ``chosen_work_for_facade``'s cache. Both are ``None`` for non-"ok"
    outcomes or when the winner's photo has been pruned from metadata.
    """

    building_id: str
    status: AuditStatus
    photo_id: str | None = None
    reason: str | None = None
    ts: AwareDatetime
    perpendicularity_deg: float | None = None
    occluder_ratio: float | None = None


class FacadeWork(FrozenModel):
    """One parallel worker's output: audit record + (on success) facade mask
    and its feature instances.

    Candidate screening and feature segmentation were originally two separate
    Parallelizable/Collect passes — one to pick the best candidate photo, a
    second to re-encode it for window/door prompts. Bundling every candidate's
    SAM call with the feature prompts lets a single worker keep the winner's
    ViT output in memory: 1 ViT encode per candidate instead of (candidates + 1).

    ``audit`` is always present and describes the outcome; ``mask`` and
    ``features`` are only populated when ``audit.status == "ok"``.
    """

    audit: AuditRecord
    mask: FacadeMask | None = None
    features: FacadeFeatures | None = None


class FacadeGrammar(FrozenModel):
    """Parametric split-grammar fit to one facade's detected features.

    All y / x / width / height values are normalised to the facade bbox —
    0,0 is top-left, 1,1 is bottom-right, widths are fractions of facade
    width, heights of facade height. That lets grammars be pooled across
    buildings of different image sizes.

    Floor/column counts come from histogram-peak 1D clustering of window bbox
    centres; the gable region is the strip above the topmost floor band.
    """

    building_id: str
    photo_id: str

    n_floors: int = Field(ge=1)
    n_columns: int = Field(ge=0)
    # Window-row centres — cluster means of detected window y-coordinates.
    # These pass through the middle of the windows, not between floors.
    window_row_y_centers: list[float]
    # Horizontal split lines between adjacent floors (midpoints between
    # consecutive window rows). Always ``n_floors - 1`` entries.
    floor_split_ys: list[float]
    column_x_centers: list[float]

    window_w_mean: float = Field(ge=0, le=1)
    window_w_std: float = Field(ge=0)
    window_h_mean: float = Field(ge=0, le=1)
    window_h_std: float = Field(ge=0)

    door_floor_idx: int | None = None
    door_x_center: float | None = Field(default=None, ge=0, le=1)

    gable_y_end: float | None = Field(default=None, ge=0, le=1)

    n_windows_used: int = Field(ge=0)
    n_doors_used: int = Field(ge=0)

    # --- Real-world anchors (meters). ``facade_width_m`` is the WGS84 geodesic
    # length of the canal-facing edge; ``facade_height_m`` is BAG's roof-ridge
    # above ground. Denormalised window dims and floor height fall out.
    facade_width_m: float = Field(ge=0)
    facade_height_m: float = Field(ge=0)
    window_w_mean_m: float = Field(ge=0)
    window_h_mean_m: float = Field(ge=0)
    floor_height_m: float = Field(ge=0)
    # Gable prominence from BAG ridge - eaves heights (``b3_h_max - b3_h_70p``).
    # Independent of SAM; catches step-gable / bell-gable / flat-roof.
    gable_prominence_m: float | None = Field(default=None, ge=0)

    # --- BAG structural context, propagated for aggregation + QA.
    bag_floor_count: int | None = Field(default=None, ge=1)
    bag_roof_type: str | None = None
    bag_construction_year: int | None = Field(default=None, ge=1000, le=2100)
    # True/False if BAG floor count is known and (dis)agrees with our
    # image-clustering n_floors; None if BAG didn't record one.
    n_floors_matches_bag: bool | None = None
