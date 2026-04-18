"""Building and facade record types."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

FacadeClass = Literal["canal", "street", "other"]


class Building(BaseModel):
    """A single building from 3D BAG with its ground-plane footprint."""

    building_id: str
    footprint: list[tuple[float, float]]
    height_m: float = Field(ge=0)
    ground_altitude_m: float


class Facade(BaseModel):
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


class FacadeMask(BaseModel):
    """SAM 3 segmentation result attached to the chosen photo for a canal facade."""

    building_id: str
    photo_id: str
    view_path: Path
    mask_path: Path
    occluder_mask_path: Path
    facade_score: float = Field(ge=0, le=1)
    occluder_ratio: float = Field(ge=0)
    projected_bbox: tuple[int, int, int, int] | None = None
