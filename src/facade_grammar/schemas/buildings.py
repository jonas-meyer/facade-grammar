"""Building and facade record types."""

from typing import Literal

from pydantic import BaseModel, Field

FacadeClass = Literal["canal", "street", "other"]


class Building(BaseModel):
    """A single building from 3D BAG with its ground-plane footprint."""

    building_id: str
    footprint: list[tuple[float, float]]
    height_m: float = Field(ge=0)


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
