"""Building, facade, and extracted-feature records."""

from typing import Literal

from pydantic import BaseModel, Field

FacadeClass = Literal["canal", "street", "party_wall", "other"]


class Building(BaseModel):
    """A single building from 3D BAG with its ground-plane footprint."""

    building_id: str
    footprint: list[tuple[float, float]]
    height_m: float = Field(ge=0)


class Facade(BaseModel):
    """One edge of a building footprint, classified by what it faces."""

    building_id: str
    edge_start: tuple[float, float]
    edge_end: tuple[float, float]
    classification: FacadeClass


class BuildingFeatures(BaseModel):
    """Per-building features extracted from segmentation + geometry."""

    building_id: str
    num_floors: int = Field(ge=1)
    windows_per_floor: float = Field(ge=0)
    window_aspect_ratio: float = Field(gt=0)
    facade_width_m: float = Field(gt=0)
    alignment_score: float = Field(ge=0, le=1)
