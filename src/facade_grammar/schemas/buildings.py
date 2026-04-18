"""Building record type."""

from pydantic import BaseModel, Field


class Building(BaseModel):
    """A single building from 3D BAG with its ground-plane footprint."""

    building_id: str
    footprint: list[tuple[float, float]]
    height_m: float = Field(ge=0)
