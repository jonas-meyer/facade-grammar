"""SAM 3 segmentation output records."""

from pydantic import BaseModel, Field


class Mask(BaseModel):
    """A single SAM 3 mask with its confidence and bounding box."""

    score: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int = Field(ge=0)


class MaskSet(BaseModel):
    """All masks returned for a given image + prompt."""

    prompt: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    masks: list[Mask]
