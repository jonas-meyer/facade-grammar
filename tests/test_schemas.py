"""Round-trip tests for every Pydantic schema in ``facade_grammar.schemas``."""

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, HttpUrl

from facade_grammar.schemas.buildings import Building, BuildingFeatures, Facade
from facade_grammar.schemas.grammar import DistributionSummary, GrammarDistributions
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.schemas.segmentation import Mask, MaskSet

_SUMMARY = DistributionSummary(mean=3.4, std=0.7, count=20, p05=2.0, p50=3.0, p95=5.0)

_SAMPLES: list[BaseModel] = [
    Building(
        building_id="NL.IMBAG.Pand.0123456789",
        footprint=[(4.88, 52.37), (4.881, 52.37), (4.881, 52.371), (4.88, 52.371)],
        height_m=12.5,
    ),
    Facade(
        building_id="NL.IMBAG.Pand.0123456789",
        edge_start=(4.88, 52.37),
        edge_end=(4.881, 52.37),
        classification="canal",
    ),
    BuildingFeatures(
        building_id="NL.IMBAG.Pand.0123456789",
        num_floors=4,
        windows_per_floor=3.0,
        window_aspect_ratio=0.6,
        facade_width_m=6.2,
        alignment_score=0.87,
    ),
    PhotoMetadata(
        photo_id="mly-abc-123",
        lon=4.8812,
        lat=52.3701,
        bearing_deg=42.0,
        captured_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        url=HttpUrl("https://example.com/photo.jpg"),
    ),
    MaskSet(
        prompt="window",
        image_width=640,
        image_height=480,
        masks=[
            Mask(score=0.91, bbox_xyxy=(10, 20, 100, 110), area_px=8100),
            Mask(score=0.73, bbox_xyxy=(200, 20, 290, 110), area_px=8100),
        ],
    ),
    GrammarDistributions(
        num_floors=_SUMMARY,
        windows_per_floor=_SUMMARY,
        window_aspect_ratio=_SUMMARY,
        facade_width_m=_SUMMARY,
        alignment_score=_SUMMARY,
        sample_size=20,
    ),
]


@pytest.mark.parametrize("sample", _SAMPLES, ids=lambda m: type(m).__name__)
def test_roundtrip(sample: BaseModel) -> None:
    assert type(sample).model_validate(sample.model_dump(mode="json")) == sample
