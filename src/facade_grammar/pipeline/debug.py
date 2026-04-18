"""Debug visualization nodes (matplotlib), tagged ``stage=debug``."""

from pathlib import Path

import polars as pl
from hamilton.function_modifiers import cache, tag

from facade_grammar.schemas.buildings import Building
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.viz.plots import plot_area_map


@cache(behavior="recompute")
@tag(stage="debug")
def area_map(
    raw_buildings: list[Building],
    raw_streets: pl.DataFrame,
    raw_waterways: pl.DataFrame,
    raw_photo_metadata: list[PhotoMetadata],
) -> Path:
    return plot_area_map(
        buildings=raw_buildings,
        streets=raw_streets,
        waterways=raw_waterways,
        photos=raw_photo_metadata,
        out_path=Path("data/debug/area_map.png"),
    )
