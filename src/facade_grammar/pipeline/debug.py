"""Debug visualization nodes (matplotlib), tagged ``stage=debug``."""

from pathlib import Path

import polars as pl
from hamilton.function_modifiers import cache, tag

from facade_grammar.schemas.buildings import (
    Building,
    Facade,
    FacadeFeatures,
    FacadeGrammar,
    FacadeLattice,
    FacadeMask,
)
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.viz.plots import (
    plot_area_map,
    plot_canal_selection,
    plot_facade_features_contact_sheet,
    plot_facade_grammar_contact_sheet,
    plot_facade_lattice_contact_sheet,
    plot_facade_mask_contact_sheet,
)


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


@cache(behavior="recompute")
@tag(stage="debug")
def canal_selection_map(
    raw_buildings: list[Building],
    raw_streets: pl.DataFrame,
    raw_waterways: pl.DataFrame,
    canal_facades: list[Facade],
    top_photos_per_facade: dict[str, list[PhotoMetadata]],
) -> Path:
    best = {bid: photos[0] for bid, photos in top_photos_per_facade.items()}
    return plot_canal_selection(
        buildings=raw_buildings,
        streets=raw_streets,
        waterways=raw_waterways,
        canal_facades=canal_facades,
        best_photo_per_facade=best,
        out_path=Path("data/debug/canal_selection_map.png"),
    )


@cache(behavior="recompute")
@tag(stage="debug")
def facade_mask_contact_sheet(facade_masks: dict[str, FacadeMask]) -> Path:
    return plot_facade_mask_contact_sheet(
        facade_masks=facade_masks,
        out_path=Path("data/debug/facade_mask_contact_sheet.png"),
    )


@cache(behavior="recompute")
@tag(stage="debug")
def facade_features_contact_sheet(facade_features: dict[str, FacadeFeatures]) -> Path:
    return plot_facade_features_contact_sheet(
        facade_features=facade_features,
        out_path=Path("data/debug/facade_features_contact_sheet.png"),
    )


@cache(behavior="recompute")
@tag(stage="debug")
def facade_grammar_contact_sheet(
    facade_grammars: dict[str, FacadeGrammar],
    facade_masks: dict[str, FacadeMask],
) -> Path:
    return plot_facade_grammar_contact_sheet(
        facade_grammars=facade_grammars,
        facade_masks=facade_masks,
        out_path=Path("data/debug/facade_grammar_contact_sheet.png"),
    )


@cache(behavior="recompute")
@tag(stage="debug")
def facade_lattice_contact_sheet(
    facade_lattices: dict[str, FacadeLattice],
    facade_masks: dict[str, FacadeMask],
) -> Path:
    return plot_facade_lattice_contact_sheet(
        facade_lattices=facade_lattices,
        facade_masks=facade_masks,
        out_path=Path("data/debug/facade_lattice_contact_sheet.png"),
    )
