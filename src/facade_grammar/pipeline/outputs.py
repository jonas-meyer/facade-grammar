"""Final artifacts: grammar JSON, features table CSV, lattice JSON.

Three file writers dropped under ``cache.output_dir`` (default
``data/outputs/``). Each returns its output path so the Hamilton DAG can
track them as first-class artifacts and the CLI can echo them back.

Always recomputed (``@hamilton_cache(behavior="recompute")``): Hamilton's cache
would otherwise skip the write on a cache hit, leaving stale files on
disk whenever an upstream node was invalidated via ``recompute_nodes``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import get_args

import polars as pl
from hamilton.function_modifiers import cache as hamilton_cache
from hamilton.function_modifiers import tag
from pydantic import TypeAdapter

from facade_grammar.config import CacheConfig
from facade_grammar.schemas.buildings import (
    AuditRecord,
    FacadeFeatures,
    FacadeGrammar,
    FacadeLattice,
    FacadeMask,
    FeatureClass,
)

_GRAMMAR_LIST: TypeAdapter[list[FacadeGrammar]] = TypeAdapter(list[FacadeGrammar])
_LATTICE_LIST: TypeAdapter[list[FacadeLattice]] = TypeAdapter(list[FacadeLattice])


@hamilton_cache(behavior="recompute")
@tag(stage="outputs")
def grammar_json(
    facade_grammars: dict[str, FacadeGrammar],
    cache: CacheConfig,
) -> Path:
    out = cache.output_dir / "grammar.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_GRAMMAR_LIST.dump_json(list(facade_grammars.values()), indent=2))
    return out


_FEATURES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "building_id": pl.String,
    "photo_id": pl.String,
    "cls": pl.Enum(list(get_args(FeatureClass))),
    "x0": pl.Int64,
    "y0": pl.Int64,
    "x1": pl.Int64,
    "y1": pl.Int64,
    "score": pl.Float64,
    "x0_norm": pl.Float64,
    "y0_norm": pl.Float64,
    "x1_norm": pl.Float64,
    "y1_norm": pl.Float64,
    "building_year": pl.Int64,
    "facade_width_m": pl.Float64,
    "facade_height_m": pl.Float64,
}


@hamilton_cache(behavior="recompute")
@tag(stage="outputs")
def features_csv(
    facade_features: dict[str, FacadeFeatures],
    facade_grammars: dict[str, FacadeGrammar],
    facade_masks: dict[str, FacadeMask],
    cache: CacheConfig,
) -> Path:
    out = cache.output_dir / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _iter_feature_rows(facade_features, facade_grammars, facade_masks)
    pl.from_dicts(rows, schema=_FEATURES_SCHEMA).write_csv(out)
    return out


def _iter_feature_rows(
    facade_features: dict[str, FacadeFeatures],
    facade_grammars: dict[str, FacadeGrammar],
    facade_masks: dict[str, FacadeMask],
) -> Iterator[dict[str, object]]:
    for bid, ff in facade_features.items():
        fm = facade_masks.get(bid)
        g = facade_grammars.get(bid)
        if fm is None or g is None:
            continue
        fx0, fy0, fx1, fy1 = fm.facade_bbox_px
        fw, fh = fx1 - fx0, fy1 - fy0
        for inst in ff.instances:
            x0, y0, x1, y1 = inst.bbox
            yield {
                "building_id": bid,
                "photo_id": ff.photo_id,
                "cls": inst.cls,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "score": inst.score,
                "x0_norm": (x0 - fx0) / fw,
                "y0_norm": (y0 - fy0) / fh,
                "x1_norm": (x1 - fx0) / fw,
                "y1_norm": (y1 - fy0) / fh,
                "building_year": g.bag_construction_year,
                "facade_width_m": g.facade_width_m,
                "facade_height_m": g.facade_height_m,
            }


@hamilton_cache(behavior="recompute")
@tag(stage="outputs")
def lattices_json(
    facade_lattices: dict[str, FacadeLattice],
    cache: CacheConfig,
) -> Path:
    out = cache.output_dir / "lattices.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_LATTICE_LIST.dump_json(list(facade_lattices.values()), indent=2))
    return out


@hamilton_cache(behavior="recompute")
@tag(stage="outputs")
def audit_jsonl(
    audit_records: list[AuditRecord],
    cache: CacheConfig,
) -> Path:
    """One-line-per-facade outcome log, reconstructed every run.

    ``audit_records`` already carries the enriched per-facade data
    (including winner perpendicularity_deg); we just materialise it to disk.
    """
    out = cache.output_dir / "audit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(r.model_dump_json() + "\n" for r in audit_records))
    return out
