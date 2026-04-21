"""facade-grammar CLI entry point.

Invoked via the ``facade-grammar`` console script registered under
``[project.scripts]`` in ``pyproject.toml``::

    uv run facade-grammar                # warm run, uses on-disk cache
    uv run facade-grammar --no-cache     # force a cold run; cache still gets written
"""

from pathlib import Path, PurePath
from typing import Annotated, Any

import typer
from hamilton import driver
from hamilton.caching.fingerprinting import hash_repr, hash_value
from hamilton.execution.executors import MultiThreadingExecutor, SynchronousLocalTaskExecutor
from hamilton.plugins import h_rich

from facade_grammar.config import AppConfig
from facade_grammar.pipeline import (
    audit,
    debug,
    grammar,
    ingestion,
    outputs,
    regularization,
    selection,
    spatial,
    vision,
)


@hash_value.register(PurePath)
def _hash_path(obj: PurePath, *_args: Any, **_kwargs: Any) -> str:
    return hash_repr(obj)


def main(
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Force every node (including @dataloader fetchers) to recompute.",
        ),
    ] = False,
) -> None:
    """Run the facade-grammar pipeline and emit the debug overlays."""
    cfg = AppConfig()  # ty: ignore[missing-argument]  # settings sources fill all fields

    dr = (
        driver.Builder()
        .with_modules(
            ingestion,
            spatial,
            selection,
            vision,
            audit,
            grammar,
            regularization,
            debug,
            outputs,
        )
        .with_cache(
            path=str(Path(cfg.cache.cache_dir)),
            # ``True`` forces recompute on every node — regular, @dataloader, and
            # @datasaver — which ``default_behavior="recompute"`` alone does not.
            recompute=True if no_cache else cfg.cache.recompute_nodes,
        )
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_local_executor(SynchronousLocalTaskExecutor())
        .with_remote_executor(MultiThreadingExecutor(max_tasks=cfg.sam.max_concurrency))
        .with_adapters(h_rich.RichProgressBar(run_desc="facade-grammar"))
        .build()
    )

    result = dr.execute(
        final_vars=[
            "area_map",
            "canal_selection_map",
            "facade_mask_contact_sheet",
            "facade_features_contact_sheet",
            "facade_grammar_contact_sheet",
            "facade_lattice_contact_sheet",
            "facade_grammars",
            "grammar_json",
            "features_csv",
            "lattices_json",
            "audit_jsonl",
        ],
        inputs={
            "area_bbox": cfg.area,
            "mapillary_token": cfg.mapillary_token,
            "ingestion": cfg.ingestion,
            "spatial": cfg.spatial,
            "selection": cfg.selection,
            "sam": cfg.sam,
            "grammar": cfg.grammar,
            "cache": cfg.cache,
        },
    )
    typer.echo(f"area_map: {result['area_map']}")
    typer.echo(f"canal_selection_map: {result['canal_selection_map']}")
    typer.echo(f"facade_mask_contact_sheet: {result['facade_mask_contact_sheet']}")
    typer.echo(f"facade_features_contact_sheet: {result['facade_features_contact_sheet']}")
    typer.echo(f"facade_grammar_contact_sheet: {result['facade_grammar_contact_sheet']}")
    typer.echo(f"facade_lattice_contact_sheet: {result['facade_lattice_contact_sheet']}")
    typer.echo(f"grammar_json: {result['grammar_json']}")
    typer.echo(f"features_csv: {result['features_csv']}")
    typer.echo(f"lattices_json: {result['lattices_json']}")
    typer.echo(f"audit_jsonl: {result['audit_jsonl']}")
    typer.echo(f"facade_grammars: {len(result['facade_grammars'])} building(s)")
    for bid, g in result["facade_grammars"].items():
        year = g.bag_construction_year or "?"
        bag_fc = "?" if g.bag_floor_count is None else g.bag_floor_count
        match = {True: "✓", False: "✗", None: "—"}[g.n_floors_matches_bag]
        gable = f"{g.gable_prominence_m:.1f}m" if g.gable_prominence_m is not None else "?"
        typer.echo(
            f"  {bid[-16:]}  yr={year}  roof={g.bag_roof_type or '?'}  "
            f"floors={g.n_floors}/{bag_fc}{match}  "
            f"cols={g.n_columns}  w={g.n_windows_used}  "
            f"facade={g.facade_width_m:.1f}x{g.facade_height_m:.1f}m  "
            f"gable={gable}"
        )


def cli() -> None:
    """Entry-point wrapper for ``[project.scripts]``."""
    typer.run(main)


if __name__ == "__main__":
    cli()
