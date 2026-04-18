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
    debug,
    grammar,
    ingestion,
    outputs,
    per_building,
    projection,
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
            projection,
            vision,
            per_building,
            grammar,
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
        final_vars=["area_map", "canal_selection_map", "facade_mask_contact_sheet"],
        inputs={
            "area_bbox": cfg.area,
            "mapillary_token": cfg.mapillary_token,
            "ingestion": cfg.ingestion,
            "spatial": cfg.spatial,
            "selection": cfg.selection,
            "sam": cfg.sam,
        },
    )
    typer.echo(f"area_map: {result['area_map']}")
    typer.echo(f"canal_selection_map: {result['canal_selection_map']}")
    typer.echo(f"facade_mask_contact_sheet: {result['facade_mask_contact_sheet']}")


def cli() -> None:
    """Entry-point wrapper for ``[project.scripts]``."""
    typer.run(main)


if __name__ == "__main__":
    cli()
