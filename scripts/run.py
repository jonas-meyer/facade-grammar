"""Hamilton driver entrypoint for facade-grammar."""

from pathlib import PurePath
from typing import Any

from hamilton import driver
from hamilton.caching.fingerprinting import hash_repr, hash_value
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


def main() -> None:
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
            path="data/.hamilton_cache",
            recompute=cfg.cache.recompute_nodes,
            default_behavior="default",
        )
        .with_adapters(h_rich.RichProgressBar(run_desc="facade-grammar"))
        .build()
    )

    result = dr.execute(
        final_vars=["area_map"],
        inputs={
            "area_bbox": cfg.area,
            "mapillary_token": cfg.mapillary_token,
        },
    )
    print(f"area_map: {result['area_map']}")


if __name__ == "__main__":
    main()
