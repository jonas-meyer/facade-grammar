"""Render the full Hamilton DAG to ``docs/dag.png``."""

from pathlib import Path

from hamilton import driver

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


def main() -> None:
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
        .build()
    )
    out_path = Path("docs/dag.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dr.display_all_functions(
        output_file_path=str(out_path),
        orient="LR",
        deduplicate_inputs=True,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
