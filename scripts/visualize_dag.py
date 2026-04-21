"""Render the full Hamilton DAG to ``docs/dag.png``.

The single source of truth for the pipeline's stage story: ingestion →
spatial → selection → vision → audit → grammar → regularization → debug +
outputs. Nodes group by their ``stage=`` tag; edges show data flow.
Cache-boundary intent (e.g. ``audit`` downstream of ``vision`` so new audit
fields don't invalidate the SAM cache) reads directly off the layout.
"""

from pathlib import Path

from hamilton import driver

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


def main() -> None:
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
