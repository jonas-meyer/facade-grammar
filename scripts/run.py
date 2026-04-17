"""Hamilton driver entrypoint: build the DAG and print the node count."""

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
    nodes = dr.list_available_variables()
    print(f"Hamilton driver built successfully ({len(nodes)} nodes registered).")


if __name__ == "__main__":
    main()
