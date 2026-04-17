# facade-grammar

Procedural grammar extraction from Dutch canal houses, by combining 3D BAG
cadastral data, OpenStreetMap geometry, Mapillary street-level photos, and
SAM 3 segmentation into a single Hamilton pipeline.

Output: a JSON of learned distributions describing canal-house structure,
suitable for driving a procedural generator.

## Quickstart

```sh
uv sync
uv run python scripts/run.py
```

## Development checks

```sh
uv run ruff check
uv run ty check
uv run pytest
```
