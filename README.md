# facade-grammar

A Hamilton pipeline that extracts procedural grammars from real Dutch canal
houses by combining 3D BAG cadastral data, OpenStreetMap geometry, Mapillary
street-level photos, and SAM 3 segmentation.

## Install

```sh
uv sync
cp .env.example .env
```

Set `FG_MAPILLARY_TOKEN` in `.env`. Get one at
<https://www.mapillary.com/dashboard/developers>.

## Run

```sh
uv run python scripts/run.py
```

Writes `data/debug/area_map.png`: buildings, streets, canals, and photo
locations overlaid for the bbox in `config/default.yaml`.
