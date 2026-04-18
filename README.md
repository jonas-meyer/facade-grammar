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
uv run facade-grammar              # warm run, uses on-disk cache
uv run facade-grammar --no-cache   # force a cold run
```

Writes two PNGs under `data/debug/`: `area_map.png` (buildings, streets,
canals, and all photo locations overlaid) and `canal_selection_map.png`
(canal facades with the selected Mapillary photo per building).
