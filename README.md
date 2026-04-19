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

## SAM service (ROCm)

`services/sam/` is a small HTTP wrapper around SAM 3 (`facebook/sam3`) that
the pipeline calls over localhost. There's a ROCm Dockerfile for AMD GPUs
(tested on RX 6800 — warm `/predict` ~2 s/prompt amortised on a 4-prompt
request at 512², bf16).

**Prereqs:**

- Host kernel with AMDGPU + ROCm drivers (`/dev/kfd` and `/dev/dri/*` visible).
- Accept the license at <https://huggingface.co/facebook/sam3> — it's a
  gated repo.
- Authenticate once so the token lands in `~/.cache/huggingface/token`:

  ```sh
  uvx --from 'huggingface_hub[cli]' hf auth login
  ```

**Build and run:**

```sh
docker build -t sam-service -f services/sam/Dockerfile services/sam

docker run -d --name sam-service -p 8000:8000 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=8g \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  -v sam-hf-cache:/root/.cache/huggingface \
  sam-service

curl http://localhost:8000/health   # "ok"
curl http://localhost:8000/info     # device, workers
```

First launch downloads ~2.5 GB of weights into the `sam-hf-cache` volume;
subsequent starts are ~20 s.

**If Docker bridge networking is broken on your host** (e.g. Arch with
certain kernel builds — symptom: `failed to add the host <=> sandbox pair
interfaces: operation not supported`), swap `-p 8000:8000` for
`--network=host` on both `docker build` and `docker run`.
