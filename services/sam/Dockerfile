# syntax=docker/dockerfile:1.7
#
# ROCm build of sam-service for AMD GPUs (RX 6800 / MI series / etc).
#
#   Build: docker build -t sam-service -f services/sam/Dockerfile services/sam
#   Run:   docker run -d --name sam-service -p 8000:8000 \
#            --device=/dev/kfd --device=/dev/dri --group-add video \
#            --ipc=host --shm-size=8g \
#            -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
#            -v sam-hf-cache:/root/.cache/huggingface \
#            sam-service
#
# facebook/sam3 is a gated HF repo: accept its license and authenticate
# locally first (`uvx --from 'huggingface_hub[cli]' hf auth login`). First
# run downloads ~2.5 GB of weights into the mounted HF cache.
# Pick the pytorch_release_2.10.0 tag explicitly to avoid :latest drift;
# the image ships torch + torchvision pre-built against ROCm, so we install
# everything else on top with --no-deps for torch.

FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# ROCm torch ships inside /opt/venv — not the Debian system python (which is
# PEP 668 externally-managed and has no torch). Point uv at the venv so our
# installs land next to the existing torch 2.10.0+rocm7.2.2.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Install runtime deps first — depends only on pyproject.toml, so code edits
# in src/ don't bust this layer. Skip torch/torchvision (ROCm builds in
# /opt/venv shouldn't be replaced by PyPI CPU/CUDA wheels).
COPY pyproject.toml ./
RUN uv pip install --no-cache-dir \
    "transformers>=5.5,<6" \
    "pillow>=11.0,<12" \
    "litserve>=0.2.17,<0.3" \
    "pydantic>=2.13,<3" \
    "pydantic-settings>=2.13,<3" \
    "numpy>=2.2,<3"

COPY src ./src
RUN uv pip install --no-cache-dir --no-deps .

ENV SAM_ACCELERATOR=cuda \
    SAM_DEVICES=1 \
    SAM_HOST=0.0.0.0 \
    HIP_VISIBLE_DEVICES=0

EXPOSE 8000

# Model load takes ~30–60s on first run, so start-period is generous.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sam-service"]
