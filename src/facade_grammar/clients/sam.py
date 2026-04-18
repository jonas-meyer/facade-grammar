"""SAM 3 HTTP client.

Thin sync wrapper around the ``sam-service`` ``POST /predict`` endpoint. Masks
arrive as base64-ed 1-bit PNGs; this module decodes them so pipeline nodes
never see the wire format. A single call carries one image + N prompts and
returns N aligned instance lists.
"""

import base64
import io
from typing import NamedTuple

import httpx
import numpy as np
from PIL import Image


class SamInstance(NamedTuple):
    mask: np.ndarray
    score: float
    box: tuple[float, float, float, float]


class SamPrompt(NamedTuple):
    text: str
    boxes: list[list[float]] | None = None


def segment(
    image_bytes: bytes,
    *,
    prompts: list[SamPrompt],
    base_url: str,
    timeout_s: float,
) -> list[list[SamInstance]]:
    """POST one image with N prompts; return N aligned instance lists."""
    payload: dict[str, object] = {
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "prompts": [
            {"text": p.text, **({"boxes": p.boxes} if p.boxes is not None else {})}
            for p in prompts
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{base_url.rstrip('/')}/predict", json=payload)
        resp.raise_for_status()
        body = resp.json()
    return [
        [
            SamInstance(
                mask=_decode_mask(mask_b64),
                score=score,
                box=(b[0], b[1], b[2], b[3]),
            )
            for mask_b64, score, b in zip(r["masks"], r["scores"], r["boxes"], strict=True)
        ]
        for r in body["results"]
    ]


def _decode_mask(mask_b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    return np.asarray(img, dtype=bool)
