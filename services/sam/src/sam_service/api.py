"""LitAPI + Pydantic request/response models for the SAM service.

A single request carries an image + a list of prompts (each a text phrase plus
optional box hints). SAM runs once per prompt; the response holds one result
per prompt, aligned to the request order — so a caller can ask for facade +
tree in one round-trip.
"""

import base64
import io
from typing import Any

import litserve as ls
from PIL import Image
from pydantic import BaseModel, Field

from sam_service.config import ServiceConfig
from sam_service.model import SamBackend


class Prompt(BaseModel):
    text: str = Field(description="Phrase prompt, e.g. 'canal house', 'tree'.")
    boxes: list[list[float]] | None = Field(
        default=None,
        description="Shape [num_boxes, 4] — xyxy image pixel coords.",
    )


class SegmentRequest(BaseModel):
    image_b64: str = Field(description="Base64-encoded image bytes (any Pillow-readable format).")
    prompts: list[Prompt] = Field(min_length=1)
    threshold: float = Field(default=0.3)
    mask_threshold: float = Field(default=0.5)


class PromptResult(BaseModel):
    masks: list[str] = Field(description="Base64-encoded 1-bit PNGs, one per predicted instance.")
    scores: list[float]
    boxes: list[list[float]]


class SegmentResponse(BaseModel):
    results: list[PromptResult] = Field(description="Aligned with the request's prompts list.")


class SamAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        cfg = ServiceConfig()
        self.backend = SamBackend(cfg.model_id, device=device, dtype=cfg.dtype)

    def decode_request(self, request: SegmentRequest) -> dict[str, Any]:
        return {
            "image": _decode_image(request.image_b64),
            "prompts": [(p.text, p.boxes) for p in request.prompts],
            "threshold": request.threshold,
            "mask_threshold": request.mask_threshold,
        }

    def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return inputs

    def predict(self, x: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.backend.segment(**item) for item in x]

    def unbatch(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return output

    def encode_response(self, output: dict[str, Any]) -> SegmentResponse:
        return SegmentResponse(**output)


def _decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
