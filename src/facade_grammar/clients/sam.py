"""SAM 3 HTTP client.

Thin sync wrapper around the ``sam-service`` ``POST /predict`` endpoint. Masks
arrive as base64-ed 1-bit PNGs; this module decodes them so pipeline nodes
never see the wire format. A single call carries one image + N prompts and
returns N aligned instance lists.
"""

import base64
import io
import logging
from typing import NamedTuple

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

BoolMask = NDArray[np.bool_]


class SamInstance(NamedTuple):
    mask: BoolMask
    score: float
    box: tuple[float, float, float, float]


class SamPrompt(NamedTuple):
    text: str
    boxes: list[list[float]] | None = None


class _PredictPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    boxes: list[list[float]] | None = None


class _PredictRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_b64: str
    prompts: list[_PredictPrompt]


class _PredictResult(BaseModel):
    """One prompt's worth of SAM output: three aligned arrays."""

    model_config = ConfigDict(frozen=True)

    masks: list[str]
    scores: list[float]
    boxes: list[list[float]]


_RESULTS_ADAPTER: TypeAdapter[list[_PredictResult]] = TypeAdapter(list[_PredictResult])


def segment(
    image_bytes: bytes,
    *,
    prompts: list[SamPrompt],
    base_url: str,
    timeout_s: float,
    max_attempts: int = 1,
    retry_base_delay_s: float = 1.0,
) -> list[list[SamInstance]]:
    """POST one image with N prompts; return N aligned instance lists.

    Retries transport errors and 5xx with exponential backoff; 4xx (caller
    bug) is not retried. ``max_attempts=1`` disables retrying entirely.
    """
    request = _PredictRequest(
        image_b64=base64.b64encode(image_bytes).decode("ascii"),
        prompts=[_PredictPrompt(text=p.text, boxes=p.boxes) for p in prompts],
    )
    url = f"{base_url.rstrip('/')}/predict"
    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=retry_base_delay_s, exp_base=2),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    for attempt in retrying:
        with attempt, httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=request.model_dump(exclude_none=True))
            resp.raise_for_status()
            results = _RESULTS_ADAPTER.validate_python(resp.json()["results"])
            return [_decode_result(r) for r in results]
    raise RuntimeError("unreachable")


def _decode_result(result: _PredictResult) -> list[SamInstance]:
    return [
        SamInstance(
            mask=_decode_mask(mask_b64),
            score=score,
            box=(box[0], box[1], box[2], box[3]),
        )
        for mask_b64, score, box in zip(result.masks, result.scores, result.boxes, strict=True)
    ]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _decode_mask(mask_b64: str) -> BoolMask:
    img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    return np.asarray(img, dtype=bool)
