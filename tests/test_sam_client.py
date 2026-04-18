"""Tests for the SAM client's pure-logic helpers (no live HTTP)."""

import base64
import io

import numpy as np
from PIL import Image

from facade_grammar.clients.sam import _decode_mask


def _encode_mask(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("1")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_mask_round_trip_all_true() -> None:
    arr = np.ones((4, 6), dtype=bool)
    out = _decode_mask(_encode_mask(arr))
    assert out.shape == arr.shape
    assert out.dtype == bool
    assert bool(out.all())


def test_decode_mask_round_trip_checkerboard() -> None:
    arr = np.indices((4, 4)).sum(axis=0) % 2 == 0  # checkerboard
    out = _decode_mask(_encode_mask(arr))
    assert np.array_equal(out, arr)


def test_decode_mask_preserves_shape_on_empty() -> None:
    arr = np.zeros((3, 5), dtype=bool)
    out = _decode_mask(_encode_mask(arr))
    assert out.shape == arr.shape
    assert not out.any()
