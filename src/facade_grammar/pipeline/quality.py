"""Post-SAM mask-quality filter.

Rejects facades whose cached mask PNG is visibly bad — too tiny (building
too far / SAM hallucinated nothing useful), too large (building too close,
frame completely filled), or cropped at opposite edges (vertical or
horizontal cut-off). Lives downstream of vision so its thresholds live in
``QualityConfig`` rather than ``SamConfig`` — tweaking the filter costs
only a grammar/induction re-run, not a SAM re-run.
"""

import logging

import numpy as np
from hamilton.function_modifiers import tag
from PIL import Image

from facade_grammar.config import QualityConfig
from facade_grammar.schemas.buildings import FacadeMask

log = logging.getLogger(__name__)


@tag(stage="quality")
def facade_masks(
    raw_facade_masks: dict[str, FacadeMask],
    quality: QualityConfig,
) -> dict[str, FacadeMask]:
    """Drop facades with unusable mask geometry; keep everything else intact."""
    kept: dict[str, FacadeMask] = {}
    rejected: dict[str, int] = {"tiny": 0, "huge": 0, "vcrop": 0, "hcrop": 0}
    for bid, mask in raw_facade_masks.items():
        reason = _reject_reason(mask, quality)
        if reason is None:
            kept[bid] = mask
        else:
            rejected[reason] += 1
    log.info(
        "quality filter: kept %d / %d (dropped: %s)",
        len(kept),
        len(raw_facade_masks),
        ", ".join(f"{k}={v}" for k, v in rejected.items() if v),
    )
    return kept


def _reject_reason(mask: FacadeMask, quality: QualityConfig) -> str | None:
    arr = np.asarray(Image.open(mask.mask_path), dtype=bool)
    area_ratio = float(arr.sum()) / arr.size
    if area_ratio < quality.min_mask_area_ratio:
        return "tiny"
    if area_ratio > quality.max_mask_area_ratio:
        return "huge"
    if quality.reject_vertically_cropped and arr[0, :].any() and arr[-1, :].any():
        return "vcrop"
    if quality.reject_horizontally_cropped and arr[:, 0].any() and arr[:, -1].any():
        return "hcrop"
    return None
