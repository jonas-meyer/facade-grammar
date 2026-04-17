"""Learned grammar distributions (the final pipeline output)."""

from pydantic import BaseModel, Field


class DistributionSummary(BaseModel):
    """Summary statistics for a continuous variable across the sample."""

    mean: float
    std: float
    count: int = Field(ge=0)
    p05: float
    p50: float
    p95: float


class GrammarDistributions(BaseModel):
    """Canal-house archetype as a bundle of learned distributions."""

    num_floors: DistributionSummary
    windows_per_floor: DistributionSummary
    window_aspect_ratio: DistributionSummary
    facade_width_m: DistributionSummary
    alignment_score: DistributionSummary
    sample_size: int = Field(ge=0)
