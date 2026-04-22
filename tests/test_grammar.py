"""Tests for 1D histogram-peak clustering in ``facade_grammar.pipeline.grammar``."""

from functools import partial

import numpy as np

from facade_grammar.pipeline.grammar import _histogram_peak_cluster_1d

_cluster = partial(_histogram_peak_cluster_1d, n_bins=40, smooth_sigma=1.5, min_prominence=0.2)


def test_empty_input_returns_no_peaks() -> None:
    assert _cluster(np.empty(0)).size == 0


def test_two_well_separated_clusters_return_two_peaks() -> None:
    values = np.concatenate([np.full(10, 0.25), np.full(10, 0.75)])
    peaks = _cluster(values)
    assert peaks.size == 2
    assert abs(peaks[0] - 0.25) < 0.05
    assert abs(peaks[1] - 0.75) < 0.05


def test_low_prominence_secondary_peak_is_filtered() -> None:
    # Strong mode at 0.3 (100 values), weak mode at 0.7 (5 values) < 20% prominence.
    values = np.concatenate([np.full(100, 0.3), np.full(5, 0.7)])
    peaks = _cluster(values)
    assert peaks.size == 1
    assert abs(peaks[0] - 0.3) < 0.05


def test_single_cluster_returns_single_peak() -> None:
    peaks = _cluster(np.full(50, 0.5))
    assert peaks.size == 1
    assert abs(peaks[0] - 0.5) < 0.05
