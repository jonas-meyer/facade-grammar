"""Grammar extraction: fit a parametric split grammar to detected facade features.

For each building that survived feature segmentation we cluster window bbox centres in y
(floor bands) and x (columns), derive a gable region above the topmost floor,
and assign the detected door to a floor band. Output is one ``FacadeGrammar``
per building — normalised to the facade bbox so records pool across photos
of different sizes.

Clustering is histogram-peak 1D: build a density histogram over [0,1],
gaussian-smooth it, and take local maxima above a prominence threshold as
cluster centres. Gap-based splitting is fragile when window heights are
comparable to floor-to-floor gaps (as they are on Dutch canal houses);
density peaks are invariant to that ratio.
"""

import numpy as np
from hamilton.function_modifiers import tag

from facade_grammar.config import GrammarConfig
from facade_grammar.geo import WGS84_GEOD
from facade_grammar.schemas.buildings import (
    Building,
    Facade,
    FacadeFeatures,
    FacadeGrammar,
    FacadeMask,
    FeatureInstance,
)


@tag(stage="grammar")
def facade_grammars(
    facade_features: dict[str, FacadeFeatures],
    facade_masks: dict[str, FacadeMask],
    raw_buildings: list[Building],
    canal_facades: list[Facade],
    grammar: GrammarConfig,
) -> dict[str, FacadeGrammar]:
    """Fit one ``FacadeGrammar`` per building with SAM-detected features.

    Sequential — clustering is sub-ms per facade; no Parallelizable needed
    at blog-scale. BAG attributes and real-world facade dimensions are
    looked up per building_id and folded into the record.
    """
    buildings_by_id = {b.building_id: b for b in raw_buildings}
    facades_by_id = {f.building_id: f for f in canal_facades}
    grammars: dict[str, FacadeGrammar] = {}
    for bid, ff in facade_features.items():
        fm = facade_masks.get(bid)
        building = buildings_by_id.get(bid)
        facade_rec = facades_by_id.get(bid)
        if fm is None or building is None or facade_rec is None:
            continue
        g = _fit_grammar(ff, fm, building, facade_rec, grammar)
        if g is not None:
            grammars[bid] = g
    return grammars


def _fit_grammar(
    ff: FacadeFeatures,
    fm: FacadeMask,
    building: Building,
    facade_rec: Facade,
    cfg: GrammarConfig,
) -> FacadeGrammar | None:
    fx0, fy0, fx1, fy1 = fm.facade_bbox_px
    fw, fh = fx1 - fx0, fy1 - fy0
    if fw <= 0 or fh <= 0:
        return None

    windows = [i for i in ff.instances if i.cls == "window"]
    doors = [i for i in ff.instances if i.cls == "door"]
    if len(windows) < cfg.min_windows:
        return None

    win_x = _norm_centres_x(windows, fx0, fw)
    win_y = _norm_centres_y(windows, fy0, fh)
    win_w = np.array([(i.bbox[2] - i.bbox[0]) / fw for i in windows])
    win_h = np.array([(i.bbox[3] - i.bbox[1]) / fh for i in windows])

    window_row_y_centers = _histogram_peak_cluster_1d(
        win_y,
        n_bins=cfg.histogram_n_bins,
        smooth_sigma=cfg.smooth_sigma_bins,
        min_prominence=cfg.min_peak_prominence,
    )
    column_x_centers = _histogram_peak_cluster_1d(
        win_x,
        n_bins=cfg.histogram_n_bins,
        smooth_sigma=cfg.smooth_sigma_bins,
        min_prominence=cfg.min_peak_prominence,
    )
    sorted_rows = np.sort(window_row_y_centers)
    floor_split_ys = (
        (sorted_rows[:-1] + sorted_rows[1:]) / 2 if sorted_rows.size >= 2 else np.empty(0)
    )

    door_floor_idx: int | None = None
    door_x_center: float | None = None
    if doors:
        best = max(doors, key=lambda d: d.score)
        dx = ((best.bbox[0] + best.bbox[2]) / 2 - fx0) / fw
        dy = ((best.bbox[1] + best.bbox[3]) / 2 - fy0) / fh
        door_x_center = float(np.clip(dx, 0.0, 1.0))
        if window_row_y_centers.size:
            door_floor_idx = int(np.argmin(np.abs(window_row_y_centers - dy)))

    # Gable: strip above the topmost window row.
    gable_y_end: float | None = None
    if window_row_y_centers.size:
        top_row_y = float(window_row_y_centers.min())
        gable_y_end = max(0.0, top_row_y - win_h.mean() / 2)
        if gable_y_end <= 0.0:
            gable_y_end = None

    n_floors = int(window_row_y_centers.size) or 1
    facade_width_m = _wgs84_distance_m(facade_rec.edge_start, facade_rec.edge_end)
    facade_height_m = float(building.height_m)
    floor_height_m = facade_height_m / max(n_floors, 1)
    window_w_mean_m = float(win_w.mean() * facade_width_m)
    window_h_mean_m = float(win_h.mean() * facade_height_m)
    gable_prominence_m = (
        float(facade_height_m - (building.eaves_height_m or 0.0))
        if building.eaves_height_m is not None
        else None
    )
    n_floors_matches_bag = (
        None if building.floor_count is None else n_floors == building.floor_count
    )

    return FacadeGrammar(
        building_id=ff.building_id,
        photo_id=ff.photo_id,
        n_floors=n_floors,
        n_columns=int(column_x_centers.size),
        window_row_y_centers=sorted(window_row_y_centers.tolist()),
        floor_split_ys=floor_split_ys.tolist(),
        column_x_centers=sorted(column_x_centers.tolist()),
        window_w_mean=float(win_w.mean()),
        window_w_std=float(win_w.std()),
        window_h_mean=float(win_h.mean()),
        window_h_std=float(win_h.std()),
        door_floor_idx=door_floor_idx,
        door_x_center=door_x_center,
        gable_y_end=gable_y_end,
        n_windows_used=len(windows),
        n_doors_used=len(doors),
        facade_width_m=facade_width_m,
        facade_height_m=facade_height_m,
        window_w_mean_m=window_w_mean_m,
        window_h_mean_m=window_h_mean_m,
        floor_height_m=floor_height_m,
        gable_prominence_m=gable_prominence_m,
        bag_floor_count=building.floor_count,
        bag_roof_type=building.roof_type,
        bag_construction_year=building.construction_year,
        n_floors_matches_bag=n_floors_matches_bag,
    )


def _wgs84_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    _, _, dist = WGS84_GEOD.inv(a[0], a[1], b[0], b[1])
    return float(dist)


def _histogram_peak_cluster_1d(
    values: np.ndarray,
    *,
    n_bins: int,
    smooth_sigma: float,
    min_prominence: float,
) -> np.ndarray:
    """Cluster 1D values in [0,1] by gaussian-smoothed-histogram peak-finding.

    Robust to overlapping/noisy detections and to the case where cluster
    widths are comparable to inter-cluster gaps — exactly the regime of
    tall canal-house windows where gap-based clustering breaks down.
    """
    if values.size == 0:
        return np.empty(0)
    hist, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    if hist.max() == 0:
        return np.empty(0)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    smoothed = _gaussian_smooth_1d(hist.astype(float), smooth_sigma)
    threshold = smoothed.max() * min_prominence
    # A bin is a local max if it exceeds its left neighbour strictly and is at
    # least as large as the right; the asymmetry breaks ties on flat plateaus.
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
            and smoothed[i] >= threshold
        ):
            peaks.append(bin_centers[i])
    return np.array(peaks)


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def _norm_centres_x(instances: list[FeatureInstance], x0: float, width: float) -> np.ndarray:
    return np.array([((i.bbox[0] + i.bbox[2]) / 2 - x0) / width for i in instances])


def _norm_centres_y(instances: list[FeatureInstance], y0: float, height: float) -> np.ndarray:
    return np.array([((i.bbox[1] + i.bbox[3]) / 2 - y0) / height for i in instances])
