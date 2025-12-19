"""Distance to spline calculations for worm space coordinate transformations."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from .cubic_spline_3d import CubicSpline3D


def dist_to_spline(
    target_point: ArrayLike,
    spline: CubicSpline3D,
    query_range: tuple[float, float],
    plot_path: str | Path | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the distance to each point along a spline for a target point.

    Considers the range of possible locations along the spline, since some cells
    may lie outside the lattice-defined range. Does not return minima at the
    endpoints of the range.

    Args:
        target_point: Array-like object with shape (3,), containing a
            three dimensional point.
        spline: The spline to compute the euclidean distance along.
        query_range: The (min, max) range to search for distance minima.
        plot_path: Optional path to save a matplotlib plot for debugging.
            Defaults to None, which will not save a plot.
        threshold: Optional maximum distance. Candidates further than this
            threshold will be excluded.

    Returns:
        Tuple of:
        - ap_locations: 1D array of parameterized locations along the spline
        - distances: 1D array of distances from target_point to each location
        - local_minima_indices: 1D integer array of indices where local minima occur
    """
    spacing = 0.1
    num_points = int((query_range[1] - query_range[0]) // spacing)
    cand_ap_pos = np.linspace(query_range[0], query_range[1], num=num_points)
    spline_points = spline.interpolate(cand_ap_pos)
    distances = cdist(np.array([target_point]), spline_points, metric="euclidean")[0]

    # non-maximal suppression within 5 values on either side, which with spacing .1 is
    # within one seam cell.
    def get_local_minima(arr: np.ndarray) -> np.ndarray:
        height = -1 * threshold if threshold is not None else None
        peaks, _ = find_peaks(-1 * arr, distance=5, height=height)
        peaks = list(peaks)
        # remove endpoints
        if 0 in peaks:
            peaks.remove(0)
        if len(arr) - 1 in peaks:
            peaks.remove(len(arr) - 1)
        return np.array(peaks)

    local_minima_indices = get_local_minima(distances)

    if plot_path is not None:
        _save_debug_plot(plot_path, cand_ap_pos, distances, local_minima_indices)

    return cand_ap_pos, distances, local_minima_indices


def _save_debug_plot(
    plot_path: str | Path,
    ap_positions: np.ndarray,
    distances: np.ndarray,
    local_minima_indices: np.ndarray,
) -> None:
    """Save a debug plot of distance vs AP position.

    Args:
        plot_path: Path to save the plot.
        ap_positions: Array of AP positions.
        distances: Array of distances at each AP position.
        local_minima_indices: Indices of local minima.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings

        warnings.warn(
            "matplotlib not installed, cannot save debug plot. "
            "Install with: pip install matplotlib",
            stacklevel=2,
        )
        return

    save_dir = Path(plot_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    for dist, local_min_idx in zip(distances, local_minima_indices, strict=False):
        plt.plot(ap_positions, dist)
        plt.plot(ap_positions[local_min_idx], dist[local_min_idx], marker="*")
    plt.savefig(plot_path)
    plt.close()
