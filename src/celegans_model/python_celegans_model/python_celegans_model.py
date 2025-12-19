"""Pure Python implementation of C. elegans worm space coordinate transformations.

This module provides the PythonCelegansModel class which uses scipy for spline
interpolation, as an alternative to the Julia-backed JuliaCelegansModel.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, overload
from warnings import warn

import numpy as np

from ..base import CelegansModelBase
from ..constants import STANDARD_SEAM_CELLS
from .cubic_spline_3d import CubicSpline3D
from .dist_to_spline import dist_to_spline


def _load_lattice_from_csv(
    csvfile: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load lattice points from a CSV file.

    Args:
        csvfile: Path to CSV with columns: name, x_voxels, y_voxels, z_voxels.
            Names should have L/R suffixes (e.g., a0L, a0R, h0L, h0R, ...).

    Returns:
        Tuple of (lattice_points array with shape (n, 2, 3), list of base names).

    Raises:
        ImportError: If pandas is not installed.
        FileNotFoundError: If the CSV file does not exist.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required to read CSV files. "
            "Install with: pip install celegans-model[io]"
        ) from e

    if not csvfile.exists():
        raise FileNotFoundError(f"Lattice CSV {csvfile} does not exist")

    df = pd.read_csv(csvfile, index_col="name")
    df.index = df.index.str.lower()

    # Find all unique base names (without L/R suffix)
    all_names = df.index.tolist()
    base_names_set: set[str] = set()
    for name in all_names:
        if name.endswith("l") or name.endswith("r"):
            base_names_set.add(name[:-1])

    # Order by standard seam cells first, then any extras
    standard_order = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
    ordered_names = []
    for std_name in standard_order:
        if std_name in base_names_set:
            ordered_names.append(std_name)
            base_names_set.remove(std_name)
    # Add any remaining non-standard names
    ordered_names.extend(sorted(base_names_set))

    # Build the lattice array
    lattice_points = []
    for name in ordered_names:
        right_row = df.loc[name + "r"]
        left_row = df.loc[name + "l"]
        right = [right_row["x_voxels"], right_row["y_voxels"], right_row["z_voxels"]]
        left = [left_row["x_voxels"], left_row["y_voxels"], left_row["z_voxels"]]
        lattice_points.append([right, left])

    return np.array(lattice_points), ordered_names


class PythonCelegansModel(CelegansModelBase):
    """Pure Python implementation of worm space coordinate transformations.

    The worm space is computed from the annotated lattice points.
    Worm space has three axes:
        - Anterior/Posterior (AP): parameterized along the spline.
          The parameterization can be configured (see below).
          Target points can extend beyond the endpoint values a small distance.
        - Medial/Lateral (ML): Defined by the line connecting the left and right
          lattice points, or left/right splines between the lattice points, with 0
          value at the middle location and right side being positive. The axis is
          normal to the central spline.
        - Dorsal/Ventral (DV): Defined by the line perpendicular to the ML axis and
          to the central spline. The 0 value is at the central spline, and "up"
          is positive.

    Can be constructed either from a numpy array or from a CSV file:
        - PythonCelegansModel(lattice_points, ...)  # from numpy array
        - PythonCelegansModel.from_csv(path, ...)   # from CSV file

    Args:
        lattice_points: A numpy array with dims (n, 2, 3) for the n lattice points
            (in head-to-tail order), 2 sides (right=0, left=1), and 3 spatial dimensions.
        parameterization: How to parameterize the AP axis.
            Options:
            - "arc_length": position seam cells by cumulative physical distance (default)
            - "uniform": standard seam cells (a0, h0-h2, v1-v6, t) get canonical positions
              (0-10 * spacing). Virtual cells are interpolated based on arc-length ratio.
              Requires lattice_point_names to be provided.
        spacing: For "uniform" parameterization, the interval between standard seam cells.
            Default is 1.0 (gives 0, 1, ..., 10), or use 150 (gives 0, 150, ..., 1500).
        lattice_point_names: Optional list of base names for each lattice point
            (e.g., ['a0', 'h0', 'h1', ...]). Length must match lattice_points.shape[0].
    """

    def __init__(
        self,
        lattice_points: np.ndarray,
        parameterization: str = "arc_length",
        spacing: float = 1.0,
        lattice_point_names: list[str] | None = None,
    ):
        self._init_from_array(
            lattice_points, parameterization, spacing, lattice_point_names
        )

    def _init_from_array(
        self,
        lattice_points: np.ndarray,
        parameterization: str,
        spacing: float,
        lattice_point_names: list[str] | None,
    ) -> None:
        """Initialize from a numpy array."""
        self.lattice_points = lattice_points
        self.parameterization = parameterization
        self.spacing = spacing
        self.lattice_point_names = lattice_point_names

        if lattice_point_names is not None:
            if len(lattice_point_names) != lattice_points.shape[0]:
                raise ValueError(
                    f"Number of lattice_point_names ({len(lattice_point_names)}) "
                    f"must match number of lattice points "
                    f"({lattice_points.shape[0]})"
                )
        right = lattice_points[:, 0]
        left = lattice_points[:, 1]
        center = (right + left) / 2

        self._internal_range = (0.0, float(lattice_points.shape[0]))

        indices = list(range(int(self._internal_range[0]), int(self._internal_range[1])))

        self.right_spline = CubicSpline3D(indices, right)
        self.left_spline = CubicSpline3D(indices, left)
        self.center_spline = CubicSpline3D(indices, center)
        self._reparameterize()
        buffer_amt = 0.05 * (self._internal_range[1] - self._internal_range[0])
        self._valid_range = (
            self._internal_range[0] - buffer_amt,
            self._internal_range[1] + buffer_amt,
        )

    @classmethod
    def from_csv(
        cls,
        csvfile: Path,
        parameterization: str = "uniform",
        spacing: float = 1.0,
    ) -> PythonCelegansModel:
        """Create a PythonCelegansModel from a CSV file.

        Args:
            csvfile: Path to CSV with columns: name, x_voxels, y_voxels, z_voxels.
                Names should have L/R suffixes (e.g., a0L, a0R, h0L, h0R, ...).
            parameterization: How to parameterize the AP axis ("arc_length" or "uniform").
                Default is "uniform" since CSV files typically have named lattice points.
            spacing: For "uniform" parameterization, the interval between standard cells.

        Returns:
            A new PythonCelegansModel instance.

        Raises:
            ImportError: If pandas is not installed.
            FileNotFoundError: If the CSV file does not exist.
        """
        lattice_points, names = _load_lattice_from_csv(csvfile)
        instance = cls.__new__(cls)
        instance._init_from_array(lattice_points, parameterization, spacing, names)
        return instance

    @property
    def internal_range(self) -> tuple[float, float]:
        """Range of AP values defined by lattice points."""
        return self._internal_range

    @property
    def valid_range(self) -> tuple[float, float]:
        """Extended range allowing slight extrapolation beyond lattice points."""
        return self._valid_range

    @overload
    def get_candidate_locations(
        self,
        target_point: np.ndarray,
        threshold: float | None = None,
        return_scores: Literal[False] = False,
        steepness: float = 2.0,
    ) -> list[tuple[float, float, float]]: ...

    @overload
    def get_candidate_locations(
        self,
        target_point: np.ndarray,
        threshold: float | None = None,
        *,
        return_scores: Literal[True],
        steepness: float = 2.0,
    ) -> list[tuple[tuple[float, float, float], float]]: ...

    def get_candidate_locations(
        self,
        target_point: np.ndarray,
        threshold: float | None = None,
        return_scores: bool = False,
        steepness: float = 2.0,
    ) -> (
        list[tuple[float, float, float]] | list[tuple[tuple[float, float, float], float]]
    ):
        """Get the possible worm space locations for a given point in input pixel space.

        First computes the distance to the center spline along the length of the worm.
        Then finds local minima where distance is below threshold. (Warns if none exist).
        For each local minima, computes the worm space coordinates of the point.

        Args:
            target_point: The input space location of the point.
            threshold: Exclude candidates further than threshold from the worm center
                spline. Defaults to None, which returns candidate locations at all
                local distance minima.
            return_scores: If True, return (coords, score) tuples where score
                reflects likelihood based on distance from worm surface.
            steepness: Sigmoid steepness for score calculation (higher = sharper).

        Returns:
            If return_scores=False: list of (ML, DV, AP) tuples
            If return_scores=True: list of ((ML, DV, AP), score) tuples
        """
        ap_locs, _distances, local_minima_indices = dist_to_spline(
            target_point, self.center_spline, self._valid_range, threshold=threshold
        )

        if len(local_minima_indices) == 0:
            warn(
                f"No candidate locations found for {target_point} "
                f"with threshold {threshold}.",
                stacklevel=2,
            )
            return []

        cand_ap_locs = ap_locs[local_minima_indices]
        candidates = [self.get_worm_coords(target_point, s) for s in cand_ap_locs]

        if return_scores:
            scores = [
                self.get_surface_score(target_point, ap, steepness) for ap in cand_ap_locs
            ]
            return list(zip(candidates, scores, strict=False))
        else:
            return candidates

    def get_best_candidate(
        self, target_point: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Get the single best candidate for a target point.

        Currently chooses the candidate with the minimum distance to the spline.

        Args:
            target_point: The pixel location to be converted to worm space.

        Returns:
            The most likely worm space coordinates (ML, DV, AP) for the given pixel
            location, based on nearness to the center spline. Returns None if no
            candidates are found.
        """
        ap_locs, distances, local_minima_indices = dist_to_spline(
            target_point, self.center_spline, self._valid_range, threshold=None
        )
        if len(local_minima_indices) == 0:
            warn(
                f"No candidate locations found for {target_point}.",
                stacklevel=2,
            )
            return None
        min_distances: list = list(distances[local_minima_indices])
        min_dist = min(min_distances)
        min_idx = local_minima_indices[min_distances.index(min_dist)]
        ap = ap_locs[min_idx]
        return self.get_worm_coords(target_point, ap)

    def get_worm_coords(
        self,
        target_point: tuple[float, float, float] | np.ndarray,
        ap: float,
    ) -> tuple[float, float, float]:
        """Get the worm coordinates for a given point in input space and AP axis value.

        The AP axis value must be a local minima of the distance to central curve
        function so that the input point is on the plane normal to the central curve.

        Gets the plane normal to the center spline at the AP value.
        Computes the ML basis vector on that plane (the unit vector centered at the
        center spline intersection pointing toward the right spline intersection).
        Computes the DV basis vector on that plane (normal to the ML vector, pointing up).
        Converts the input point (which is on the plane) to the new basis.

        Args:
            target_point: Target point in input space (z, y, x).
            ap: AP value to use to compute the other two axis values. Must be
                a local minima of the distance to central spline function.

        Returns:
            The worm space location of the point: (ML, DV, AP)
        """
        try:
            self._sanity_check(ap)
        except AssertionError as e:
            print(e)

        center_point = self.center_spline.interpolate([ap])[0]
        ml_basis, dv_basis, _tan_vec = self.get_basis_vectors(ap)
        # get the vector from the center point to the target point
        target_vec = np.array(target_point) - center_point
        # convert that vector to the new basis space
        ml = np.dot(target_vec, ml_basis)
        dv = np.dot(target_vec, dv_basis)
        return ml, dv, ap

    def get_basis_vectors(self, ap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the ML, DV, and tangent basis vectors at a given AP location.

        Args:
            ap: AP location along the worm.

        Returns:
            Tuple of (ml_basis, dv_basis, tan_vec) - orthonormal basis vectors.
        """
        # If we are extrapolating beyond the seam cells,
        # get the basis vectors at the internal range end
        if ap < self._internal_range[0]:
            ap = self._internal_range[0]
        elif ap > self._internal_range[1]:
            ap = self._internal_range[1]
        right_point: np.ndarray = self.right_spline.interpolate([ap])[0]
        center_point: np.ndarray = self.center_spline.interpolate([ap])[0]
        tan_vec = self.center_spline.get_tan_vec(ap)
        ml_basis = center_point - right_point
        ml_basis = ml_basis / np.linalg.norm(ml_basis)
        dv_basis = np.cross(ml_basis, tan_vec)
        dv_basis = dv_basis / np.linalg.norm(dv_basis)
        return ml_basis, dv_basis, np.array(tan_vec)

    def _sanity_check(self, ap: float) -> None:
        """Check that left, center, and right points are colinear at given AP."""
        center_point = self.center_spline.interpolate([ap])[0]
        left_point = self.left_spline.interpolate([ap])[0]
        right_point = self.right_spline.interpolate([ap])[0]
        vec1 = left_point - center_point
        vec2 = right_point - center_point
        assert math.isclose(
            abs(np.dot(vec1, vec2)),
            abs(np.linalg.norm(vec1) * np.linalg.norm(vec2)),
            abs_tol=0.01,
        ), f"Left and right points at {ap} are not colinear with center point"

    def get_max_side_spline_distance(self) -> float:
        """Get the maximum distance from center to side spline (worm width)."""
        max_dist = 0.0
        for ap in np.linspace(*self._internal_range, num=25):
            center_point = self.center_spline.interpolate([ap])[0]
            right_point = self.right_spline.interpolate([ap])[0]
            dist = np.linalg.norm(center_point - right_point)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def get_surface_score(
        self, target_point: np.ndarray, ap: float, steepness: float = 2.0
    ) -> float:
        """Compute surface distance score for a point at a given AP location.

        The score reflects how likely this candidate is based on its distance
        from the worm surface. Points within the expected seam cell width
        (distance from center to right/left spline) get scores close to 1,
        while points far outside get scores approaching 0.

        Args:
            target_point: (z, y, x) pixel coordinates.
            ap: AP location along the worm.
            steepness: Controls sigmoid falloff (higher = sharper transition).

        Returns:
            Score in [0, 1] where 1 = within seam cell width, 0 = far outside.
        """
        # Get seam cell width at this AP (distance from center to right spline)
        center_point = self.center_spline.interpolate([ap])[0]
        right_point = self.right_spline.interpolate([ap])[0]
        seam_width = np.linalg.norm(center_point - right_point)

        # Get distance from target point to center spline
        distance_to_center = np.linalg.norm(np.array(target_point) - center_point)

        # Sigmoid score: 1 when close, 0 when far
        return 1 / (1 + (distance_to_center / seam_width) ** steepness)

    def _reparameterize(self) -> None:
        """Reparameterize spline based on chosen parameterization mode."""
        right = self.lattice_points[:, 0]
        left = self.lattice_points[:, 1]
        center = (right + left) / 2

        if self.parameterization == "arc_length":
            indices = self._compute_arc_length_indices()
        elif self.parameterization == "uniform":
            indices = self._compute_uniform_indices()
        else:
            raise ValueError(
                f"Unknown parameterization: {self.parameterization}. "
                f"Must be 'arc_length' or 'uniform'"
            )

        self.right_spline = CubicSpline3D(indices, right)
        self.left_spline = CubicSpline3D(indices, left)
        self.center_spline = CubicSpline3D(indices, center)
        self._internal_range = (0.0, max(indices))

    def _compute_arc_length_indices(self) -> list[float]:
        """Compute indices based on cumulative arc length along the spline."""
        n_points = self.lattice_points.shape[0]
        position = 0.0
        indices = [0.0]
        for i in range(1, n_points):
            distance = self.center_spline.get_dist_along_spline(i - 1, i, num_samples=25)
            position += distance
            indices.append(position)
        return indices

    def _compute_uniform_indices(self) -> list[float]:
        """Compute indices with standard cells at canonical positions.

        Standard seam cells get positions 0, spacing, 2*spacing, etc.
        Virtual cells are interpolated based on arc-length ratio between neighbors.
        """
        if self.lattice_point_names is None:
            raise ValueError(
                "lattice_point_names is required for uniform parameterization"
            )

        n_points = self.lattice_points.shape[0]
        indices: list[float | None] = [None] * n_points

        # Assign canonical positions to standard seam cells
        for i, name in enumerate(self.lattice_point_names):
            canonical_idx = STANDARD_SEAM_CELLS.get(name.lower())
            if canonical_idx is not None:
                indices[i] = canonical_idx * self.spacing

        # Find indices of standard cells for interpolation
        standard_indices = [i for i, idx in enumerate(indices) if idx is not None]

        # Interpolate virtual cells based on arc-length
        for i in range(n_points):
            if indices[i] is not None:
                continue  # Already assigned (standard cell)

            indices[i] = self._interpolate_virtual_cell_index(
                i, indices, standard_indices
            )

        # All indices should be assigned at this point
        return [float(idx) for idx in indices if idx is not None]

    def _interpolate_virtual_cell_index(
        self,
        i: int,
        indices: list[float | None],
        standard_indices: list[int],
    ) -> float:
        """Compute the index for a virtual cell based on neighboring standard cells."""
        prev_std, next_std = self._find_neighbor_standard_cells(i, standard_indices)

        if prev_std is not None and next_std is not None:
            return self._interpolate_between(i, prev_std, next_std, indices)
        elif prev_std is not None:
            return self._extrapolate_after(i, prev_std, standard_indices, indices)
        elif next_std is not None:
            return self._extrapolate_before(i, next_std, standard_indices, indices)
        else:
            raise ValueError(f"No standard cells found to interpolate index {i}")

    def _find_neighbor_standard_cells(
        self, i: int, standard_indices: list[int]
    ) -> tuple[int | None, int | None]:
        """Find the previous and next standard cell indices for position i."""
        prev_std = None
        next_std = None
        for si in standard_indices:
            if si < i:
                prev_std = si
            elif si > i:
                next_std = si
                break
        return prev_std, next_std

    def _interpolate_between(
        self,
        i: int,
        prev_std: int,
        next_std: int,
        indices: list[float | None],
    ) -> float:
        """Interpolate index for a virtual cell between two standard cells."""
        prev_idx = indices[prev_std]
        next_idx = indices[next_std]
        assert prev_idx is not None and next_idx is not None

        dist_prev_to_i = self.center_spline.get_dist_along_spline(
            prev_std, i, num_samples=25
        )
        dist_prev_to_next = self.center_spline.get_dist_along_spline(
            prev_std, next_std, num_samples=25
        )
        ratio = dist_prev_to_i / dist_prev_to_next
        return prev_idx + ratio * (next_idx - prev_idx)

    def _extrapolate_after(
        self,
        i: int,
        prev_std: int,
        standard_indices: list[int],
        indices: list[float | None],
    ) -> float:
        """Extrapolate index for a virtual cell after the last standard cell."""
        prev_idx = indices[prev_std]
        assert prev_idx is not None

        # Find the second-to-last standard cell
        prev_prev_std = None
        for si in reversed(standard_indices):
            if si < prev_std:
                prev_prev_std = si
                break

        if prev_prev_std is not None:
            prev_prev_idx = indices[prev_prev_std]
            assert prev_prev_idx is not None

            dist_prev_prev_to_prev = self.center_spline.get_dist_along_spline(
                prev_prev_std, prev_std, num_samples=25
            )
            dist_prev_to_i = self.center_spline.get_dist_along_spline(
                prev_std, i, num_samples=25
            )
            ratio = dist_prev_to_i / dist_prev_prev_to_prev
            return prev_idx + ratio * (prev_idx - prev_prev_idx)
        else:
            # Only one standard cell before, use arc-length from it
            dist = self.center_spline.get_dist_along_spline(prev_std, i, num_samples=25)
            return prev_idx + dist

    def _extrapolate_before(
        self,
        i: int,
        next_std: int,
        standard_indices: list[int],
        indices: list[float | None],
    ) -> float:
        """Extrapolate index for a virtual cell before the first standard cell."""
        next_idx = indices[next_std]
        assert next_idx is not None

        # Find the second standard cell
        next_next_std = None
        for si in standard_indices:
            if si > next_std:
                next_next_std = si
                break

        if next_next_std is not None:
            next_next_idx = indices[next_next_std]
            assert next_next_idx is not None

            dist_next_to_next_next = self.center_spline.get_dist_along_spline(
                next_std, next_next_std, num_samples=25
            )
            dist_i_to_next = self.center_spline.get_dist_along_spline(
                i, next_std, num_samples=25
            )
            ratio = dist_i_to_next / dist_next_to_next_next
            return next_idx - ratio * (next_next_idx - next_idx)
        else:
            # Only one standard cell after, use arc-length to it
            dist = self.center_spline.get_dist_along_spline(i, next_std, num_samples=25)
            return next_idx - dist
