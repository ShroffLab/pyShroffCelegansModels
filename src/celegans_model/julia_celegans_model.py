"""Julia-backed implementation of C. elegans worm space coordinate transformations.

This module provides a Python wrapper around the ShroffCelegansModelsCore Julia package,
which implements efficient worm space coordinate transformations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from warnings import warn

import numpy as np

from .base import CelegansModelBase


def _write_lattice_to_csv(
    lattice_points: np.ndarray,
    lattice_point_names: list[str],
    csvfile: Path,
) -> None:
    """Write lattice points to a CSV file in the expected format.

    Args:
        lattice_points: Array with shape (n, 2, 3) for n points, left/right, xyz.
        lattice_point_names: List of base names (e.g., ['a0', 'h0', ...]).
        csvfile: Path to write the CSV file.
    """
    with open(csvfile, "w") as f:
        f.write("name,x_voxels,y_voxels,z_voxels\n")
        for i, name in enumerate(lattice_point_names):
            right = lattice_points[i, 0]
            left = lattice_points[i, 1]
            f.write(f"{name}R,{right[0]},{right[1]},{right[2]}\n")
            f.write(f"{name}L,{left[0]},{left[1]},{left[2]}\n")


class JuliaCelegansModel(CelegansModelBase):
    """Worm space coordinate transformations using the Julia backend.

    This implementation wraps the ShroffCelegansModelsCore Julia package for
    efficient coordinate transformations. It requires Julia and the
    ShroffCelegansModelsCore package to be installed.

    Can be constructed either from a CSV file or from a numpy array:
        - JuliaCelegansModel(lattice_csv)  # from CSV file path
        - JuliaCelegansModel.from_array(lattice_points, names)  # from numpy array

    Args:
        lattice_csv: Path to a CSV file containing lattice point annotations.

    Raises:
        ImportError: If juliacall or juliapkg are not installed.
        FileNotFoundError: If the lattice CSV file does not exist.
    """

    def __init__(self, lattice_csv: Path):
        self._init_julia()

        if not lattice_csv.exists():
            raise FileNotFoundError(f"Lattice csv {lattice_csv} does not exist")
        self.lattice_csv = lattice_csv
        self._build_model(lattice_csv)

    def _init_julia(self) -> None:
        """Initialize Julia and load the required package."""
        try:
            import juliapkg  # noqa: F401 - ensures Julia deps are installed
            from juliacall import Main as jl
        except ImportError as e:
            raise ImportError(
                "Julia dependencies are required for JuliaCelegansModel. "
                "Install with: pip install celegans-model[julia]"
            ) from e

        self._jl = jl
        jl.seval("using ShroffCelegansModelsCore")

    def _build_model(self, lattice_csv: Path) -> None:
        """Build the Julia model from a CSV file."""
        self.julia_model = self._jl.ShroffCelegansModelsCore.build_celegans_model(
            str(lattice_csv)
        )
        # Standard lattice points define positions 0-10
        self._internal_range = (0.0, 10.0)
        # Allow some extrapolation beyond the lattice
        self._valid_range = (-1.0, 11.0)

    @classmethod
    def from_array(
        cls,
        lattice_points: np.ndarray,
        lattice_point_names: list[str],
    ) -> JuliaCelegansModel:
        """Create a JuliaCelegansModel from a numpy array.

        Args:
            lattice_points: Array with shape (n, 2, 3) for n lattice points,
                2 sides (right=0, left=1), and 3 spatial dimensions (x, y, z).
            lattice_point_names: List of base names for each lattice point
                (e.g., ['a0', 'h0', 'h1', ...]). Length must match n.

        Returns:
            A new JuliaCelegansModel instance.

        Raises:
            ImportError: If juliacall or juliapkg are not installed.
            ValueError: If lattice_point_names length doesn't match lattice_points.
        """
        if len(lattice_point_names) != lattice_points.shape[0]:
            raise ValueError(
                f"Number of lattice_point_names ({len(lattice_point_names)}) "
                f"must match number of lattice points ({lattice_points.shape[0]})"
            )

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._init_julia()

        # Write to a temporary CSV file and build the model
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        _write_lattice_to_csv(lattice_points, lattice_point_names, tmp_path)
        instance.lattice_csv = tmp_path
        instance._build_model(tmp_path)

        return instance

    @property
    def internal_range(self) -> tuple[float, float]:
        """Range of AP values defined by lattice points (0-10 for standard lattice)."""
        return self._internal_range

    @property
    def valid_range(self) -> tuple[float, float]:
        """Extended range allowing extrapolation beyond lattice points."""
        return self._valid_range

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

        Calls the Julia function get_untwisted_candidate_locations which returns a
        list of tuples, where each tuple has a point in the untwisted space and the
        distance of that point to the central spline.

        Args:
            target_point: The input space location of the point as (x, y, z).
            threshold: Exclude candidates further than threshold from the worm
                center spline. Defaults to None, which will return candidate
                locations at all local distance minima.
            return_scores: If True, return (coords, score) tuples. Note: the Julia
                backend does not support surface scoring, so scores will be based
                on distance only (1.0 for all candidates within threshold).
            steepness: Sigmoid steepness for score calculation (unused in Julia backend).

        Returns:
            If return_scores=False: list of (ML, DV, AP) worm space coordinates
            If return_scores=True: list of ((ML, DV, AP), score) tuples
        """
        candidates = (
            self._jl.ShroffCelegansModelsCore.get_untwisted_annotation_candidates(
                self.julia_model, target_point.tolist()
            )
        )
        points: list[tuple[float, float, float]] = []
        for candidate_point, distance in candidates:
            if threshold is None or distance <= threshold:
                points.append(tuple(candidate_point))

        if len(points) == 0:
            warn(
                f"No candidate locations found for {target_point} "
                f"with threshold {threshold}.",
                stacklevel=2,
            )

        if return_scores:
            # Julia backend doesn't compute surface scores, return 1.0 for all
            return [(p, 1.0) for p in points]
        return points

    def get_best_candidate(
        self, target_point: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Get the single best candidate for a target point.

        Chooses the candidate with the minimum distance to the spline.

        Args:
            target_point: The pixel location to be converted to worm space.

        Returns:
            The most likely worm space coordinates (ML, DV, AP) for the given
            pixel location, based on nearness to the center spline, or None
            if no candidates are found.
        """
        candidates = (
            self._jl.ShroffCelegansModelsCore.get_untwisted_annotation_candidates(
                self.julia_model, target_point.tolist()
            )
        )
        if len(candidates) == 0:
            warn(
                f"No candidate locations found for {target_point}.",
                stacklevel=2,
            )
            return None

        point = None
        min_dist = None
        for candidate_point, distance in candidates:
            if min_dist is None or distance < min_dist:
                point = candidate_point
                min_dist = distance
        return tuple(point)
