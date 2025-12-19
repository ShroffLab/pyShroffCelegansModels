"""Abstract base class for C. elegans worm space coordinate transformations.

This module defines the common interface for both Julia-backed and pure Python
implementations of worm space coordinate transformations.
"""

from abc import ABC, abstractmethod
from typing import Literal, overload

import numpy as np


class CelegansModelBase(ABC):
    """Abstract base class for worm space coordinate transformations.

    Worm space is a coordinate system defined relative to the worm's body:
    - Anterior/Posterior (AP): parameterized along the central spline
    - Medial/Lateral (ML): perpendicular to AP, with right side positive
    - Dorsal/Ventral (DV): perpendicular to both AP and ML, with "up" positive

    Implementations may use different backends (Julia, Python/scipy) but
    provide the same coordinate transformation interface.
    """

    @property
    @abstractmethod
    def internal_range(self) -> tuple[float, float]:
        """Range of AP values defined by lattice points.

        Returns:
            Tuple of (min_ap, max_ap) for the defined lattice points.
        """
        ...

    @property
    @abstractmethod
    def valid_range(self) -> tuple[float, float]:
        """Extended range allowing slight extrapolation beyond lattice points.

        Returns:
            Tuple of (min_ap, max_ap) including extrapolation buffer.
        """
        ...

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

    @abstractmethod
    def get_candidate_locations(
        self,
        target_point: np.ndarray,
        threshold: float | None = None,
        return_scores: bool = False,
        steepness: float = 2.0,
    ) -> (
        list[tuple[float, float, float]] | list[tuple[tuple[float, float, float], float]]
    ):
        """Get possible worm space locations for a given pixel coordinate.

        A point in pixel space may map to multiple locations in worm space
        (e.g., when the worm is coiled). This method finds all local minima
        of the distance-to-spline function.

        Args:
            target_point: (z, y, x) or (x, y, z) pixel coordinates
            threshold: Exclude candidates further than threshold from center spline.
                If None, returns all local distance minima.
            return_scores: If True, return (coords, score) tuples where score
                reflects likelihood based on distance from worm surface.
            steepness: Sigmoid steepness for score calculation (higher = sharper).

        Returns:
            If return_scores=False: list of (ML, DV, AP) tuples
            If return_scores=True: list of ((ML, DV, AP), score) tuples
        """
        ...

    @abstractmethod
    def get_best_candidate(
        self, target_point: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Get the single best candidate based on minimum distance to spline.

        Args:
            target_point: (z, y, x) or (x, y, z) pixel coordinates

        Returns:
            The (ML, DV, AP) tuple for the most likely worm space location,
            or None if no candidates are found.
        """
        ...

    def get_worm_coords(
        self,
        target_point: tuple[float, float, float],
        ap: float,
    ) -> tuple[float, float, float]:
        """Get worm coordinates for a point at a specific AP location.

        The AP value must be a local minimum of the distance-to-spline function
        so that the point lies on the plane normal to the central spline.

        Args:
            target_point: (z, y, x) or (x, y, z) pixel coordinates
            ap: AP value (must be a local distance minimum)

        Returns:
            (ML, DV, AP) worm space coordinates

        Raises:
            NotImplementedError: If the implementation does not support this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_worm_coords"
        )

    def get_basis_vectors(self, ap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ML, DV, and tangent basis vectors at a given AP location.

        Args:
            ap: AP location along the worm

        Returns:
            Tuple of (ml_basis, dv_basis, tan_vec) - orthonormal basis vectors

        Raises:
            NotImplementedError: If the implementation does not support this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_basis_vectors"
        )

    def get_surface_score(
        self, target_point: np.ndarray, ap: float, steepness: float = 2.0
    ) -> float:
        """Compute surface distance score for a point at a given AP location.

        The score reflects how likely this candidate is based on its distance
        from the worm surface. Points within the expected seam cell width
        get scores close to 1, while points far outside get scores approaching 0.

        Args:
            target_point: (z, y, x) pixel coordinates
            ap: AP location along the worm
            steepness: Controls sigmoid falloff (higher = sharper transition)

        Returns:
            Score in [0, 1] where 1 = within seam cell width, 0 = far outside

        Raises:
            NotImplementedError: If the implementation does not support this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_surface_score"
        )

    def get_max_side_spline_distance(self) -> float:
        """Get the maximum distance from center to side spline (worm width).

        Returns:
            Maximum width of the worm in pixels

        Raises:
            NotImplementedError: If the implementation does not support this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_max_side_spline_distance"
        )
