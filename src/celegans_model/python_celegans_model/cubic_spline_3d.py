"""3D cubic spline implementation for worm space coordinate transformations."""

from collections.abc import Iterable

import numpy as np
from scipy.interpolate import CubicSpline


class CubicSpline3D:
    """A helper class for combining three splines, one for each dimension x y and z.

    Also adds an explicit boundary polynomial so that the spline is extended linearly,
    rather than just having zero first and second derivatives at the endpoint as
    the bc_type "natural" implies.

    Args:
        s: The cell "locations" along the main worm axis. Defined by
            the index of the seam cells in the following list (usually 0 to 10)
            a0 h0 h1 h2 v1 v2 v3 v4 v5 v6 t
        locations: An array with shape (n, 3), where n is usually 11,
            representing the 3D locations of the points along the parameterization
            of the spline
    """

    def __init__(self, s: Iterable[float], locations: np.ndarray):
        self.splines: Iterable[CubicSpline] = [
            CubicSpline(s, locations[:, dim], bc_type="natural")
            for dim in range(locations.shape[1])
        ]
        for spline in self.splines:
            self.add_boundary_knots(spline)

    def add_boundary_knots(self, spline: CubicSpline):
        """Add knots infinitesimally to the left and right.

        Additional intervals are added to have zero 2nd and 3rd derivatives,
        and to maintain the first derivative from whatever boundary condition
        was selected. The spline is modified in place.
        """
        # determine the slope at the left edge
        leftx = spline.x[0]
        lefty = spline(leftx)
        leftslope = spline(leftx, nu=1)

        # add a new breakpoint just to the left and use the
        # known slope to construct the PPoly coefficients.
        leftxnext = np.nextafter(leftx, leftx - 1)
        leftynext = lefty + leftslope * (leftxnext - leftx)
        leftcoeffs = np.array([0, 0, leftslope, leftynext])
        spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

        # repeat with additional knots to the right
        rightx = spline.x[-1]
        righty = spline(rightx)
        rightslope = spline(rightx, nu=1)
        rightxnext = np.nextafter(rightx, rightx + 1)
        rightynext = righty + rightslope * (rightxnext - rightx)
        rightcoeffs = np.array([0, 0, rightslope, rightynext])
        spline.extend(rightcoeffs[..., None], np.r_[rightxnext])

    def interpolate(self, indices: np.ndarray) -> np.ndarray:
        """Find the point at a given index along the parameterization of the spline.

        Args:
            indices: Number(s) representing the location along the worm's main axis
                parameterized by the lattice points (e.g., between 0 and 10).

        Returns:
            Array with shape (len(indices), 3) containing [x, y, z] locations
            for each query index.
        """
        return np.stack([spline(indices) for spline in self.splines], axis=1)

    def get_tan_vec(self, s: float) -> list[float]:
        """Get the tangent vector at a given position along the spline.

        Args:
            s: Position along the spline parameterization.

        Returns:
            List of [dx, dy, dz] derivatives at the given position.
        """
        derivatives = []
        for spline in self.splines:
            der = spline.derivative()
            derivatives.append(float(der(s)))
        return derivatives

    def get_dist_along_spline(
        self, start: float, end: float, num_samples: int = 20
    ) -> float:
        """Compute the arc length along the spline between two positions.

        Args:
            start: Starting position along the spline.
            end: Ending position along the spline.
            num_samples: Number of sample points for numerical integration.

        Returns:
            Approximate arc length between start and end.
        """
        sample_points = np.linspace(start, end, num=num_samples)
        sample_locs = self.interpolate(sample_points)

        o1 = sample_locs[:-1]
        o2 = sample_locs[1:]
        diff = np.abs(o1 - o2)
        norms = np.linalg.norm(diff, axis=1)
        total_dist = np.sum(norms)
        return total_dist.item()
