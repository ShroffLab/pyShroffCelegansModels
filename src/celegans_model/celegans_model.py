import numpy as np
from pathlib import Path
from juliacall import Main as jl
import juliapkg
from warnings import warn

jl.seval("using ShroffCelegansModelsCore")

class CelegansModel:
    def __init__(self, lattice_csv: Path):
        """A worm space definition for one time point of data, using the Julia definition.
        """
        assert lattice_csv.exists(), f"Lattice csv {lattice_csv} does not exist"
        self.lattice_csv = lattice_csv
        self.julia_model = jl.ShroffCelegansModelsCore.build_celegans_model(str(self.lattice_csv))


    def get_candidate_locations(
        self, target_point: np.ndarray, threshold: float | None = None
    ) -> list[tuple[float, float, float]]:
        """Get the possible worm space locations for a given point in input pixel space.

        Calls the julia function get_untwisted_candidate_locations which returns a
        list of tuples, where each tuple has a point in the untwisted space and the
        distance of that point to the central spline

        Args:
            target_point (np.ndarray): The input space location of the point
            threshold (float | None, optional): Exclude candidates further than
                threshold from the worm center spline. Defaults to None, which will
                return candidate locations at all local distance minima.

        Returns:
            list[tuple[float, float, float]]: All possible worm space coordiantes
            of the point, in order (AP, ML, DV)
        """

        candidates = jl.ShroffCelegansModelsCore.get_untwisted_annotation_candidates(self.julia_model, [0,0,0])
        points: list[tuple[float,float,float]] = []
        for candidate_point, distance in candidates:
            if threshold is None or distance <= threshold:
                points.append(tuple(candidate_point))
        if len(points) == 0:
            warn(
                f"No candidate locations found for {target_point} "
                f"with threshold {threshold}.",
                stacklevel=2,
            )
        return points

    def get_best_candidate(
        self, target_point: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Get the single best candidate for a target point. Currently chooses the
        candidate with the minimum distance to the spline.

        Args:
            target_point (np.ndarray): The pixel location to be converted to worm space

        Returns:
            tuple[float, float, float] | None: The most likely worm space coordinates
                for the given pixel location, based on nearness to the center spline
        """

        candidates = jl.ShroffCelegansModelsCore.get_untwisted_annotation_candidates(self.julia_model, [0,0,0])
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

    # def get_worm_coords(
    #     self,
    #     target_point: tuple[float, float, float],
    #     ap: float,
    # ) -> tuple[float, float, float]:
    #     """Get the worm coordinates for a given point in input space and ap axis value.
    #     The ap axis value must be a local minima of the distance to central curve
    #     function so that the input point is on the plane normal to the central curve.

    #     Gets the plane normal to the center spline at the AP value.
    #     Computes the ML basis vector on that plane (the unit vector centered at the
    #     center spline intersection pointing toward the right spline intersection).
    #     Computes the DV basis vector on that plane (normal to the ML vector, pointing up)
    #     Converts the input point (which is on the plane) to the new basis.


    #     Args:
    #         target_point (tuple[float, float, float]): target point in input space
    #         ap (float): ap value to use to compute the other two axis values. Must be
    #             a local minima of the distance to central spline function.

    #     Returns:
    #         tuple[float, float, float]: The worm space location of the point:
    #             (AP, ML, DV)
    #     """
    #     try:
    #         self.sanity_check(ap)
    #     except AssertionError as e:
    #         print(e)

    #     center_point = self.center_spline.interpolate([ap])[0]
    #     ml_basis, dv_basis, tan_vec = self.get_basis_vectors(ap)
    #     # get the vector from the center point to the target point
    #     target_vec = np.array(target_point) - center_point
    #     # convert that vector to the new basis space
    #     ml = np.dot(target_vec, ml_basis)
    #     dv = np.dot(target_vec, dv_basis)
    #     return ap, ml, dv

    # def get_basis_vectors(self, ap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     # If we are extrapolating beyond the seam cells,
    #     # get the basis vectors at the internal range end
    #     if ap < self.internal_range[0]:
    #         ap = self.internal_range[0]
    #     elif ap > self.internal_range[1]:
    #         ap = self.internal_range[1]
    #     right_point: np.ndarray = self.right_spline.interpolate([ap])[0]
    #     center_point: np.ndarray = self.center_spline.interpolate([ap])[0]
    #     tan_vec = self.center_spline.get_tan_vec(ap)
    #     ml_basis = center_point - right_point
    #     ml_basis = ml_basis / np.linalg.norm(ml_basis)
    #     dv_basis = np.cross(ml_basis, tan_vec)
    #     dv_basis = dv_basis / np.linalg.norm(dv_basis)
    #     return ml_basis, dv_basis, tan_vec

    # def sanity_check(self, ap):
    #     center_point = self.center_spline.interpolate([ap])[0]

    #     # Make sure the three points form a line
    #     left_point = self.left_spline.interpolate([ap])[0]
    #     right_point = self.right_spline.interpolate([ap])[0]
    #     vec1 = left_point - center_point
    #     vec2 = right_point - center_point
    #     assert math.isclose(
    #         abs(np.dot(vec1, vec2)),
    #         abs(np.linalg.norm(vec1) * np.linalg.norm(vec2)),
    #         abs_tol=0.01,
    #     ), f"Left and right points at {ap} are not colinear with center point"

    # def get_max_side_spline_distance(self):
    #     max_dist = 0
    #     for ap in np.linspace(*self.internal_range, num=25):
    #         center_point = self.center_spline.interpolate([ap])[0]
    #         right_point = self.right_spline.interpolate([ap])[0]
    #         dist = np.linalg.norm(center_point - right_point)
    #         if dist > max_dist:
    #             max_dist = dist
    #     return max_dist

    # def reparameterize(self):
    #     right = self.lattice_points[:, 0]
    #     left = self.lattice_points[:, 1]
    #     center = (right + left) / 2

    #     position = 0
    #     indices = [0]
    #     for i in range(1, 11):
    #         distance = self.center_spline.get_dist_along_spline(i - 1, i, num_samples=25)
    #         position += distance
    #         indices.append(position)

    #     self.right_spline = CubicSpline3D(indices, right)
    #     self.left_spline = CubicSpline3D(indices, left)
    #     self.center_spline = CubicSpline3D(indices, center)
    #     self.internal_range = (0, position)
