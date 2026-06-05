"""Parameterized tests for CelegansModel implementations.

These tests run against both PythonCelegansModel and JuliaCelegansModel
to ensure consistent behavior across implementations.
"""

from pathlib import Path

import numpy as np
import pytest

from celegans_model import CelegansModelBase


def create_test_lattice_points(n_points: int) -> np.ndarray:
    """Create simple curved lattice points for testing.

    Returns shape (n_points, 2, 3) for (points, right/left sides, xyz).
    """
    t = np.linspace(0, np.pi, n_points)
    right = np.column_stack([t * 100, np.sin(t) * 50, np.zeros(n_points)])
    left = np.column_stack([t * 100, np.sin(t) * 50 + 20, np.zeros(n_points)])
    return np.stack([right, left], axis=1)


STANDARD_NAMES = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
LATTICE_CSV = Path(__file__).parent / "resources" / "lattice.csv"


def get_implementations():
    """Get list of available implementations for parameterization."""
    implementations = []

    # Check if Python/scipy is available
    try:
        import scipy  # noqa: F401

        implementations.append("python")
    except ImportError:
        pass

    # Check if Julia is available
    try:
        import juliacall  # noqa: F401

        implementations.append("julia")
    except ImportError:
        pass

    return implementations


def create_model(impl: str, from_csv: bool = False):
    """Create a model instance for the given implementation."""
    if impl == "python":
        from celegans_model import PythonCelegansModel

        if from_csv:
            return PythonCelegansModel.from_csv(LATTICE_CSV)
        else:
            lattice_points = create_test_lattice_points(len(STANDARD_NAMES))
            return PythonCelegansModel(
                lattice_points,
                parameterization="uniform",
                spacing=1.0,
                lattice_point_names=STANDARD_NAMES,
            )
    elif impl == "julia":
        from celegans_model import JuliaCelegansModel

        if from_csv:
            return JuliaCelegansModel(LATTICE_CSV)
        else:
            lattice_points = create_test_lattice_points(len(STANDARD_NAMES))
            return JuliaCelegansModel.from_array(lattice_points, STANDARD_NAMES)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


@pytest.fixture(params=get_implementations())
def model_impl(request):
    """Fixture that yields each available implementation name."""
    return request.param


@pytest.fixture
def model_from_array(model_impl):
    """Create a model from array for the given implementation."""
    return create_model(model_impl, from_csv=False)


@pytest.fixture
def model_from_csv(model_impl):
    """Create a model from CSV for the given implementation."""
    return create_model(model_impl, from_csv=True)


class TestBaseInterface:
    """Test that all implementations satisfy the base interface."""

    def test_is_base_subclass(self, model_from_array):
        """Test that model is a subclass of CelegansModelBase."""
        assert isinstance(model_from_array, CelegansModelBase)

    def test_has_internal_range(self, model_from_array):
        """Test that implementation has internal_range property."""
        result = model_from_array.internal_range
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_has_valid_range(self, model_from_array):
        """Test that implementation has valid_range property."""
        result = model_from_array.valid_range
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_valid_range_contains_internal_range(self, model_from_array):
        """Test that valid_range contains internal_range."""
        assert model_from_array.valid_range[0] <= model_from_array.internal_range[0]
        assert model_from_array.valid_range[1] >= model_from_array.internal_range[1]


class TestGetCandidateLocations:
    """Test get_candidate_locations method across implementations."""

    def test_returns_list_of_tuples(self, model_from_csv):
        """Test that method returns a list of tuples."""
        # Use a point that should be near the worm
        target_point = np.array([350.0, 120.0, 150.0])
        candidates = model_from_csv.get_candidate_locations(target_point)
        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert all(isinstance(c, tuple) for c in candidates)
            assert all(len(c) == 3 for c in candidates)

    def test_returns_three_coordinates(self, model_from_csv):
        """Test that each candidate has exactly 3 coordinates (AP, ML, DV)."""
        target_point = np.array([350.0, 120.0, 150.0])
        candidates = model_from_csv.get_candidate_locations(target_point)
        if len(candidates) > 0:
            for candidate in candidates:
                assert len(candidate) == 3, "Each candidate should have AP, ML, DV"

    def test_threshold_filters_candidates(self, model_from_csv):
        """Test that threshold parameter filters candidates by distance."""
        target_point = np.array([350.0, 120.0, 150.0])
        all_candidates = model_from_csv.get_candidate_locations(
            target_point, threshold=None
        )
        filtered_candidates = model_from_csv.get_candidate_locations(
            target_point, threshold=1.0
        )
        assert len(filtered_candidates) <= len(all_candidates)

    def test_warns_when_no_candidates_found(self, model_from_csv):
        """Test that a warning is issued when no candidates meet threshold."""
        # Point far from the worm
        target_point = np.array([10000.0, 10000.0, 10000.0])
        with pytest.warns(UserWarning, match="No candidate locations found"):
            candidates = model_from_csv.get_candidate_locations(
                target_point, threshold=0.001
            )
            assert len(candidates) == 0

    def test_return_scores(self, model_from_csv):
        """Test that return_scores returns tuples with scores."""
        target_point = np.array([350.0, 120.0, 150.0])
        result = model_from_csv.get_candidate_locations(target_point, return_scores=True)
        if len(result) > 0:
            assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
            assert all(isinstance(item[0], tuple) for item in result)
            assert all(isinstance(item[1], int | float) for item in result)


class TestGetBestCandidate:
    """Test get_best_candidate method across implementations."""

    def test_returns_tuple_or_none(self, model_from_csv):
        """Test that method returns a tuple or None."""
        target_point = np.array([350.0, 120.0, 150.0])
        best = model_from_csv.get_best_candidate(target_point)
        assert best is None or (isinstance(best, tuple) and len(best) == 3)

    def test_returns_three_coordinates(self, model_from_csv):
        """Test that result has exactly 3 coordinates (AP, ML, DV)."""
        target_point = np.array([350.0, 120.0, 150.0])
        best = model_from_csv.get_best_candidate(target_point)
        if best is not None:
            assert len(best) == 3, "Result should have AP, ML, DV coordinates"

    def test_consistent_results(self, model_from_csv):
        """Test that calling the method twice returns the same result."""
        target_point = np.array([350.0, 120.0, 150.0])
        best1 = model_from_csv.get_best_candidate(target_point)
        best2 = model_from_csv.get_best_candidate(target_point)
        if best1 is not None and best2 is not None:
            assert np.allclose(best1, best2, rtol=1e-10, atol=1e-10)

    def test_best_candidate_in_candidate_list(self, model_from_csv):
        """Test that best candidate is one of the candidates."""
        target_point = np.array([350.0, 120.0, 150.0])
        candidates = model_from_csv.get_candidate_locations(target_point)
        best = model_from_csv.get_best_candidate(target_point)

        if len(candidates) > 0 and best is not None:
            assert any(
                np.allclose(best, candidate, rtol=1e-5, atol=1e-8)
                for candidate in candidates
            ), "Best candidate should be one of the candidate locations"


class TestFromCSV:
    """Test loading models from CSV files."""

    def test_from_csv_loads_successfully(self, model_from_csv):
        """Test that model loads from CSV without error."""
        assert model_from_csv is not None
        assert model_from_csv.internal_range[0] == 0
        assert model_from_csv.internal_range[1] >= 10


class TestFromArray:
    """Test loading models from numpy arrays."""

    def test_from_array_loads_successfully(self, model_from_array):
        """Test that model loads from array without error."""
        assert model_from_array is not None
        assert model_from_array.internal_range == (0, 10)


class TestSeamCellUntwisting:
    """Test that seam cells untwist correctly."""

    def test_all_seam_cells_have_candidates(self, model_from_csv, model_impl):
        """Test that all seam cells produce at least one candidate."""
        if model_impl == "julia":
            pytest.xfail("Julia doesn't find candidates for seam cells at nose/tail")

        import pandas as pd

        df = pd.read_csv(LATTICE_CSV)

        for _, row in df.iterrows():
            name = row["name"]
            twisted_point = np.array([row["x_voxels"], row["y_voxels"], row["z_voxels"]])
            candidates = model_from_csv.get_candidate_locations(
                twisted_point, threshold=None
            )
            assert len(candidates) > 0, (
                f"[{model_impl}] Seam cell {name} at position {twisted_point} "
                f"produced no candidates"
            )


class TestUntwistingValues:
    """Test that untwisting produces correct coordinate values.

    Both implementations use arc-length parameterization for these tests.
    Julia doesn't find candidates for points beyond nose/tail (xfail).
    """

    def test_straight_worm_center_point(self, model_impl):
        """Test untwisting a point on the center of a straight worm.

        With a straight worm (left/right are parallel lines), a point on the
        center line should untwist to (~0, ~0, mid_AP) where mid_AP is the
        middle of the worm's arc length.
        """
        # Create a straight worm: left and right are parallel lines along x-axis
        # Right side at y=0, left side at y=20, both at z=0
        # Worm length is 100 units (x from 0 to 100)
        n_points = 11
        x_coords = np.linspace(0, 100, n_points)
        right = np.column_stack([x_coords, np.zeros(n_points), np.zeros(n_points)])
        left = np.column_stack([x_coords, np.full(n_points, 20.0), np.zeros(n_points)])
        lattice_points = np.stack([right, left], axis=1)

        if model_impl == "python":
            from celegans_model import PythonCelegansModel

            # Use arc_length to match Julia's parameterization
            model = PythonCelegansModel(
                lattice_points,
                parameterization="arc_length",
                lattice_point_names=STANDARD_NAMES,
            )
        else:
            from celegans_model import JuliaCelegansModel

            model = JuliaCelegansModel.from_array(lattice_points, STANDARD_NAMES)

        # Expected AP is at the arc-length midpoint (50 for a 100-unit worm)
        # Note: Julia's internal_range is hardcoded and doesn't reflect actual arc length
        expected_ap = 50.0

        # Test a point on the center line at the middle of the worm (x=50)
        # Center is at y=10 (midpoint between y=0 and y=20)
        center_point = np.array([50.0, 10.0, 0.0])

        best = model.get_best_candidate(center_point)
        assert best is not None, "Should find a candidate for center point"

        ml, dv, ap = best
        assert ml == pytest.approx(0.0, abs=1.0), f"ML should be ~0 (center), got {ml}"
        assert dv == pytest.approx(0.0, abs=1.0), f"DV should be ~0, got {dv}"
        # Allow 5% tolerance on AP
        assert ap == pytest.approx(
            expected_ap, rel=0.05
        ), f"AP should be ~{expected_ap}, got {ap}"

    def test_straight_worm_right_side_point(self, model_impl):
        """Test untwisting a point on the right side of a straight worm.

        A point on the right seam cell line should have ML < 0 (or > 0 depending
        on convention) indicating it's on the right side.
        """
        # Worm length is 100 units (x from 0 to 100)
        n_points = 11
        x_coords = np.linspace(0, 100, n_points)
        right = np.column_stack([x_coords, np.zeros(n_points), np.zeros(n_points)])
        left = np.column_stack([x_coords, np.full(n_points, 20.0), np.zeros(n_points)])
        lattice_points = np.stack([right, left], axis=1)

        if model_impl == "python":
            from celegans_model import PythonCelegansModel

            model = PythonCelegansModel(
                lattice_points,
                parameterization="arc_length",
                lattice_point_names=STANDARD_NAMES,
            )
        else:
            from celegans_model import JuliaCelegansModel

            model = JuliaCelegansModel.from_array(lattice_points, STANDARD_NAMES)

        # Expected AP is at the arc-length midpoint (50 for a 100-unit worm)
        expected_ap = 50.0

        # Test a point on the right side at the middle of the worm
        right_point = np.array([50.0, 0.0, 0.0])  # On the right seam cell line

        best = model.get_best_candidate(right_point)
        assert best is not None, "Should find a candidate for right side point"

        ml, dv, ap = best
        # ML should be -10 (right side is 10 units from center in negative direction)
        assert ml == pytest.approx(-10.0, abs=1.0), f"ML should be ~-10, got {ml}"
        assert dv == pytest.approx(0.0, abs=1.0), f"DV should be ~0, got {dv}"
        assert ap == pytest.approx(
            expected_ap, rel=0.05
        ), f"AP should be ~{expected_ap}, got {ap}"

    def test_u_shaped_worm_center_point(self, model_impl):
        """Test untwisting a point on the center of a U-shaped worm.

        The worm curves in a U shape (following a sine wave in y),
        but left/right seam cells remain parallel (constant offset in z).
        A point on the center line should still untwist to (~0, ~0, mid_AP).
        """
        n_points = 11
        t = np.linspace(0, np.pi, n_points)
        # U-shape: x increases linearly, y follows sine wave
        x_coords = np.linspace(0, 100, n_points)
        y_coords = 50 * np.sin(t)  # Goes from 0 to 50 and back to 0

        # Right and left are parallel, offset by 20 in z
        right = np.column_stack([x_coords, y_coords, np.zeros(n_points)])
        left = np.column_stack([x_coords, y_coords, np.full(n_points, 20.0)])
        lattice_points = np.stack([right, left], axis=1)

        if model_impl == "python":
            from celegans_model import PythonCelegansModel

            model = PythonCelegansModel(
                lattice_points,
                parameterization="arc_length",
                lattice_point_names=STANDARD_NAMES,
            )
        else:
            from celegans_model import JuliaCelegansModel

            model = JuliaCelegansModel.from_array(lattice_points, STANDARD_NAMES)

        # Expected AP is the arc length to the midpoint (~73.18 for this geometry)
        # The sine curve y=50*sin(pi*x/100) has arc length ~146.37, so midpoint ~73.18
        expected_ap = 73.18

        # Test a point on the center line at the middle of the worm
        # At the middle, x=50, y=50 (peak of the U), center z=10
        center_point = np.array([50.0, 50.0, 10.0])

        best = model.get_best_candidate(center_point)
        assert best is not None, "Should find a candidate for center point"

        ml, dv, ap = best
        assert ml == pytest.approx(0.0, abs=1.0), f"ML should be ~0 (center), got {ml}"
        assert dv == pytest.approx(0.0, abs=1.0), f"DV should be ~0, got {dv}"
        assert ap == pytest.approx(
            expected_ap, rel=0.05
        ), f"AP should be ~{expected_ap}, got {ap}"

    def test_u_shaped_worm_off_center_point(self, model_impl):
        """Test untwisting a point offset from the center of a U-shaped worm.

        A point that's offset from the center in the ML direction should
        have non-zero ML coordinate after untwisting.
        """
        if model_impl == "julia":
            pytest.xfail("Julia doesn't find candidates for points on seam cell surface")

        n_points = 11
        t = np.linspace(0, np.pi, n_points)
        x_coords = np.linspace(0, 100, n_points)
        y_coords = 50 * np.sin(t)

        # Right at z=0, left at z=20
        right = np.column_stack([x_coords, y_coords, np.zeros(n_points)])
        left = np.column_stack([x_coords, y_coords, np.full(n_points, 20.0)])
        lattice_points = np.stack([right, left], axis=1)

        if model_impl == "python":
            from celegans_model import PythonCelegansModel

            model = PythonCelegansModel(
                lattice_points,
                parameterization="arc_length",
                lattice_point_names=STANDARD_NAMES,
            )
        else:
            from celegans_model import JuliaCelegansModel

            model = JuliaCelegansModel.from_array(lattice_points, STANDARD_NAMES)

        # Expected AP is the arc length to the midpoint (~73.18 for this geometry)
        expected_ap = 73.18

        # Test a point on the right side at the middle of the worm (z=0, on the
        # right seam cell)
        right_point = np.array([50.0, 50.0, 0.0])

        best = model.get_best_candidate(right_point)
        assert best is not None, "Should find a candidate for right side point"

        ml, dv, ap = best
        # ML should be -10 (right side is 10 units from center in negative direction)
        assert ml == pytest.approx(-10.0, abs=1.0), f"ML should be ~-10, got {ml}"
        assert dv == pytest.approx(0.0, abs=1.0), f"DV should be ~0, got {dv}"
        assert ap == pytest.approx(
            expected_ap, rel=0.05
        ), f"AP should be ~{expected_ap}, got {ap}"
