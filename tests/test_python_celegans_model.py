"""Python-specific tests for PythonCelegansModel.

These tests cover Python-specific behavior that doesn't apply to JuliaCelegansModel,
such as parameterization modes, surface scores, basis vectors, and worm coords.
Common interface tests are in test_celegans_model_interface.py.
"""

import numpy as np
import pytest

# Skip entire module if scipy is not available
pytest.importorskip("scipy")

from celegans_model import STANDARD_SEAM_CELLS, PythonCelegansModel  # noqa: E402


def create_test_lattice_points(n_points: int) -> np.ndarray:
    """Create simple curved lattice points for testing.

    Returns an array of shape (n_points, 2, 3) with left/right lattice points
    along a curved path. Axis 1: right=0, left=1. Axis 2: x, y, z.
    """
    t = np.linspace(0, np.pi, n_points)
    right = np.column_stack([t * 100, np.sin(t) * 50, np.zeros(n_points)])
    left = np.column_stack([t * 100, np.sin(t) * 50 + 20, np.zeros(n_points)])
    return np.stack([right, left], axis=1)


@pytest.fixture
def standard_names():
    """Standard seam cell names."""
    return ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]


@pytest.fixture
def python_model(standard_names):
    """Create a PythonCelegansModel instance for testing."""
    lattice_points = create_test_lattice_points(len(standard_names))
    return PythonCelegansModel(
        lattice_points,
        parameterization="uniform",
        spacing=1.0,
        lattice_point_names=standard_names,
    )


class TestPythonCelegansModelInit:
    """Test PythonCelegansModel initialization."""

    def test_init_with_valid_lattice(self, standard_names):
        """Test that model initializes correctly with valid lattice."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="uniform",
            lattice_point_names=standard_names,
        )
        assert model.internal_range == (0, 10)

    def test_init_with_arc_length(self, standard_names):
        """Test initialization with arc_length parameterization."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="arc_length",
            lattice_point_names=standard_names,
        )
        assert model.internal_range[0] == 0
        assert model.internal_range[1] > 0

    def test_init_requires_names_for_uniform(self, standard_names):
        """Test that uniform parameterization requires names."""
        lattice_points = create_test_lattice_points(len(standard_names))
        with pytest.raises(ValueError, match="lattice_point_names is required"):
            PythonCelegansModel(
                lattice_points,
                parameterization="uniform",
                lattice_point_names=None,
            )

    def test_from_csv(self):
        """Test creating model from CSV file."""
        from pathlib import Path

        csv_path = Path(__file__).parent / "resources" / "lattice.csv"
        model = PythonCelegansModel.from_csv(csv_path)
        # The test CSV may have extra lattice points beyond standard 11
        # so internal_range may extend beyond (0, 10)
        assert model.internal_range[0] == 0
        assert model.internal_range[1] >= 10
        assert model.lattice_point_names is not None
        assert "a0" in model.lattice_point_names

    def test_from_csv_nonexistent(self):
        """Test that from_csv raises error for nonexistent file."""
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            PythonCelegansModel.from_csv(Path("/nonexistent/lattice.csv"))


class TestPythonRangeProperties:
    """Test range properties."""

    def test_internal_range_uniform(self, python_model):
        """Test that internal_range returns expected values for uniform."""
        assert python_model.internal_range == (0, 10)

    def test_valid_range_extends_internal(self, python_model):
        """Test that valid_range extends beyond internal_range."""
        assert python_model.valid_range[0] < python_model.internal_range[0]
        assert python_model.valid_range[1] > python_model.internal_range[1]


class TestPythonGetWormCoords:
    """Test get_worm_coords method."""

    def test_center_point_has_zero_ml_dv(self, python_model):
        """Test that center spline points have ML=0 and DV=0."""
        ap = 5.0
        center_point = python_model.center_spline.interpolate([ap])[0]
        ml, dv, result_ap = python_model.get_worm_coords(tuple(center_point), ap)

        assert ml == pytest.approx(0.0, abs=0.01)
        assert dv == pytest.approx(0.0, abs=0.01)
        assert result_ap == pytest.approx(ap, abs=0.01)

    def test_right_point_has_nonzero_ml(self, python_model):
        """Test that right spline points have non-zero ML value."""
        ap = 5.0
        right_point = python_model.right_spline.interpolate([ap])[0]
        ml, _dv, _result_ap = python_model.get_worm_coords(tuple(right_point), ap)

        # Right side should have non-zero ML (distance from center)
        # The sign depends on the coordinate system convention in the test data
        assert abs(ml) > 0.1


class TestPythonGetBasisVectors:
    """Test get_basis_vectors method."""

    def test_basis_vectors_orthonormal(self, python_model):
        """Test that basis vectors are orthonormal."""
        ap = 5.0
        ml_basis, dv_basis, _tan_vec = python_model.get_basis_vectors(ap)

        # Check normalization (within tolerance since these are computed)
        assert np.linalg.norm(ml_basis) == pytest.approx(1.0, abs=0.01)
        assert np.linalg.norm(dv_basis) == pytest.approx(1.0, abs=0.01)

        # Check orthogonality
        assert np.dot(ml_basis, dv_basis) == pytest.approx(0.0, abs=0.01)


class TestPythonGetSurfaceScore:
    """Tests for get_surface_score method."""

    def test_score_at_center_is_high(self, python_model):
        """Points on the center spline should have high scores."""
        ap = 5.0
        center_point = python_model.center_spline.interpolate([ap])[0]
        score = python_model.get_surface_score(center_point, ap)

        # At center, distance_to_center = 0, so score = 1/(1+0) = 1
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_at_surface_is_moderate(self, python_model):
        """Points at the seam cell surface should have score ~0.5."""
        ap = 5.0
        right_point = python_model.right_spline.interpolate([ap])[0]
        score = python_model.get_surface_score(right_point, ap)

        # At surface, distance = seam_width, so score = 1/(1+1^k) = 0.5 for k=2
        assert score == pytest.approx(0.5, abs=0.1)

    def test_score_far_outside_is_low(self, python_model):
        """Points far outside the worm should have low scores."""
        ap = 5.0
        center_point = python_model.center_spline.interpolate([ap])[0]
        right_point = python_model.right_spline.interpolate([ap])[0]
        seam_width = np.linalg.norm(center_point - right_point)

        direction = (right_point - center_point) / seam_width
        far_point = center_point + direction * seam_width * 3

        score = python_model.get_surface_score(far_point, ap)

        # At 3x seam_width: score = 1/(1+3^2) = 1/10 = 0.1
        assert score == pytest.approx(0.1, abs=0.05)

    def test_steepness_affects_falloff(self, python_model):
        """Higher steepness should cause sharper score falloff."""
        ap = 5.0
        right_point = python_model.right_spline.interpolate([ap])[0]
        center_point = python_model.center_spline.interpolate([ap])[0]
        seam_width = np.linalg.norm(center_point - right_point)
        direction = (right_point - center_point) / seam_width
        point_1_5x = center_point + direction * seam_width * 1.5

        score_steep = python_model.get_surface_score(point_1_5x, ap, steepness=4.0)
        score_gentle = python_model.get_surface_score(point_1_5x, ap, steepness=1.0)

        assert score_steep < score_gentle


class TestUniformParameterization:
    """Test uniform parameterization behavior."""

    def test_standard_cells_at_canonical_positions(self, standard_names):
        """Test that standard cells are at canonical positions."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="uniform",
            spacing=1.0,
            lattice_point_names=standard_names,
        )

        for i, name in enumerate(standard_names):
            expected_idx = STANDARD_SEAM_CELLS[name.lower()]
            # lattice_points has shape (n_points, 2, 3) - index 0=right, 1=left
            center = (lattice_points[i, 0, :] + lattice_points[i, 1, :]) / 2
            interpolated = model.center_spline.interpolate([expected_idx])[0]
            dist = np.linalg.norm(center - interpolated)
            assert dist < 0.01, f"{name} not at canonical position {expected_idx}"

    def test_with_virtual_cells(self, standard_names):
        """Test that virtual cells are interpolated between standard cells."""
        names = [
            "a0",
            "h0",
            "h1",
            "virtual_a",
            "h2",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "t",
        ]
        lattice_points = create_test_lattice_points(len(names))

        model = PythonCelegansModel(
            lattice_points,
            parameterization="uniform",
            spacing=1.0,
            lattice_point_names=names,
        )

        # Standard cells should still be at canonical positions
        for i, name in enumerate(names):
            if name.lower() in STANDARD_SEAM_CELLS:
                expected_idx = STANDARD_SEAM_CELLS[name.lower()]
                # lattice_points has shape (n_points, 2, 3) - index 0=right, 1=left
                center = (lattice_points[i, 0, :] + lattice_points[i, 1, :]) / 2
                interpolated = model.center_spline.interpolate([expected_idx])[0]
                dist = np.linalg.norm(center - interpolated)
                assert dist < 0.01, f"{name} not at canonical position {expected_idx}"

    def test_custom_spacing(self, standard_names):
        """Test uniform parameterization with custom spacing."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="uniform",
            spacing=150.0,
            lattice_point_names=standard_names,
        )

        assert model.internal_range == (0, 1500)


class TestArcLengthParameterization:
    """Test arc_length parameterization behavior."""

    def test_first_point_at_zero(self, standard_names):
        """Test that first point is at 0."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="arc_length",
            lattice_point_names=standard_names,
        )

        # lattice_points has shape (n_points, 2, 3) - index 0=right, 1=left
        a0_center = (lattice_points[0, 0, :] + lattice_points[0, 1, :]) / 2
        a0_interp = model.center_spline.interpolate([0])[0]
        assert np.linalg.norm(a0_center - a0_interp) < 0.01

    def test_range_based_on_arc_length(self, standard_names):
        """Test that internal_range is based on cumulative arc length."""
        lattice_points = create_test_lattice_points(len(standard_names))
        model = PythonCelegansModel(
            lattice_points,
            parameterization="arc_length",
            lattice_point_names=standard_names,
        )

        assert model.internal_range[0] == 0
        assert model.internal_range[1] > 0


class TestCubicSpline3D:
    """Test CubicSpline3D helper class."""

    def test_interpolation(self):
        """Test basic interpolation."""
        from celegans_model.python_celegans_model import CubicSpline3D

        indices = np.arange(0, 11)
        locations = np.zeros(shape=(11, 3))
        for i in range(11):
            locations[i] = [i, i, i]

        spline = CubicSpline3D(indices, locations)
        result = spline.interpolate([3])[0]
        np.testing.assert_array_almost_equal(result, [3, 3, 3])

    def test_tan_vec(self):
        """Test tangent vector calculation."""
        import math

        from celegans_model.python_celegans_model import CubicSpline3D

        indices = np.arange(0, 11)
        locations = np.zeros(shape=(11, 3))
        for i in range(11):
            locations[i] = [i, 0, 0]

        spline = CubicSpline3D(indices, locations)

        for i in range(11):
            a, b, c = spline.get_tan_vec(i)
            assert math.isclose(a, 1, abs_tol=0.01)
            assert math.isclose(b, 0, abs_tol=0.01)
            assert math.isclose(c, 0, abs_tol=0.01)
