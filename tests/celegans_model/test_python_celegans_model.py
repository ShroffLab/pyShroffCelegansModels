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


class TestRetwist:
    """Test retwist method (inverse of get_worm_coords)."""

    def test_retwist_at_center_returns_center_point(self, python_model):
        """retwist(0, 0, ap) should return the central spline point at ap."""
        ap = 5.0
        center_point = python_model.center_spline.interpolate([ap])[0]
        result = python_model.retwist(0.0, 0.0, ap)
        np.testing.assert_array_almost_equal(result, center_point, decimal=4)

    def test_retwist_inverts_get_worm_coords(self, python_model):
        """retwist(*get_worm_coords(p, ap)) should equal p when p is on the AP plane."""
        ap = 5.0
        center_point = python_model.center_spline.interpolate([ap])[0]
        ml_basis, dv_basis, _ = python_model.get_basis_vectors(ap)
        # Construct a point on the AP-plane (no tangential component).
        original = center_point + 3.0 * ml_basis + (-2.0) * dv_basis

        ml, dv, ap_out = python_model.get_worm_coords(tuple(original), ap)
        recovered = python_model.retwist(ml, dv, ap_out)

        np.testing.assert_array_almost_equal(recovered, original, decimal=4)

    def test_retwist_round_trip_multiple_ap(self, python_model):
        """Round-trip retwist(get_worm_coords(p)) for several AP values."""
        for ap in [1.5, 3.0, 5.0, 7.5, 9.0]:
            center_point = python_model.center_spline.interpolate([ap])[0]
            ml_basis, dv_basis, _ = python_model.get_basis_vectors(ap)
            original = center_point + 4.0 * ml_basis + 1.5 * dv_basis

            ml, dv, ap_out = python_model.get_worm_coords(tuple(original), ap)
            recovered = python_model.retwist(ml, dv, ap_out)

            err = float(np.linalg.norm(recovered - original))
            assert err < 1e-3, f"Round-trip error {err} at ap={ap}"

    def test_retwist_returns_ndarray_length_3(self, python_model):
        result = python_model.retwist(0.0, 0.0, 5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


class TestStraightenVolume:
    """Test straighten_volume method."""

    @staticmethod
    def _make_phantom(model, radius=8.0):
        """Build a 3D pixel volume with a bright tube along the central spline.

        The volume axis order matches the lattice convention used in the test
        fixtures, which is ``(x, y, z)``: the lattice points span x in
        [0, 100*pi], y in [-30, 80], z = 0. Phantom shape is (X, Y, Z).
        """
        ap_dense = np.linspace(*model.internal_range, num=400)
        center_pts = model.center_spline.interpolate(ap_dense)

        x_lo, x_hi = float(center_pts[:, 0].min()) - 30, float(center_pts[:, 0].max()) + 30
        y_lo, y_hi = float(center_pts[:, 1].min()) - 30, float(center_pts[:, 1].max()) + 30
        z_lo, z_hi = float(center_pts[:, 2].min()) - 30, float(center_pts[:, 2].max()) + 30

        X = int(np.ceil(x_hi - x_lo))
        Y = int(np.ceil(y_hi - y_lo))
        Z = int(np.ceil(z_hi - z_lo))

        # Shift center points into volume index space.
        offset = np.array([x_lo, y_lo, z_lo])
        center_pts_shifted = center_pts - offset

        vol = np.zeros((X, Y, Z), dtype=np.float32)
        xx, yy, zz = np.indices((X, Y, Z))
        grid = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        for c in center_pts_shifted:
            d2 = ((grid - c) ** 2).sum(-1)
            vol = np.maximum(vol, np.exp(-d2 / (2 * (radius / 2.0) ** 2)))
        return vol, offset

    def test_output_shape(self, python_model):
        """straighten_volume returns array with expected shape."""
        vol, _ = self._make_phantom(python_model)
        # Build a model whose center spline is shifted to live inside the
        # phantom's index space.
        out, ap_values = python_model.straighten_volume(vol, n_ap=64, extent=10)
        assert out.shape == (64, 21, 21)
        assert ap_values.shape == (64,)

    def test_centerline_alignment(self, python_model):
        """Worm centerline should appear at (dv=0, ml=0) — center column of each slice."""
        vol, offset = self._make_phantom(python_model, radius=8.0)
        # Construct a shifted model whose splines live in volume-index space.
        shifted_lattice = python_model.lattice_points - offset
        shifted_model = PythonCelegansModel(
            shifted_lattice,
            parameterization=python_model.parameterization,
            spacing=python_model.spacing,
            lattice_point_names=python_model.lattice_point_names,
        )
        out, _ = shifted_model.straighten_volume(vol, n_ap=80, extent=15)

        # Column at (dv=0, ml=0) in straightened space corresponds to centerline.
        center_col_mean = float(out[:, 15, 15].mean())
        corner_col_mean = float(out[:, 0, 0].mean())
        # Centerline is bright; corners are background ~0.
        assert center_col_mean > 0.5
        assert center_col_mean > 5.0 * (corner_col_mean + 1e-6)

    def test_retwist_round_trip_via_volume(self, python_model):
        """Map a known straightened-space pt → retwist → confirm it's on AP plane."""
        out, ap_values = python_model.straighten_volume(
            self._make_phantom(python_model)[0], n_ap=20, extent=10
        )
        # Pick a sample (ml, dv, ap) and round-trip via retwist + get_worm_coords.
        ml_in, dv_in = 2.0, -1.5
        ap_in = float(ap_values[10])
        twisted = python_model.retwist(ml_in, dv_in, ap_in)
        ml_out, dv_out, ap_out = python_model.get_worm_coords(tuple(twisted), ap_in)
        assert ml_out == pytest.approx(ml_in, abs=1e-4)
        assert dv_out == pytest.approx(dv_in, abs=1e-4)
        assert ap_out == pytest.approx(ap_in, abs=1e-4)

    def test_default_extent_and_n_ap(self, python_model):
        """Defaults pick reasonable values when not specified."""
        vol, _ = self._make_phantom(python_model)
        out, ap_values = python_model.straighten_volume(vol)
        assert out.ndim == 3
        assert out.shape[1] == out.shape[2]  # square cross-section
        assert out.shape[1] >= 3  # at least some extent
        assert out.shape[0] == ap_values.shape[0]

    def test_invalid_extent_raises(self, python_model):
        with pytest.raises(ValueError, match="extent must be"):
            python_model.straighten_volume(np.zeros((10, 10, 10)), n_ap=10, extent=0)

    def test_invalid_n_ap_raises(self, python_model):
        with pytest.raises(ValueError, match="n_ap must be"):
            python_model.straighten_volume(np.zeros((10, 10, 10)), n_ap=1, extent=5)


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
