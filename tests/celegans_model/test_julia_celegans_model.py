"""Julia-specific tests for JuliaCelegansModel.

These tests cover Julia-specific behavior that doesn't apply to PythonCelegansModel.
Common interface tests are in test_celegans_model_interface.py.

These tests require Julia and the ShroffCelegansModelsCore package.
Use `pytest -k julia` to run only these tests or `pytest -k 'not julia'` to skip them.
"""

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if Julia is not available
pytest.importorskip("juliacall")

from celegans_model import JuliaCelegansModel  # noqa: E402


@pytest.fixture
def lattice_csv_path():
    """Path to the test lattice CSV file."""
    return (Path(__file__).parent / "resources" / "lattice.csv").resolve()


@pytest.fixture
def julia_model(lattice_csv_path):
    """Create a JuliaCelegansModel instance for testing."""
    return JuliaCelegansModel(lattice_csv_path)


class TestJuliaSpecificBehavior:
    """Test Julia-specific behavior."""

    def test_init_with_nonexistent_lattice(self):
        """Test that initialization fails with nonexistent lattice file."""
        nonexistent_path = Path("/nonexistent/lattice.csv")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            JuliaCelegansModel(nonexistent_path)

    def test_from_array(self):
        """Test creating JuliaCelegansModel from numpy array."""
        names = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
        t = np.linspace(0, np.pi, len(names))
        right = np.column_stack([t * 100, np.sin(t) * 50, np.zeros(len(names))])
        left = np.column_stack([t * 100, np.sin(t) * 50 + 20, np.zeros(len(names))])
        lattice_points = np.stack([right, left], axis=1)

        model = JuliaCelegansModel.from_array(lattice_points, names)
        assert model is not None
        assert model.internal_range == (0.0, 10.0)

    def test_from_array_validates_names_length(self):
        """Test that from_array validates names length matches array."""
        names = ["a0", "h0"]  # Too few names
        lattice_points = np.random.rand(11, 2, 3)

        with pytest.raises(ValueError, match="must match"):
            JuliaCelegansModel.from_array(lattice_points, names)


class TestJuliaNotImplemented:
    """Test that Python-only methods raise NotImplementedError."""

    def test_get_worm_coords_raises(self, julia_model):
        """Test that get_worm_coords raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            julia_model.get_worm_coords((0, 0, 0), 5.0)

    def test_get_basis_vectors_raises(self, julia_model):
        """Test that get_basis_vectors raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            julia_model.get_basis_vectors(5.0)

    def test_get_surface_score_raises(self, julia_model):
        """Test that get_surface_score raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            julia_model.get_surface_score(np.array([0, 0, 0]), 5.0)

    def test_get_max_side_spline_distance_raises(self, julia_model):
        """Test that get_max_side_spline_distance raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            julia_model.get_max_side_spline_distance()
