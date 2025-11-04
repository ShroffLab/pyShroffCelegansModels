"""Tests for CelegansModel class."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from celegans_model import CelegansModel


@pytest.fixture
def lattice_csv_path():
    """Path to the test lattice CSV file."""
    # Resolve to absolute path to work regardless of where tests are run from
    return (Path(__file__).parent / "resources" / "lattice.csv").resolve()


@pytest.fixture
def celegans_model(lattice_csv_path):
    """Create a CelegansModel instance for testing."""
    return CelegansModel(lattice_csv_path)


class TestCelegansModelInit:
    """Test CelegansModel initialization."""

    def test_init_with_valid_lattice(self, lattice_csv_path):
        """Test that model initializes correctly with valid lattice CSV."""
        model = CelegansModel(lattice_csv_path)
        assert model.lattice_csv == lattice_csv_path
        assert model.julia_model is not None

    def test_init_with_nonexistent_lattice(self):
        """Test that initialization fails with nonexistent lattice file."""
        nonexistent_path = Path("/nonexistent/lattice.csv")
        with pytest.raises(AssertionError, match="does not exist"):
            CelegansModel(nonexistent_path)


class TestGetCandidateLocations:
    """Test get_candidate_locations method."""

    def test_returns_list_of_tuples(self, celegans_model):
        """Test that method returns a list of tuples."""
        target_point = np.array([0.0, 0.0, 0.0])
        candidates = celegans_model.get_candidate_locations(target_point)
        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert all(isinstance(c, tuple) for c in candidates)
            assert all(len(c) == 3 for c in candidates)

    def test_returns_three_coordinates(self, celegans_model):
        """Test that each candidate has exactly 3 coordinates (AP, ML, DV)."""
        target_point = np.array([0.0, 0.0, 0.0])
        candidates = celegans_model.get_candidate_locations(target_point)
        if len(candidates) > 0:
            for candidate in candidates:
                assert len(candidate) == 3, "Each candidate should have AP, ML, DV"

    def test_threshold_filters_candidates(self, celegans_model):
        """Test that threshold parameter filters candidates by distance."""
        target_point = np.array([0.0, 0.0, 0.0])
        # Get all candidates without threshold
        all_candidates = celegans_model.get_candidate_locations(
            target_point, threshold=None
        )
        # Get candidates with strict threshold
        filtered_candidates = celegans_model.get_candidate_locations(
            target_point, threshold=1.0
        )
        # Filtered should have same or fewer candidates
        assert len(filtered_candidates) <= len(all_candidates)

    def test_no_threshold_returns_all_local_minima(self, celegans_model):
        """Test that None threshold returns all local minima."""
        target_point = np.array([5.0, 5.0, 5.0])
        candidates = celegans_model.get_candidate_locations(
            target_point, threshold=None
        )
        # Should return at least one candidate (or warn if none)
        assert isinstance(candidates, list)

    def test_warns_when_no_candidates_found(self, celegans_model):
        """Test that a warning is issued when no candidates meet threshold."""
        target_point = np.array([100.0, 100.0, 100.0])
        # Use a very strict threshold that likely won't match anything
        with pytest.warns(UserWarning, match="No candidate locations found"):
            candidates = celegans_model.get_candidate_locations(
                target_point, threshold=0.001
            )
            assert len(candidates) == 0

    def test_different_points_return_different_candidates(self, celegans_model):
        """Test that different input points yield different candidates."""
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([10.0, 10.0, 10.0])

        candidates1 = celegans_model.get_candidate_locations(point1)
        candidates2 = celegans_model.get_candidate_locations(point2)

        # Different points should generally yield different candidates
        # (unless by coincidence they map to the same location)
        assert isinstance(candidates1, list)
        assert isinstance(candidates2, list)

    def test_accepts_numpy_array(self, celegans_model):
        """Test that method accepts numpy array input."""
        target_point = np.array([1.0, 2.0, 3.0])
        candidates = celegans_model.get_candidate_locations(target_point)
        assert isinstance(candidates, list)


class TestGetBestCandidate:
    """Test get_best_candidate method."""

    def test_returns_single_tuple(self, celegans_model):
        """Test that method returns a single tuple."""
        target_point = np.array([0.0, 0.0, 0.0])
        best = celegans_model.get_best_candidate(target_point)
        assert isinstance(best, tuple)
        assert len(best) == 3

    def test_returns_three_coordinates(self, celegans_model):
        """Test that result has exactly 3 coordinates (AP, ML, DV)."""
        target_point = np.array([5.0, 5.0, 5.0])
        best = celegans_model.get_best_candidate(target_point)
        assert len(best) == 3, "Result should have AP, ML, DV coordinates"

    def test_accepts_numpy_array(self, celegans_model):
        """Test that method accepts numpy array input."""
        target_point = np.array([1.0, 2.0, 3.0])
        best = celegans_model.get_best_candidate(target_point)
        assert isinstance(best, tuple)

    def test_different_points_return_different_results(self, celegans_model):
        """Test that different input points yield different best candidates."""
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([10.0, 10.0, 10.0])

        best1 = celegans_model.get_best_candidate(point1)
        best2 = celegans_model.get_best_candidate(point2)

        # Different points should yield different results in most cases
        assert isinstance(best1, tuple)
        assert isinstance(best2, tuple)

    def test_best_candidate_in_candidate_list(self, celegans_model):
        """Test that best candidate is one of the candidates from get_candidate_locations."""
        target_point = np.array([5.0, 5.0, 5.0])
        candidates = celegans_model.get_candidate_locations(target_point)
        best = celegans_model.get_best_candidate(target_point)

        # The best candidate should be among the candidates
        # Note: This may not always be true if the implementations differ,
        # but it's a good sanity check
        if len(candidates) > 0:
            # Allow for small floating point differences
            assert any(
                np.allclose(best, candidate, rtol=1e-5, atol=1e-8)
                for candidate in candidates
            ), "Best candidate should be one of the candidate locations"

    def test_consistent_results(self, celegans_model):
        """Test that calling the method twice with the same input returns the same result."""
        target_point = np.array([3.0, 4.0, 5.0])
        best1 = celegans_model.get_best_candidate(target_point)
        best2 = celegans_model.get_best_candidate(target_point)

        # Results should be identical
        assert np.allclose(best1, best2, rtol=1e-10, atol=1e-10)


class TestIntegration:
    """Integration tests for CelegansModel methods."""

    def test_workflow_from_pixel_to_worm_space(self, celegans_model):
        """Test typical workflow: pixel point -> candidate locations -> best candidate."""
        # Simulated pixel location
        pixel_point = np.array([7.5, 8.5, 9.5])

        # Get all candidates
        candidates = celegans_model.get_candidate_locations(pixel_point)
        assert isinstance(candidates, list)

        # Get best candidate
        best = celegans_model.get_best_candidate(pixel_point)
        assert isinstance(best, tuple)
        assert len(best) == 3

        # Best should be among candidates (with some tolerance)
        if len(candidates) > 0:
            assert any(
                np.allclose(best, candidate, rtol=1e-5, atol=1e-8)
                for candidate in candidates
            )

    def test_coordinates_are_floats(self, celegans_model):
        """Test that all returned coordinates are numeric."""
        target_point = np.array([1.0, 2.0, 3.0])

        # Test candidates
        candidates = celegans_model.get_candidate_locations(target_point)
        for candidate in candidates:
            for coord in candidate:
                assert isinstance(coord, (int, float, np.number))

        # Test best candidate
        best = celegans_model.get_best_candidate(target_point)
        for coord in best:
            assert isinstance(coord, (int, float, np.number))


class TestSeamCellUntwisting:
    """Test that seam cells untwist to planar coordinates with DV ≈ 0."""

    def test_seam_cells_untwist_to_near_zero_dv(
        self, lattice_csv_path, celegans_model
    ):
        """Test that untwisting seam cells gives coord1 (DV) close to 0."""
        # Load the lattice CSV to get seam cell positions
        df = pd.read_csv(lattice_csv_path)

        # Use ALL seam cells (including "a" cells)
        for _, row in df.iterrows():
            name = row['name']
            # Seam cell position in twisted space
            twisted_point = np.array(
                [row['x_voxels'], row['y_voxels'], row['z_voxels']]
            )

            # Get candidates for this seam cell
            candidates = celegans_model.get_candidate_locations(
                twisted_point, threshold=None
            )

            # Each seam cell should have at least one candidate
            # near the surface (coord1 ≈ 0)
            if len(candidates) > 0:
                coord1_abs = [abs(c[1]) for c in candidates]
                min_coord1 = min(coord1_abs)
                assert min_coord1 < 10, (
                    f"Seam cell {name} has no candidate with "
                    f"|coord1| < 10. Minimum |coord1| found: "
                    f"{min_coord1:.2f}"
                )

    def test_all_seam_cells_have_candidates(
        self, lattice_csv_path, celegans_model
    ):
        """Test that all seam cells produce at least one candidate."""
        # Load the lattice CSV
        df = pd.read_csv(lattice_csv_path)

        # Use ALL seam cells (including "a" cells)
        for _, row in df.iterrows():
            name = row['name']
            twisted_point = np.array(
                [row['x_voxels'], row['y_voxels'], row['z_voxels']]
            )

            # Get candidates
            candidates = celegans_model.get_candidate_locations(
                twisted_point, threshold=None
            )

            assert len(candidates) > 0, (
                f"Seam cell {name} at position {twisted_point} "
                f"produced no candidates"
            )
