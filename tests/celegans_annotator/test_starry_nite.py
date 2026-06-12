"""Tests for celegans_annotator.starry_nite module."""

import zipfile
from pathlib import Path

from celegans_annotator.starry_nite import load_nuclei_from_zip


def _make_nuclei_content(rows: list[list[str]]) -> str:
    """Format rows into StarryNite nuclei file content.

    StarryNite nuclei files are space-newline delimited, with fields
    comma-space separated. We need at least 10 columns, with:
        column 5 (0-indexed): x coordinate
        column 6: y coordinate
        column 7: z coordinate
        column 9: cell name
    """
    lines = []
    for row in rows:
        # Pad to 10+ columns if needed
        while len(row) < 10:
            row.append("")
        lines.append(", ".join(row))
    return " \n".join(lines) + " \n"


def _make_test_zip(tmp_path: Path) -> Path:
    """Create a minimal StarryNite zip file for testing.

    Contains two timepoints (001, 002) with known nuclei.
    """
    zip_path = tmp_path / "test_SN.zip"

    # Timepoint 001 (will become t=0): P0 at z=10, y=20, x=30
    # Columns: 0  1  2  3  4  5(x) 6(y) 7(z) 8  9(name)
    tp1_rows = [
        ["0", "0", "0", "0", "0", "30", "20", "10", "0", "P0"],
        ["0", "0", "0", "0", "0", "35", "25", "15", "0", "AB"],
    ]

    # Timepoint 002 (will become t=1): AB and P1
    tp2_rows = [
        ["0", "0", "0", "0", "0", "32", "22", "12", "0", "AB"],
        ["0", "0", "0", "0", "0", "38", "28", "18", "0", "P1"],
    ]

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("t001-nuclei", _make_nuclei_content(tp1_rows))
        zf.writestr("t002-nuclei", _make_nuclei_content(tp2_rows))

    return zip_path


class TestLoadNucleiFromZip:
    def test_loads_correct_shape(self, tmp_path):
        """Loaded coordinates should be (N, 4) with [time, z, y, x]."""
        zip_path = _make_test_zip(tmp_path)
        coords, labels = load_nuclei_from_zip(zip_path)

        assert coords.shape == (4, 4)  # 2 nuclei per timepoint * 2 timepoints
        assert len(labels) == 4

    def test_time_indices(self, tmp_path):
        """Timepoint indices should be 0-based (001 -> 0, 002 -> 1)."""
        zip_path = _make_test_zip(tmp_path)
        coords, _labels = load_nuclei_from_zip(zip_path)

        times = coords[:, 0]
        assert 0.0 in times
        assert 1.0 in times

    def test_coordinate_order(self, tmp_path):
        """Columns should be [time, z, y, x]."""
        zip_path = _make_test_zip(tmp_path)
        coords, labels = load_nuclei_from_zip(zip_path)

        # Find P0 (t=0, z=10, y=20, x=30)
        p0_idx = labels.index("P0")
        p0_coords = coords[p0_idx]
        assert p0_coords[0] == 0.0  # time
        assert p0_coords[1] == 10.0  # z (column 7)
        assert p0_coords[2] == 20.0  # y (column 6)
        assert p0_coords[3] == 30.0  # x (column 5)

    def test_labels_match_coords(self, tmp_path):
        """Each label should correspond to its coordinate row."""
        zip_path = _make_test_zip(tmp_path)
        coords, labels = load_nuclei_from_zip(zip_path)

        assert len(labels) == len(coords)

        # P1 should be at t=1
        p1_idx = labels.index("P1")
        assert coords[p1_idx, 0] == 1.0

    def test_empty_zip(self, tmp_path):
        """Empty zip should return empty arrays."""
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass

        coords, labels = load_nuclei_from_zip(zip_path)
        assert coords.shape == (0, 4)
        assert labels == []

    def test_accepts_string_path(self, tmp_path):
        """Should accept string paths in addition to Path objects."""
        zip_path = _make_test_zip(tmp_path)
        _coords, labels = load_nuclei_from_zip(str(zip_path))
        assert len(labels) > 0
