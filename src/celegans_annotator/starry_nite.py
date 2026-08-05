"""Load nuclei tracking data from StarryNite zip files.

StarryNite is a cell-tracking tool that outputs nuclei positions per timepoint.
Each nuclei file in the zip contains rows of space-separated values where:
    - Column 5 (0-indexed): x coordinate
    - Column 6: y coordinate
    - Column 7: z coordinate
    - Column 9: cell name

The timepoint index is extracted from the 3-digit number in the filename.
"""

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_nuclei_from_zip(
    sn_zip_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Load nuclei coordinates and labels from a StarryNite zip file.

    Args:
        sn_zip_path: Path to the StarryNite zip file containing nuclei files.
            Each file in the zip ending with 'nuclei' is parsed as a
            comma-separated table of cell detections for one timepoint.

    Returns:
        Tuple of (nuclei_coords, nuclei_labels) where:
            nuclei_coords: numpy array of shape (N, 4) with columns
                [time, z, y, x]
            nuclei_labels: list of N cell name strings
    """
    sn_zip_path = Path(sn_zip_path)

    all_coords = []
    all_labels = []

    with zipfile.ZipFile(sn_zip_path, "r") as zf:
        nuclei_files = [f for f in zf.namelist() if f.endswith("nuclei")]
        logger.info("Found %d nuclei files in %s", len(nuclei_files), sn_zip_path.name)

        for filename in nuclei_files:
            with zf.open(filename) as f:
                # Extract timepoint index from the 3-digit number in filename
                match = re.search(r"\d{3}", filename)
                if match is None:
                    logger.warning("No 3-digit index in filename: %s", filename)
                    continue
                # StarryNite uses 1-based indexing, convert to 0-based
                time_idx = int(match.group()) - 1

                raw = f.read().decode("utf-8")
                rows = raw.split(" \n")
                rows = [row.split(", ") for row in rows]

                content = pd.DataFrame(rows).dropna(axis=0, thresh=4)
                content = content[content[9] != ""]
                # Columns: 7=z, 6=y, 5=x, 9=cell_name
                content = content[[7, 6, 5, 9]]
                content.set_index(9, inplace=True)
                content = content.astype("float64")

                coords = np.array(content)
                time_column = np.full((len(coords), 1), time_idx)
                coords = np.hstack((time_column, coords))

                all_coords.append(coords)
                all_labels.extend(content.index.tolist())

    if not all_coords:
        return np.empty((0, 4)), []

    nuclei_coords = np.vstack(all_coords)
    logger.info(
        "Loaded %d nuclei across %d timepoints",
        len(nuclei_coords),
        len(all_coords),
    )
    return nuclei_coords, all_labels
