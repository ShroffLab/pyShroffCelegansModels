# celegans-model

Python package for C. elegans worm space coordinate transformations.

This package provides tools for converting between pixel coordinates and worm-relative coordinates (AP, ML, DV axes). Two implementations are available:

- **JuliaCelegansModel**: Julia-backed implementation (requires Julia and juliacall)
- **PythonCelegansModel**: Pure Python/scipy implementation (no Julia required)

Both implementations inherit from `CelegansModelBase` and provide the same core interface for coordinate transformations.

## Installation

### Basic installation (Python implementation only)

```bash
pip install celegans-model[python]
```

### With Julia backend

```bash
pip install celegans-model[julia]
```

### Full installation

```bash
pip install celegans-model[all]
```

### Using pixi

```bash
# Default (Python implementation with I/O)
pixi install

# With Julia support
pixi install -e julia

# Full development environment
pixi install -e dev
```

## Usage

Both implementations accept either a CSV file or a numpy array as input.

### From CSV File

```python
from celegans_model import PythonCelegansModel, JuliaCelegansModel
from pathlib import Path

# Python implementation (no Julia required)
model = PythonCelegansModel.from_csv(Path("lattice.csv"))

# Julia implementation
model = JuliaCelegansModel(Path("lattice.csv"))

# Convert pixel coordinates to worm space
pixel_point = np.array([100.0, 200.0, 50.0])
worm_coords = model.get_best_candidate(pixel_point)
```

### From Numpy Array

```python
from celegans_model import PythonCelegansModel, JuliaCelegansModel
import numpy as np

# Create lattice points: (n_points, 2, 3) array
# Shape: (11 lattice points, right/left sides, xyz coordinates)
lattice_points = np.random.rand(11, 2, 3) * 100

# Standard seam cell names
names = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]

# Python implementation
model = PythonCelegansModel(
    lattice_points,
    parameterization="uniform",
    spacing=1.0,
    lattice_point_names=names,
)

# Julia implementation
model = JuliaCelegansModel.from_array(lattice_points, names)

# Convert pixel coordinates to worm space
pixel_point = np.array([50.0, 25.0, 10.0])
worm_coords = model.get_best_candidate(pixel_point)
# Returns (ML, DV, AP) tuple
ml, dv, ap = worm_coords
print(f"Worm coordinates: ML={ml:.2f}, DV={dv:.2f}, AP={ap:.2f}")

# Get all candidate locations (for coiled worms)
candidates = model.get_candidate_locations(pixel_point, threshold=20.0)
print(f"Found {len(candidates)} candidate locations")

# Get candidates with surface scores (Python only)
candidates_with_scores = model.get_candidate_locations(
    pixel_point, threshold=20.0, return_scores=True
)
for coords, score in candidates_with_scores:
    print(f"  {coords}: score={score:.3f}")
```

### Loading Splines from Files

```python
from celegans_model.python_celegans_model.compute_central_spline import (
    compute_central_spline_csv,
    compute_central_splines,
)
from pathlib import Path

# From CSV file (single time point)
spline = compute_central_spline_csv(Path("lattice.csv"))

# From zarr array (multiple time points)
splines = compute_central_splines(Path("lattice_array.zarr"), time_range=(0, 100))
```

## Coordinate System

The worm space coordinate system has three axes:

- **Anterior/Posterior (AP)**: Position along the worm's central axis
  - Parameterized by lattice points (0 = head, 10 = tail for standard cells)
  - Can be arc-length based or uniformly spaced

- **Medial/Lateral (ML)**: Position perpendicular to AP, in the plane of the seam cells
  - Positive = right side of worm
  - Negative = left side of worm

- **Dorsal/Ventral (DV)**: Position perpendicular to both AP and ML
  - Positive = dorsal (top)
  - Negative = ventral (bottom)

## Parameterization Modes

### Uniform Parameterization

Standard seam cells are placed at canonical integer positions (0-10). Virtual cells are interpolated between their neighboring standard cells.

```python
model = PythonCelegansModel(
    lattice_points,
    parameterization="uniform",
    spacing=1.0,  # Distance between standard cells
    lattice_point_names=names,
)
```

### Arc Length Parameterization

Points are placed based on cumulative arc length along the central spline.

```python
model = PythonCelegansModel(
    lattice_points,
    parameterization="arc_length",
    lattice_point_names=names,
)
```

## API Reference

### Coordinate Order

All methods return worm space coordinates in **(ML, DV, AP)** order:
- First element: Medial/Lateral position
- Second element: Dorsal/Ventral position
- Third element: Anterior/Posterior position

### CelegansModelBase

Abstract base class defining the interface for both implementations.

**Properties:**
- `internal_range`: Range of AP values defined by lattice points
- `valid_range`: Extended range allowing extrapolation

**Methods:**
- `get_candidate_locations(target_point, threshold=None, return_scores=False)`: Get all possible worm space locations as list of (ML, DV, AP) tuples
- `get_best_candidate(target_point)`: Get the most likely worm space location as (ML, DV, AP) tuple
- `get_worm_coords(target_point, ap)`: Get worm coords at a specific AP as (ML, DV, AP) tuple (Python only)
- `get_basis_vectors(ap)`: Get ML, DV, tangent basis vectors (Python only)
- `get_surface_score(target_point, ap, steepness=2.0)`: Get surface distance score (Python only)

### STANDARD_SEAM_CELLS

Dictionary mapping standard seam cell names to canonical positions:

```python
from celegans_model import STANDARD_SEAM_CELLS

print(STANDARD_SEAM_CELLS)
# {'a0': 0, 'h0': 1, 'h1': 2, 'h2': 3, 'v1': 4, 'v2': 5, 'v3': 6, 'v4': 7, 'v5': 8, 'v6': 9, 't': 10}
```

## Running Tests

```bash
# Run all tests (requires Julia)
pixi run test

# Run Python-only tests
pixi run test-python

# Or with pytest directly
pytest tests/ -v -k 'not julia'
```
