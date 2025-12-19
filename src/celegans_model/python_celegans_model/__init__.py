"""Pure Python implementation of C. elegans worm space coordinate transformations.

This subpackage provides a Python-only implementation of the worm coordinate
system model, without requiring Julia. It includes:

- PythonCelegansModel: Main class for worm space coordinate transformations
- CubicSpline3D: Helper class for 3D spline interpolation
- compute_central_spline_csv: Compute splines from CSV lattice annotations
- compute_central_splines: Compute splines from zarr lattice arrays
- dist_to_spline: Distance calculation utilities
"""

from .compute_central_spline import compute_central_spline_csv, compute_central_splines
from .cubic_spline_3d import CubicSpline3D
from .dist_to_spline import dist_to_spline
from .python_celegans_model import PythonCelegansModel

__all__ = [
    "CubicSpline3D",
    "PythonCelegansModel",
    "compute_central_spline_csv",
    "compute_central_splines",
    "dist_to_spline",
]
