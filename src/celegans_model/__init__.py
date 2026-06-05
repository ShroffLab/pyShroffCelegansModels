"""C. elegans worm space coordinate transformation models.

This package provides implementations for converting between pixel coordinates
and worm-relative coordinates (AP, ML, DV axes). Two implementations are available:

- JuliaCelegansModel: Julia-backed implementation (requires Julia and juliacall)
- PythonCelegansModel: Pure Python/scipy implementation

Both implementations inherit from CelegansModelBase and provide the same core
interface for coordinate transformations.

Example usage:

    # Using Python implementation (no Julia required)
    from celegans_model import PythonCelegansModel
    import numpy as np

    lattice = np.random.rand(11, 3, 2)  # (points, xyz, left/right)
    model = PythonCelegansModel(lattice)
    worm_coords = model.get_best_candidate(np.array([100, 200, 50]))

    # Using Julia implementation
    from celegans_model import JuliaCelegansModel
    from pathlib import Path

    model = JuliaCelegansModel(Path("lattice.csv"))
    worm_coords = model.get_best_candidate(np.array([100, 200, 50]))
"""

from typing import TYPE_CHECKING

from .base import CelegansModelBase
from .constants import STANDARD_SEAM_CELLS

if TYPE_CHECKING:
    from .julia_celegans_model import JuliaCelegansModel
    from .python_celegans_model import PythonCelegansModel


def __getattr__(name: str):
    """Lazy import of implementation classes.

    This allows importing the package without requiring all dependencies.
    Julia dependencies are only loaded when JuliaCelegansModel is accessed.
    Scipy dependencies are only loaded when PythonCelegansModel is accessed.
    """
    if name == "JuliaCelegansModel":
        from .julia_celegans_model import JuliaCelegansModel

        return JuliaCelegansModel

    if name == "PythonCelegansModel":
        from .python_celegans_model import PythonCelegansModel

        return PythonCelegansModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Constants
    "STANDARD_SEAM_CELLS",
    # Abstract base
    "CelegansModelBase",
    # Implementations
    "JuliaCelegansModel",
    "PythonCelegansModel",
]
