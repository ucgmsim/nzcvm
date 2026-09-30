"""Velocity model, mesh I/O, and surface interpolation utilities.

This subpackage contains the core data representations and I/O routines for
NZCVM velocity models:

:mod:`nzcvm.models.mesh`
    VTKHDF-backed mesh I/O for tetrahedral and structured grids.

:mod:`nzcvm.models.surface`
    Surface interpolation for topography-based depth transforms.

:mod:`nzcvm.models.model`
    High-level Python wrappers around the compiled Rust velocity-model backend.
"""

from nzcvm.models import mesh, model, surface

__all__ = ["mesh", "model", "surface"]
