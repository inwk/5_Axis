"""CAM package.

Submodules are intentionally not imported at package import time to avoid
side effects from NXOpen session access during module initialization.
"""

__all__ = [
    "geometry",
    "measurements",
    "operations",
    "session",
    "utils",
]
