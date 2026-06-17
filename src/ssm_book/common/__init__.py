"""Backend-agnostic helpers shared by the NumPy, PyTorch, and JAX code."""

from .array import to_numpy

__all__ = ["to_numpy"]
