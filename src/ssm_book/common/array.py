"""Convert a NumPy / PyTorch / JAX array to a plain NumPy array.

PyTorch wheels are sometimes built against a different NumPy ABI than the one
installed, which breaks ``tensor.numpy()`` / ``np.asarray(tensor)``. Going
through ``tensor.tolist()`` avoids that bridge entirely, so the book's
equivalence checks work regardless of the exact NumPy/PyTorch build.
"""
from __future__ import annotations

import numpy as np

__all__ = ["to_numpy"]


def to_numpy(x):
    """Return ``x`` as a NumPy array, without relying on the NumPy/Torch C-API."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return np.array(x.detach().cpu().tolist())
    except ImportError:
        pass
    return np.asarray(x)
