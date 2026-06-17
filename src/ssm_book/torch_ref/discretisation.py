r"""PyTorch port of the discretisation rules (book section 2.3).

Mirrors :mod:`ssm_book.numpy.discretisation`; checked against it in
``tests/test_discretisation.py``.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["zoh_discretise", "bilinear_discretise"]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _as_AB(A, B):
    A = torch.atleast_2d(_t(A))
    B = _t(B)
    if B.ndim == 1:
        B = B.unsqueeze(1)
    return A, B


def zoh_discretise(A, B, dt):
    A, B = _as_AB(A, B)
    N, p = A.shape[0], B.shape[1]
    M = torch.zeros((N + p, N + p), dtype=torch.complex128)
    M[:N, :N] = A
    M[:N, N:] = B
    E = torch.linalg.matrix_exp(M * dt)
    return E[:N, :N], E[:N, N:]


def bilinear_discretise(A, B, dt):
    A, B = _as_AB(A, B)
    N = A.shape[0]
    I = torch.eye(N, dtype=torch.complex128)
    M = torch.linalg.inv(I - (dt / 2) * A)
    Abar = M @ (I + (dt / 2) * A)
    Bbar = M @ (dt * B)
    return Abar, Bbar
