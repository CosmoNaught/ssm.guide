r"""PyTorch port of the diagonal state space model (book section 3.5).

Mirrors :mod:`ssm_book.numpy_ref.diagonal`; checked against it in
``tests/test_diagonal.py``.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "diagonal_kernel",
    "diagonal_recurrence",
    "s4d_lin_init",
    "s4d_inv_init",
]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _vec(x):
    return _t(x).reshape(-1)


def diagonal_kernel(lambdas, weights, L):
    r"""Length-``L`` diagonal kernel :math:`\bar K_m=\sum_n w_n\bar\lambda_n^{\,m}`."""
    lambdas, weights = _vec(lambdas), _vec(weights)
    m = torch.arange(L)
    V = lambdas[:, None] ** m[None, :].to(torch.complex128)  # V_{nm} = lambda_n ** m
    return weights @ V


def diagonal_recurrence(lambdas, Bbar, C, u):
    r"""Run :math:`x_{k+1}=\operatorname{diag}(\lambda)x_k+\bar B u_k,\ y_k=Cx_{k+1}`."""
    lambdas, Bbar, C, u = _vec(lambdas), _vec(Bbar), _vec(C), _vec(u)
    N, L = lambdas.shape[0], u.shape[0]
    x = torch.zeros(N, dtype=torch.complex128)
    ys = []
    for k in range(L):
        x = lambdas * x + Bbar * u[k]
        ys.append(C @ x)
    return torch.stack(ys)


def s4d_lin_init(M):
    r"""S4D-Lin eigenvalues :math:`\lambda_n=-\tfrac12+i\pi n`, ``n=0..M-1``."""
    n = torch.arange(M, dtype=torch.float64)
    return (-0.5 + 1j * np.pi * n).to(torch.complex128)


def s4d_inv_init(M):
    r"""S4D-Inv eigenvalues, ``n=0..M-1`` (book section 3.5.6)."""
    n = torch.arange(M, dtype=torch.float64)
    imag = (M / np.pi) * (M / (2 * n + 1) - 1)
    return (-0.5 + 1j * imag).to(torch.complex128)
