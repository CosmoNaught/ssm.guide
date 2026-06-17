r"""The diagonal state space model and its kernel (book section 3.5).

S4D removes the low-rank correction of S4 and keeps a purely diagonal state
matrix :math:`A=\operatorname{diag}(\lambda_0,\dots,\lambda_{N-1})`. Each state
coordinate then contributes a single geometric mode, so the length-``L`` kernel
is the Vandermonde sum

.. math::

    \bar K_m=\sum_{n=0}^{N-1}w_n\,\bar\lambda_n^{\,m},

with weights :math:`w_n=C_n\bar B_n` (book section 3.5.3). The same map can be
run as a recurrence over the diagonal state.

Eigenvalues are chosen, not drawn at random: :func:`s4d_lin_init` (S4D-Lin,
linearly spaced) and :func:`s4d_inv_init` (S4D-Inv, inverse spaced) give two
half-plane initialisations (book section 3.5.6).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "diagonal_kernel",
    "diagonal_recurrence",
    "s4d_lin_init",
    "s4d_inv_init",
]


def _vec(v):
    return np.asarray(v, dtype=complex).reshape(-1)


def diagonal_kernel(lambdas, weights, L):
    r"""Length-``L`` diagonal kernel :math:`\bar K_m=\sum_n w_n\bar\lambda_n^{\,m}`.

    With the Vandermonde matrix :math:`V_{nm}=\bar\lambda_n^{\,m}` this is the
    weighted row sum :math:`\bar K=w^\top V`.
    """
    lambdas, weights = _vec(lambdas), _vec(weights)
    m = np.arange(L)
    V = lambdas[:, None] ** m[None, :]
    return weights @ V


def diagonal_recurrence(lambdas, Bbar, C, u):
    r"""Run the diagonal recurrence.

    :math:`x_{k+1}=\operatorname{diag}(\lambda)\,x_k+\bar B\,u_k,\quad y_k=Cx_{k+1}`,
    using post-update outputs so :math:`\bar K_0=C\bar B`.
    """
    lambdas, Bbar, C, u = _vec(lambdas), _vec(Bbar), _vec(C), _vec(u)
    N, L = lambdas.shape[0], u.shape[0]
    x = np.zeros(N, dtype=complex)
    y = np.empty(L, dtype=complex)
    for k in range(L):
        x = lambdas * x + Bbar * u[k]
        y[k] = C @ x
    return y


def s4d_lin_init(M):
    r"""S4D-Lin eigenvalues :math:`\lambda_n=-\tfrac12+i\pi n`, ``n=0..M-1``."""
    n = np.arange(M)
    return -0.5 + 1j * np.pi * n


def s4d_inv_init(M):
    r"""S4D-Inv eigenvalues.

    :math:`\lambda_n=-\tfrac12+i\,\tfrac{M}{\pi}\!\left(\tfrac{M}{2n+1}-1\right)`,
    ``n=0..M-1`` (book section 3.5.6).
    """
    n = np.arange(M)
    return -0.5 + 1j * (M / np.pi) * (M / (2 * n + 1) - 1)
