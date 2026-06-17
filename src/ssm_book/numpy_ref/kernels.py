r"""Recurrence, convolution, and the discrete kernel (book section 2.4).

A discrete state space model can be run two ways. As a recurrence it updates a
state one step at a time; as a convolution it applies a fixed kernel
:math:`\bar K_m=C\bar A^m\bar B`. These functions implement both, plus the
Toeplitz and FFT forms of the convolution, so the equivalence

.. math::

    y_k=\sum_{m=0}^k \bar K_m\,u_{k-m}

can be checked against the recurrence directly (see
``tests/test_recurrence_equals_convolution.py``).

The output convention is post-update, :math:`y_k=C x_{k+1}`, so
:math:`\bar K_0=C\bar B`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ssm_recurrence", "ssm_kernel", "causal_conv", "toeplitz", "fft_conv"]


def _vec(v):
    return np.asarray(v, dtype=complex).reshape(-1)


def ssm_recurrence(Abar, Bbar, C, u):
    r"""Run the recurrence :math:`x_{k+1}=\bar A x_k+\bar B u_k,\ y_k=C x_{k+1}`."""
    Abar = np.asarray(Abar, dtype=complex)
    Bbar, C, u = _vec(Bbar), _vec(C), _vec(u)
    N, L = Abar.shape[0], u.shape[0]
    x = np.zeros(N, dtype=complex)
    y = np.empty(L, dtype=complex)
    for k in range(L):
        x = Abar @ x + Bbar * u[k]
        y[k] = C @ x
    return y


def ssm_kernel(Abar, Bbar, C, L):
    r"""The length-``L`` kernel :math:`\bar K_m=C\bar A^m\bar B`."""
    Abar = np.asarray(Abar, dtype=complex)
    Bbar, C = _vec(Bbar), _vec(C)
    K = np.empty(L, dtype=complex)
    v = Bbar.copy()  # v_0 = Bbar
    for m in range(L):
        K[m] = C @ v
        v = Abar @ v
    return K


def causal_conv(K, u):
    r"""Direct causal convolution :math:`y_k=\sum_{m=0}^k K_m u_{k-m}`."""
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    y = np.zeros(L, dtype=complex)
    for k in range(L):
        y[k] = np.dot(K[: k + 1], u[k::-1])
    return y


def toeplitz(K, L):
    r"""Lower-triangular Toeplitz convolution matrix ``T`` with ``T @ u == causal_conv``."""
    K = _vec(K)
    T = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(i + 1):
            T[i, j] = K[i - j]
    return T


def fft_conv(K, u):
    r"""Causal convolution via the FFT (truncated to length ``L``)."""
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    n = 1
    while n < 2 * L:
        n *= 2
    y = np.fft.ifft(np.fft.fft(K, n) * np.fft.fft(u, n))[:L]
    return y
