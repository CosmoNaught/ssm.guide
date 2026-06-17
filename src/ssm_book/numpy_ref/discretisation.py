r"""Discretise a continuous-time state space model (book section 2.3).

Turns :math:`x'(t)=Ax(t)+Bu(t)` into :math:`x_{k+1}=\bar A x_k+\bar B u_k` for step
size ``dt`` under two inter-sample assumptions: zero-order hold
(:func:`zoh_discretise`) and the bilinear/Tustin transform
(:func:`bilinear_discretise`).

Both return complex arrays so the same code serves the real and the complex
(diagonal) models used later in the book.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ["zoh_discretise", "bilinear_discretise"]


def _as_AB(A, B):
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    B = np.asarray(B, dtype=complex)
    if B.ndim == 1:
        B = B[:, None]
    return A, B


def zoh_discretise(A, B, dt):
    r"""Zero-order hold discretisation.

    Returns ``(Abar, Bbar)`` with :math:`\bar A=e^{A\,\mathrm{dt}}` and
    :math:`\bar B=\big(\int_0^{\mathrm{dt}} e^{A\tau}\,d\tau\big)B`. The integral is
    evaluated with the augmented-matrix identity, which stays correct even when
    ``A`` is singular::

        expm([[A, B], [0, 0]] * dt) = [[Abar, Bbar], [0, I]].
    """
    A, B = _as_AB(A, B)
    N, p = A.shape[0], B.shape[1]
    M = np.zeros((N + p, N + p), dtype=complex)
    M[:N, :N] = A
    M[:N, N:] = B
    E = expm(M * dt)
    return E[:N, :N], E[:N, N:]


def bilinear_discretise(A, B, dt):
    r"""Bilinear (Tustin) discretisation.

    Returns ``(Abar, Bbar)`` with
    :math:`\bar A=(I-\tfrac{\mathrm{dt}}{2}A)^{-1}(I+\tfrac{\mathrm{dt}}{2}A)` and
    :math:`\bar B=(I-\tfrac{\mathrm{dt}}{2}A)^{-1}\,\mathrm{dt}\,B`.
    """
    A, B = _as_AB(A, B)
    N = A.shape[0]
    I = np.eye(N, dtype=complex)
    M = np.linalg.inv(I - (dt / 2) * A)
    Abar = M @ (I + (dt / 2) * A)
    Bbar = M @ (dt * B)
    return Abar, Bbar
