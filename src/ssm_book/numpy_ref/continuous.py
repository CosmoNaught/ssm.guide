r"""Continuous-time state space models: impulse response and simulation (book section 2.2).

For :math:`x'(t)=A\,x(t)+B\,u(t),\ y(t)=C\,x(t)`:

* :func:`impulse_response` evaluates the kernel :math:`h(t)=C\,e^{At}\,B`;
* :func:`simulate` discretises with a zero-order hold
  (:mod:`ssm_book.numpy_ref.discretisation`) then runs the discrete recurrence.

State is carried in ``complex128`` for parity with the discrete code.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ssm_book.numpy_ref.discretisation import zoh_discretise

__all__ = ["impulse_response", "simulate"]


def _as_ABC(A, B, C):
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    B = np.asarray(B, dtype=complex)
    if B.ndim == 1:
        B = B[:, None]
    C = np.atleast_2d(np.asarray(C, dtype=complex))
    return A, B, C


def impulse_response(A, B, C, ts):
    r"""Continuous impulse response :math:`h(t)=C\,e^{At}\,B` at each ``t`` in ``ts``.

    Returns an array of shape ``(len(ts), C.shape[0], B.shape[1])``.
    """
    A, B, C = _as_ABC(A, B, C)
    ts = np.asarray(ts, dtype=float).reshape(-1)
    return np.stack([C @ expm(A * t) @ B for t in ts])


def simulate(A, B, C, u, dt):
    r"""Simulate the continuous system on a sampled input ``u`` with step ``dt``.

    Discretises to :math:`(\bar A,\bar B)` via zero-order hold, then runs
    :math:`x_{k+1}=\bar A x_k+\bar B u_k,\ y_k=C x_{k+1}` from rest. Returns
    ``y`` with one row per input sample.
    """
    A, B, C = _as_ABC(A, B, C)
    Abar, Bbar = zoh_discretise(A, B, dt)
    u = np.asarray(u, dtype=complex).reshape(-1)
    N, L = Abar.shape[0], u.shape[0]
    x = np.zeros((N, 1), dtype=complex)
    y = np.empty((L, C.shape[0]), dtype=complex)
    for k in range(L):
        x = Abar @ x + Bbar * u[k]
        y[k] = (C @ x).reshape(-1)
    return y
