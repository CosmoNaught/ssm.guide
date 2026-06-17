r"""Scalar state space recurrence (book section 2.1).

.. math::

    x_{k+1}=a\,x_k+b\,u_k,\qquad y_k=c\,x_{k+1}.

With :math:`|a|<1` the influence of each input decays geometrically. The
multivariable version appears in section 2.4.
"""
from __future__ import annotations

import numpy as np

__all__ = ["scalar_memory"]


def scalar_memory(u, a=0.9, b=1.0, c=1.0):
    r"""Run :math:`x_{k+1}=a x_k+b u_k,\ y_k=c x_{k+1}` from rest and return ``y``.

    Starts at :math:`x_0=0`; each output reads the state after the current
    input updates it.
    """
    u = np.asarray(u, dtype=complex).reshape(-1)
    L = u.shape[0]
    y = np.empty(L, dtype=complex)
    x = 0.0 + 0.0j
    for k in range(L):
        x = a * x + b * u[k]
        y[k] = c * x
    return y
