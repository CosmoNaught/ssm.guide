r"""Project a signal onto a Legendre polynomial basis (book section 3.1).

Static warm-up for the HiPPO view of memory: least-squares fit of a signal
sampled on :math:`[0,1]` to the first ``N`` Legendre polynomials, and
reconstruction from those coefficients.

The :math:`P_n` are orthogonal on :math:`[-1,1]`, so sample points are mapped
from :math:`[0,1]` to :math:`[-1,1]` before fitting.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as L

__all__ = ["legendre_project", "legendre_reconstruct"]


def legendre_project(signal_values, N):
    r"""Least-squares Legendre coefficients of a signal sampled on :math:`[0,1]`.

    ``signal_values`` are assumed to be taken at the evenly spaced points
    ``linspace(0, 1, len(signal_values))``. Returns the ``N`` coefficients of the
    degree-``< N`` least-squares fit in the Legendre basis (degrees ``0..N-1``).
    """
    y = np.asarray(signal_values, dtype=complex).reshape(-1)
    M = y.shape[0]
    x = np.linspace(-1.0, 1.0, M)  # map [0, 1] sample grid onto [-1, 1]
    coeffs = L.legfit(x, y, N - 1)
    return np.asarray(coeffs, dtype=complex)


def legendre_reconstruct(coeffs, num_points):
    r"""Evaluate the Legendre series ``coeffs`` at ``num_points`` points of :math:`[0,1]`.

    The inverse of :func:`legendre_project`: it returns the fitted signal sampled
    at ``linspace(0, 1, num_points)``.
    """
    coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)
    x = np.linspace(-1.0, 1.0, num_points)
    return L.legval(x, coeffs)
