r"""Transfer function and generating function of the discrete kernel (book section 2.5).

The convolution kernel :math:`\bar K_m=C\bar A^m\bar B` of a discrete state space
model has a generating function

.. math::

    G(z)=\sum_{m\ge 0}\bar K_m\,z^m=C\,(I-z\bar A)^{-1}\bar B,

which converges for :math:`|z|<1/\rho(\bar A)`. Setting :math:`z=e^{-2\pi i j/L}`
samples :math:`G` on the ``L`` roots of unity; an inverse FFT of those samples
then recovers the kernel, exactly up to the aliasing
:math:`\tilde K_m=\sum_{p\ge 0}\bar K_{m+pL}`, which is negligible for a strongly
stable system.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "resolvent",
    "generating_function",
    "evaluate_on_roots_of_unity",
    "recover_kernel_by_ifft",
]


def _vec(v):
    return np.asarray(v, dtype=complex).reshape(-1)


def resolvent(z, Abar):
    r"""The resolvent :math:`(I-z\bar A)^{-1}`."""
    Abar = np.asarray(Abar, dtype=complex)
    N = Abar.shape[0]
    I = np.eye(N, dtype=complex)
    return np.linalg.inv(I - z * Abar)


def generating_function(z, Abar, Bbar, C):
    r"""The scalar generating function :math:`G(z)=C\,(I-z\bar A)^{-1}\bar B`."""
    Bbar, C = _vec(Bbar), _vec(C)
    return C @ (resolvent(z, Abar) @ Bbar)


def evaluate_on_roots_of_unity(Abar, Bbar, C, L):
    r"""Sample ``G`` on the ``L`` roots of unity :math:`\omega_j=e^{-2\pi i j/L}`."""
    j = np.arange(L)
    omega = np.exp(-2j * np.pi * j / L)
    return np.array(
        [generating_function(z, Abar, Bbar, C) for z in omega], dtype=complex
    )


def recover_kernel_by_ifft(samples):
    r"""Recover the kernel from generating-function samples via the inverse FFT.

    Exact up to the aliasing :math:`\tilde K_m=\sum_{p\ge 0}\bar K_{m+pL}`.
    """
    return np.fft.ifft(_vec(samples))
