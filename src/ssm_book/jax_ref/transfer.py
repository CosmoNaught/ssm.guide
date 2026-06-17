r"""JAX port of the transfer / generating function (book section 2.5).

Mirrors :mod:`ssm_book.numpy_ref.transfer`.
"""
from __future__ import annotations

import jax.numpy as jnp

# Importing this module imports the ``ssm_book.jax_ref`` package first, whose
# __init__ enables 64-bit precision.

__all__ = [
    "resolvent",
    "generating_function",
    "evaluate_on_roots_of_unity",
    "recover_kernel_by_ifft",
]


def _vec(x):
    return jnp.asarray(x, dtype=jnp.complex128).reshape(-1)


def resolvent(z, Abar):
    r"""The resolvent :math:`(I-z\bar A)^{-1}`."""
    Abar = jnp.asarray(Abar, dtype=jnp.complex128)
    N = Abar.shape[0]
    I = jnp.eye(N, dtype=jnp.complex128)
    return jnp.linalg.inv(I - z * Abar)


def generating_function(z, Abar, Bbar, C):
    r"""The scalar generating function :math:`G(z)=C\,(I-z\bar A)^{-1}\bar B`."""
    Bbar, C = _vec(Bbar), _vec(C)
    return C @ (resolvent(z, Abar) @ Bbar)


def evaluate_on_roots_of_unity(Abar, Bbar, C, L):
    r"""Sample ``G`` on the ``L`` roots of unity :math:`\omega_j=e^{-2\pi i j/L}`."""
    j = jnp.arange(L)
    omega = jnp.exp(-2j * jnp.pi * j / L)
    return jnp.stack([generating_function(omega[k], Abar, Bbar, C) for k in range(L)])


def recover_kernel_by_ifft(samples):
    r"""Recover the kernel from generating-function samples via the inverse FFT.

    Exact up to the aliasing :math:`\tilde K_m=\sum_{p\ge 0}\bar K_{m+pL}`.
    """
    return jnp.fft.ifft(_vec(samples))
