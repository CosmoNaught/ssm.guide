r"""JAX port of the discretisation rules (book section 2.3).

Mirrors :mod:`ssm_book.numpy.discretisation`.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import expm

# Importing this module imports the ``ssm_book.jax`` package first, whose
# __init__ enables 64-bit precision.

__all__ = ["zoh_discretise", "bilinear_discretise"]


def _as_AB(A, B):
    A = jnp.atleast_2d(jnp.asarray(A, dtype=jnp.complex128))
    B = jnp.asarray(B, dtype=jnp.complex128)
    if B.ndim == 1:
        B = B[:, None]
    return A, B


def zoh_discretise(A, B, dt):
    A, B = _as_AB(A, B)
    N, p = A.shape[0], B.shape[1]
    M = jnp.zeros((N + p, N + p), dtype=jnp.complex128)
    M = M.at[:N, :N].set(A)
    M = M.at[:N, N:].set(B)
    E = expm(M * dt)
    return E[:N, :N], E[:N, N:]


def bilinear_discretise(A, B, dt):
    A, B = _as_AB(A, B)
    N = A.shape[0]
    I = jnp.eye(N, dtype=jnp.complex128)
    M = jnp.linalg.inv(I - (dt / 2) * A)
    Abar = M @ (I + (dt / 2) * A)
    Bbar = M @ (dt * B)
    return Abar, Bbar
