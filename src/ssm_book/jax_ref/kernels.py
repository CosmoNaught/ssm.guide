r"""JAX port of recurrence, kernel, and convolution (book section 2.4).

Mirrors :mod:`ssm_book.numpy.kernels`.
"""
from __future__ import annotations

import jax.numpy as jnp

# Importing this module imports the ``ssm_book.jax`` package first, whose
# __init__ enables 64-bit precision.

__all__ = ["ssm_recurrence", "ssm_kernel", "causal_conv", "toeplitz", "fft_conv"]


def _vec(x):
    return jnp.asarray(x, dtype=jnp.complex128).reshape(-1)


def ssm_recurrence(Abar, Bbar, C, u):
    Abar = jnp.asarray(Abar, dtype=jnp.complex128)
    Bbar, C, u = _vec(Bbar), _vec(C), _vec(u)
    N, L = Abar.shape[0], u.shape[0]
    x = jnp.zeros(N, dtype=jnp.complex128)
    ys = []
    for k in range(L):
        x = Abar @ x + Bbar * u[k]
        ys.append(C @ x)
    return jnp.stack(ys)


def ssm_kernel(Abar, Bbar, C, L):
    Abar = jnp.asarray(Abar, dtype=jnp.complex128)
    Bbar, C = _vec(Bbar), _vec(C)
    v = Bbar
    ks = []
    for _ in range(L):
        ks.append(C @ v)
        v = Abar @ v
    return jnp.stack(ks)


def causal_conv(K, u):
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    ys = [jnp.dot(K[: k + 1], jnp.flip(u[: k + 1])) for k in range(L)]
    return jnp.stack(ys)


def toeplitz(K, L):
    K = _vec(K)
    rows = [
        jnp.concatenate([jnp.flip(K[: i + 1]), jnp.zeros(L - i - 1, dtype=jnp.complex128)])
        for i in range(L)
    ]
    return jnp.stack(rows)


def fft_conv(K, u):
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    n = 1
    while n < 2 * L:
        n *= 2
    y = jnp.fft.ifft(jnp.fft.fft(K, n) * jnp.fft.fft(u, n))[:L]
    return y
