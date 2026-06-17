r"""JAX port of the S4 kernel, dense and structured (book section 3.4).

Mirrors :mod:`ssm_book.numpy_ref.s4_reference`: the dense kernel
:math:`\bar K_m=C\bar A^m\bar B` from the bilinear discretisation of the DPLR
matrix :math:`A=\Lambda-PQ^*`, and the structured kernel via the generating
function (bilinear map, Woodbury low-rank resolvent, inverse FFT).
"""
from __future__ import annotations

import jax.numpy as jnp

# 64-bit precision is enabled by ssm_book.jax_ref/__init__ on package import.

__all__ = [
    "bilinear_discretise_dplr",
    "dense_kernel",
    "woodbury_resolvent_vector",
    "generating_function",
    "structured_kernel",
]


def _vec(x):
    return jnp.asarray(x, dtype=jnp.complex128).reshape(-1)


def _mat(x, N):
    return jnp.asarray(x, dtype=jnp.complex128).reshape(N, -1)


def dplr_matrix(Lambda, P, Q):
    r"""Assemble the dense DPLR matrix :math:`A=\Lambda-PQ^*`."""
    Lambda = _vec(Lambda)
    N = Lambda.shape[0]
    P, Q = _mat(P, N), _mat(Q, N)
    return jnp.diag(Lambda) - P @ Q.conj().T


def bilinear_discretise_dplr(Lambda, P, Q, B, dt):
    r"""Bilinear (Tustin) discretisation of :math:`A=\Lambda-PQ^*`, returning ``(Abar, Bbar)``."""
    Lambda = _vec(Lambda)
    N = Lambda.shape[0]
    B = _vec(B)
    A = dplr_matrix(Lambda, P, Q)
    I = jnp.eye(N, dtype=jnp.complex128)
    M = jnp.linalg.inv(I - (dt / 2) * A)
    Abar = M @ (I + (dt / 2) * A)
    Bbar = M @ (dt * B)
    return Abar, Bbar


def dense_kernel(Lambda, P, Q, B, C, dt, L):
    r"""The length-``L`` kernel by the dense definition :math:`\bar K_m=C\bar A^m\bar B`."""
    C = _vec(C)
    Abar, Bbar = bilinear_discretise_dplr(Lambda, P, Q, B, dt)
    v = Bbar
    ks = []
    for _ in range(L):
        ks.append(C @ v)
        v = Abar @ v
    return jnp.stack(ks)


def woodbury_resolvent_vector(s, Lambda, P, Q, B):
    r"""The vector :math:`(sI-A)^{-1}B` for :math:`A=\Lambda-PQ^*` via Woodbury."""
    Lambda, B = _vec(Lambda), _vec(B)
    N = Lambda.shape[0]
    P, Q = _mat(P, N), _mat(Q, N)
    r = P.shape[1]
    Ds = 1.0 / (s - Lambda)  # diagonal of (sI - Lambda)^{-1}
    Qstar = Q.conj().T  # r x N
    DsB = Ds * B
    DsP = Ds[:, None] * P  # N x r
    inner = jnp.eye(r, dtype=jnp.complex128) + Qstar @ DsP
    correction = DsP @ jnp.linalg.solve(inner, Qstar @ DsB)
    return DsB - correction


def generating_function(Lambda, P, Q, B, C, dt, L):
    r"""Finite generating function at the ``L`` roots of unity (see NumPy reference)."""
    Lambda, B, C = _vec(Lambda), _vec(B), _vec(C)
    N = Lambda.shape[0]
    Abar, _ = bilinear_discretise_dplr(Lambda, P, Q, B, dt)

    AbarL = jnp.linalg.matrix_power(Abar, L)
    Ctilde = C @ (jnp.eye(N, dtype=jnp.complex128) - AbarL)

    omega = jnp.exp(-2j * jnp.pi * jnp.arange(L) / L)
    gs = []
    for jj in range(L):
        z = omega[jj]
        if jnp.abs(1.0 + z) < 1e-12:
            resB = (dt / 2.0) * B
        else:
            s = (2.0 / dt) * (1.0 - z) / (1.0 + z)
            resB = (2.0 / (1.0 + z)) * woodbury_resolvent_vector(s, Lambda, P, Q, B)
        gs.append(Ctilde @ resB)
    return jnp.stack(gs)


def structured_kernel(Lambda, P, Q, B, C, dt, L):
    r"""The length-``L`` S4 kernel by the structured pipeline, via inverse FFT."""
    G = generating_function(Lambda, P, Q, B, C, dt, L)
    return jnp.fft.ifft(G)
