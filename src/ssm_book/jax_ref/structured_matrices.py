r"""JAX port of the DPLR matrix and Woodbury resolvent (book section 3.3).

Mirrors :mod:`ssm_book.numpy_ref.structured_matrices`.
"""
from __future__ import annotations

import jax.numpy as jnp

# Importing this module imports the ``ssm_book.jax_ref`` package first, whose
# __init__ enables 64-bit precision.

__all__ = ["make_dplr", "woodbury_resolvent", "hippo_nplr"]


def _diag(Lambda):
    return jnp.asarray(Lambda, dtype=jnp.complex128).reshape(-1)


def _factor(P):
    P = jnp.asarray(P, dtype=jnp.complex128)
    return P.reshape(-1, 1) if P.ndim == 1 else P


def make_dplr(Lambda, P, Q):
    r"""Dense DPLR matrix :math:`\diag(\Lambda)-PQ^*`."""
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    return jnp.diag(Lambda) - P @ Q.conj().T


def woodbury_resolvent(s, Lambda, P, Q):
    r"""Resolvent :math:`(sI-\diag(\Lambda)+PQ^*)^{-1}` via the Woodbury identity."""
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    N, r = P.shape

    d = 1.0 / (s - Lambda)                  # diagonal of D^{-1}
    Dinv_P = d[:, None] * P                 # D^{-1} P, shape (N, r)
    Qs_Dinv = Q.conj().T * d                # Q* D^{-1}, shape (r, N)
    core = jnp.eye(r, dtype=jnp.complex128) + Qs_Dinv @ P
    correction = Dinv_P @ jnp.linalg.solve(core, Qs_Dinv)
    return jnp.diag(d) - correction


def hippo_nplr(N):
    r"""DPLR factors :math:`(\Lambda, \tilde p, \tilde q)` of the HiPPO-LegS matrix."""
    n = jnp.arange(N, dtype=jnp.float64)
    row = n.reshape(-1, 1)
    col = n.reshape(1, -1)
    lower = row > col
    H = jnp.where(lower, jnp.sqrt((2 * row + 1) * (2 * col + 1)), 0.0)
    H = H + jnp.diag(n + 1)
    A = (-H).astype(jnp.complex128)

    p = 0.5 * jnp.sqrt(2 * n + 1.0)
    q = jnp.sqrt(2 * n + 1.0)
    S = A + jnp.outer(p, q).astype(jnp.complex128)

    w = S + 0.5 * jnp.eye(N, dtype=jnp.complex128)  # real skew-symmetric part
    eigvals, V = jnp.linalg.eigh(1j * w)            # Hermitian: V unitary
    Lambda = -0.5 + 1j * eigvals.astype(jnp.complex128)
    p_tilde = V.conj().T @ p.astype(jnp.complex128)
    q_tilde = V.conj().T @ q.astype(jnp.complex128)
    return Lambda, p_tilde, q_tilde
