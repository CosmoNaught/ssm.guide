r"""JAX port of the affine composition and the parallel scan
(book section 3.5.9).

Mirrors :mod:`ssm_book.numpy_ref.scan`, but the parallel prefix scan is left to
``jax.lax.associative_scan``, which combines the affine pairs in a balanced tree.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Importing this module imports the ``ssm_book.jax_ref`` package first, whose
# __init__ enables 64-bit precision.

__all__ = ["affine_compose", "associative_scan", "sequential_recurrence"]


def _pair(a, c):
    a = jnp.asarray(a, dtype=jnp.complex128).reshape(-1)
    c = jnp.asarray(c, dtype=jnp.complex128).reshape(-1)
    if a.shape != c.shape:
        raise ValueError("a and c must have the same length")
    return a, c


def affine_compose(left, right):
    r"""Compose two affine steps: :math:`(a,c)\bullet(a',c')=(a a',\ a c'+c)`.

    ``left`` is the later step, ``right`` the earlier one. Each is a pair of
    arrays, so the same function serves as the binary operator for
    :func:`jax.lax.associative_scan`.
    """
    a, c = left
    ap, cp = right
    return a * ap, a * cp + c


def associative_scan(a, c):
    r"""Inclusive prefix scan of the affine steps, returning the states.

    Uses :func:`jax.lax.associative_scan` to combine the pairs
    :math:`(a_k,c_k)` under :math:`\bullet`, then returns the offset component,
    which is the state sequence :math:`x_1,\dots,x_L` for :math:`x_0=0`.

    ``jax.lax.associative_scan`` calls the binary operator with its earlier
    operand first, so the arguments are swapped to match the convention of
    :func:`affine_compose` (later step on the left).
    """
    a, c = _pair(a, c)

    def combine(earlier, later):
        return affine_compose(later, earlier)

    _, offsets = jax.lax.associative_scan(combine, (a, c))
    return offsets


def sequential_recurrence(a, c):
    r"""Reference loop via :func:`jax.lax.scan`: :math:`x_{k+1}=a_k x_k+c_k`."""
    a, c = _pair(a, c)

    def step(state, ac):
        ak, ck = ac
        new = ak * state + ck
        return new, new

    _, xs = jax.lax.scan(step, jnp.zeros((), dtype=jnp.complex128), (a, c))
    return xs
