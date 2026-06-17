r"""The affine composition and the parallel scan (book section 3.5.9).

A diagonal mode runs the scalar recurrence

.. math::

    x_{k+1}=a_k\,x_k+c_k,

where the multiplier :math:`a_k` is the (possibly time-varying) discrete
eigenvalue and the offset :math:`c_k` carries the current input. Writing each
step as the pair :math:`(a,c)`, two consecutive steps compose into one affine
map,

.. math::

    (a,c)\bullet(a',c')=(a\,a',\ a\,c'+c),

with :math:`(a,c)` the later step and :math:`(a',c')` the earlier one. The
operation :math:`\bullet` is associative, so the running offsets

.. math::

    x_1,\;x_2,\;\dots,\;x_L

(starting from :math:`x_0=0`) can be produced by an associative, or parallel
prefix, scan rather than a left-to-right loop. This module gives both: an actual
log-step scan (:func:`associative_scan`) and the sequential reference loop
(:func:`sequential_recurrence`) it is checked against.
"""
from __future__ import annotations

import numpy as np

__all__ = ["affine_compose", "associative_scan", "sequential_recurrence"]


def _pair(a, c):
    a = np.asarray(a, dtype=complex).reshape(-1)
    c = np.asarray(c, dtype=complex).reshape(-1)
    if a.shape != c.shape:
        raise ValueError("a and c must have the same length")
    return a, c


def affine_compose(left, right):
    r"""Compose two affine steps: :math:`(a,c)\bullet(a',c')=(a a',\ a c'+c)`.

    Each argument is a pair ``(a, c)`` of arrays. ``left`` is the later step,
    ``right`` the earlier one, matching the order in which the maps are applied.
    """
    a, c = left
    ap, cp = right
    return a * ap, a * cp + c


def associative_scan(a, c):
    r"""Inclusive prefix scan of the affine steps, returning the states.

    Combines the per-step pairs :math:`(a_k,c_k)` with the associative operator
    :math:`\bullet` in :math:`O(\log L)` rounds (a Hillis--Steele log-step scan)
    and returns the offset component of every prefix, which is the state
    sequence :math:`x_1,\dots,x_L` for :math:`x_0=0`.
    """
    a, c = _pair(a, c)
    L = a.shape[0]
    if L == 0:
        return c.copy()
    # Running pair arrays; index k holds the inclusive prefix up to step k.
    A = a.copy()
    C = c.copy()
    shift = 1
    while shift < L:
        # Combine prefix ending at k with the prefix ending at k - shift.
        a_lo, c_lo = A[shift:], C[shift:]
        a_hi, c_hi = A[:-shift], C[:-shift]
        a_new, c_new = affine_compose((a_lo, c_lo), (a_hi, c_hi))
        A = np.concatenate([A[:shift], a_new])
        C = np.concatenate([C[:shift], c_new])
        shift *= 2
    return C


def sequential_recurrence(a, c):
    r"""Reference left-to-right loop: :math:`x_{k+1}=a_k x_k+c_k`, :math:`x_0=0`."""
    a, c = _pair(a, c)
    L = a.shape[0]
    x = np.zeros(L, dtype=complex)
    state = 0.0 + 0.0j
    for k in range(L):
        state = a[k] * state + c[k]
        x[k] = state
    return x
