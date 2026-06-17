r"""PyTorch port of the affine composition and the parallel scan
(book section 3.5.9).

Mirrors :mod:`ssm_book.numpy_ref.scan`.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["affine_compose", "associative_scan", "sequential_recurrence"]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _pair(a, c):
    a, c = _t(a).reshape(-1), _t(c).reshape(-1)
    if a.shape != c.shape:
        raise ValueError("a and c must have the same length")
    return a, c


def affine_compose(left, right):
    r"""Compose two affine steps: :math:`(a,c)\bullet(a',c')=(a a',\ a c'+c)`."""
    a, c = left
    ap, cp = right
    return a * ap, a * cp + c


def associative_scan(a, c):
    r"""Inclusive log-step prefix scan of the affine steps, returning the states.

    Combines the pairs :math:`(a_k,c_k)` with :math:`\bullet` in
    :math:`O(\log L)` rounds and returns the offsets, i.e. the state sequence
    :math:`x_1,\dots,x_L` for :math:`x_0=0`.
    """
    a, c = _pair(a, c)
    L = a.shape[0]
    if L == 0:
        return c.clone()
    A = a.clone()
    C = c.clone()
    shift = 1
    while shift < L:
        a_lo, c_lo = A[shift:], C[shift:]
        a_hi, c_hi = A[:-shift], C[:-shift]
        a_new, c_new = affine_compose((a_lo, c_lo), (a_hi, c_hi))
        A = torch.cat([A[:shift], a_new])
        C = torch.cat([C[:shift], c_new])
        shift *= 2
    return C


def sequential_recurrence(a, c):
    r"""Reference left-to-right loop: :math:`x_{k+1}=a_k x_k+c_k`, :math:`x_0=0`."""
    a, c = _pair(a, c)
    L = a.shape[0]
    state = torch.zeros((), dtype=torch.complex128)
    xs = []
    for k in range(L):
        state = a[k] * state + c[k]
        xs.append(state)
    return torch.stack(xs) if xs else torch.zeros(0, dtype=torch.complex128)
