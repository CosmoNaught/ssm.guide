r"""PyTorch port of the DPLR matrix and Woodbury resolvent (book section 3.3).

Mirrors :mod:`ssm_book.numpy_ref.structured_matrices`.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["make_dplr", "woodbury_resolvent", "hippo_nplr"]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _diag(Lambda):
    return _t(Lambda).reshape(-1)


def _factor(P):
    P = _t(P)
    return P.reshape(-1, 1) if P.ndim == 1 else P


def make_dplr(Lambda, P, Q):
    r"""Dense DPLR matrix :math:`\diag(\Lambda)-PQ^*`."""
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    return torch.diag(Lambda) - P @ Q.conj().T


def woodbury_resolvent(s, Lambda, P, Q):
    r"""Resolvent :math:`(sI-\diag(\Lambda)+PQ^*)^{-1}` via the Woodbury identity."""
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    N, r = P.shape
    s = _t(s).reshape(())

    d = 1.0 / (s - Lambda)                  # diagonal of D^{-1}
    Dinv_P = d.unsqueeze(1) * P             # D^{-1} P, shape (N, r)
    Qs_Dinv = Q.conj().T * d                # Q* D^{-1}, shape (r, N)
    core = torch.eye(r, dtype=torch.complex128) + Qs_Dinv @ P
    correction = Dinv_P @ torch.linalg.solve(core, Qs_Dinv)
    return torch.diag(d) - correction


def hippo_nplr(N):
    r"""DPLR factors :math:`(\Lambda, \tilde p, \tilde q)` of the HiPPO-LegS matrix."""
    n = torch.arange(N, dtype=torch.float64)
    row = n.reshape(-1, 1)
    col = n.reshape(1, -1)
    H = torch.zeros((N, N), dtype=torch.float64)
    lower = row > col
    H[lower] = torch.sqrt((2 * row + 1) * (2 * col + 1)).to(torch.float64)[lower]
    H[torch.eye(N, dtype=torch.bool)] = n + 1
    A = (-H).to(torch.complex128)

    p = 0.5 * torch.sqrt(2 * n + 1.0)
    q = torch.sqrt(2 * n + 1.0)
    S = A + torch.outer(p, q).to(torch.complex128)

    w = (S + 0.5 * torch.eye(N, dtype=torch.complex128))  # real skew-symmetric part
    eigvals, V = torch.linalg.eigh(1j * w)                # Hermitian: V unitary
    Lambda = -0.5 + 1j * eigvals.to(torch.complex128)
    p_tilde = V.conj().T @ p.to(torch.complex128)
    q_tilde = V.conj().T @ q.to(torch.complex128)
    return Lambda, p_tilde, q_tilde
