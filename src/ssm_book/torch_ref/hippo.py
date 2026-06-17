r"""PyTorch port of the HiPPO-LegS matrix (book section 3.2).

Mirrors :mod:`ssm_book.numpy_ref.hippo`. See that module for the convention:
:func:`legs_matrix` returns the stable state matrix
:math:`A_{\mathrm{HiPPO}}=-H_N`, and the input vector has entries
:math:`(b_N)_n=\sqrt{2n+1}`. Everything is built in ``complex128`` for parity
with the NumPy reference.
"""
from __future__ import annotations

import torch

__all__ = ["legs_hippo", "legs_matrix", "legs_input"]


def legs_hippo(N):
    r"""The positive HiPPO-LegS matrix :math:`H_N`, as a ``complex128`` tensor."""
    n = torch.arange(N, dtype=torch.float64)
    r = torch.sqrt(2 * n + 1.0)  # r_n = sqrt(2n+1)
    H = r[:, None] * r[None, :]  # H_{nk} = sqrt((2n+1)(2k+1))
    H = torch.tril(H, diagonal=-1)  # strictly-lower part
    H = H + torch.diag(n + 1.0)  # diagonal H_{nn} = n+1
    return H.to(torch.complex128)


def legs_matrix(N):
    r"""The stable HiPPO-LegS state matrix :math:`A_{\mathrm{HiPPO}}=-H_N`."""
    return -legs_hippo(N)


def legs_input(N):
    r"""The HiPPO-LegS input vector :math:`b_N`, with :math:`(b_N)_n=\sqrt{2n+1}`."""
    n = torch.arange(N, dtype=torch.float64)
    return torch.sqrt(2 * n + 1.0).to(torch.complex128)
