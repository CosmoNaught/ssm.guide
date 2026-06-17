r"""PyTorch port of the transfer / generating function (book section 2.5).

Mirrors :mod:`ssm_book.numpy_ref.transfer`.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "resolvent",
    "generating_function",
    "evaluate_on_roots_of_unity",
    "recover_kernel_by_ifft",
]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _vec(x):
    return _t(x).reshape(-1)


def resolvent(z, Abar):
    r"""The resolvent :math:`(I-z\bar A)^{-1}`."""
    Abar = _t(Abar)
    N = Abar.shape[0]
    I = torch.eye(N, dtype=torch.complex128)
    z = _t(z)
    return torch.linalg.inv(I - z * Abar)


def generating_function(z, Abar, Bbar, C):
    r"""The scalar generating function :math:`G(z)=C\,(I-z\bar A)^{-1}\bar B`."""
    Bbar, C = _vec(Bbar), _vec(C)
    return C @ (resolvent(z, Abar) @ Bbar)


def evaluate_on_roots_of_unity(Abar, Bbar, C, L):
    r"""Sample ``G`` on the ``L`` roots of unity :math:`\omega_j=e^{-2\pi i j/L}`."""
    j = torch.arange(L, dtype=torch.float64)
    omega = torch.exp(-2j * np.pi * j.to(torch.complex128) / L)
    return torch.stack([generating_function(omega[k], Abar, Bbar, C) for k in range(L)])


def recover_kernel_by_ifft(samples):
    r"""Recover the kernel from generating-function samples via the inverse FFT.

    Exact up to the aliasing :math:`\tilde K_m=\sum_{p\ge 0}\bar K_{m+pL}`.
    """
    return torch.fft.ifft(_vec(samples))
