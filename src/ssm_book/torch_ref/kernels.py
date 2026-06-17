r"""PyTorch port of recurrence, kernel, and convolution (book section 2.4).

Mirrors :mod:`ssm_book.numpy.kernels`.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["ssm_recurrence", "ssm_kernel", "causal_conv", "toeplitz", "fft_conv"]


def _t(x):
    if isinstance(x, torch.Tensor):
        return x.to(torch.complex128)
    return torch.tensor(np.asarray(x, dtype=np.complex128).tolist(), dtype=torch.complex128)


def _vec(x):
    return _t(x).reshape(-1)


def ssm_recurrence(Abar, Bbar, C, u):
    Abar = _t(Abar)
    Bbar, C, u = _vec(Bbar), _vec(C), _vec(u)
    N, L = Abar.shape[0], u.shape[0]
    x = torch.zeros(N, dtype=torch.complex128)
    ys = []
    for k in range(L):
        x = Abar @ x + Bbar * u[k]
        ys.append(C @ x)
    return torch.stack(ys)


def ssm_kernel(Abar, Bbar, C, L):
    Abar = _t(Abar)
    Bbar, C = _vec(Bbar), _vec(C)
    v = Bbar.clone()
    ks = []
    for _ in range(L):
        ks.append(C @ v)
        v = Abar @ v
    return torch.stack(ks)


def causal_conv(K, u):
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    ys = []
    for k in range(L):
        ys.append(torch.dot(K[: k + 1], torch.flip(u[: k + 1], dims=(0,))))
    return torch.stack(ys)


def toeplitz(K, L):
    K = _vec(K)
    T = torch.zeros((L, L), dtype=torch.complex128)
    for i in range(L):
        for j in range(i + 1):
            T[i, j] = K[i - j]
    return T


def fft_conv(K, u):
    K, u = _vec(K), _vec(u)
    L = u.shape[0]
    n = 1
    while n < 2 * L:
        n *= 2
    y = torch.fft.ifft(torch.fft.fft(K, n) * torch.fft.fft(u, n))[:L]
    return y
