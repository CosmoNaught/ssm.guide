r"""The S4 kernel, the dense way and the structured way (book section 3.4).

The S4 layer is the discrete model :math:`x_{k+1}=\bar A x_k+\bar B u_k`,
:math:`y_k=Cx_{k+1}`, whose kernel is :math:`\bar K_m=C\bar A^m\bar B`. The
continuous-time state matrix is **diagonal plus low rank** (DPLR),

.. math::

    A=\Lambda-PQ^*,\qquad \Lambda=\operatorname{diag}(\lambda_0,\dots,\lambda_{N-1}),
    \quad P,Q\in\mathbb C^{N\times r},

and :func:`bilinear_discretise_dplr` turns it into :math:`(\bar A,\bar B)`.

Two routines compute the same length-:math:`L` kernel:

* :func:`dense_kernel` forms :math:`\bar A,\bar B` explicitly and iterates
  :math:`\bar K_m=C\bar A^m\bar B`. This is the :math:`O(LN^2)` definition and the
  source of truth.
* :func:`structured_kernel` follows the S4 pipeline of book section 3.4. It
  evaluates the finite generating function at the :math:`L` roots of unity using
  the bilinear map :math:`s(z)`, the diagonal (Cauchy) resolvent, and the
  Woodbury low-rank correction, then inverse-FFTs back to the time domain. The
  expensive per-node work touches only diagonals and :math:`r\times r` inverses;
  it never forms an :math:`N\times N` matrix.

For a DPLR system the two agree to machine precision, which
``tests/test_s4_kernel.py`` checks.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "dplr_matrix",
    "bilinear_discretise_dplr",
    "dense_kernel",
    "cauchy_dot",
    "woodbury_resolvent_vector",
    "generating_function",
    "structured_kernel",
]


def _vec(v):
    return np.asarray(v, dtype=complex).reshape(-1)


def _factors(Lambda, P, Q, B, C):
    r"""Coerce the DPLR ingredients to complex arrays of consistent shape."""
    Lambda = _vec(Lambda)
    N = Lambda.shape[0]
    P = np.asarray(P, dtype=complex).reshape(N, -1)
    Q = np.asarray(Q, dtype=complex).reshape(N, -1)
    return Lambda, P, Q, _vec(B), _vec(C)


def dplr_matrix(Lambda, P, Q):
    r"""Assemble the dense DPLR matrix :math:`A=\Lambda-PQ^*`."""
    Lambda = _vec(Lambda)
    N = Lambda.shape[0]
    P = np.asarray(P, dtype=complex).reshape(N, -1)
    Q = np.asarray(Q, dtype=complex).reshape(N, -1)
    return np.diag(Lambda) - P @ Q.conj().T


def bilinear_discretise_dplr(Lambda, P, Q, B, dt):
    r"""Bilinear (Tustin) discretisation of the DPLR system :math:`A=\Lambda-PQ^*`.

    Returns ``(Abar, Bbar)`` with
    :math:`\bar A=(I-\tfrac{\mathrm{dt}}{2}A)^{-1}(I+\tfrac{\mathrm{dt}}{2}A)` and
    :math:`\bar B=(I-\tfrac{\mathrm{dt}}{2}A)^{-1}\,\mathrm{dt}\,B`. The dense
    inverse here is only for the teaching reference; the structured kernel never
    forms it.
    """
    Lambda, P, Q, B, _ = _factors(Lambda, P, Q, B, B)
    A = dplr_matrix(Lambda, P, Q)
    N = A.shape[0]
    I = np.eye(N, dtype=complex)
    M = np.linalg.inv(I - (dt / 2) * A)
    Abar = M @ (I + (dt / 2) * A)
    Bbar = M @ (dt * B)
    return Abar, Bbar


def dense_kernel(Lambda, P, Q, B, C, dt, L):
    r"""The length-``L`` S4 kernel by the dense definition :math:`\bar K_m=C\bar A^m\bar B`.

    Forms :math:`(\bar A,\bar B)` from the bilinear discretisation of the DPLR
    matrix and iterates the recurrence. This is the :math:`O(LN^2)` reference.
    """
    Lambda, P, Q, B, C = _factors(Lambda, P, Q, B, C)
    Abar, Bbar = bilinear_discretise_dplr(Lambda, P, Q, B, dt)
    K = np.empty(L, dtype=complex)
    v = Bbar.copy()  # v_0 = Bbar
    for m in range(L):
        K[m] = C @ v
        v = Abar @ v
    return K


def cauchy_dot(w, s, Lambda):
    r"""The Cauchy sum :math:`\sum_n w_n/(s-\lambda_n)` for a scalar node ``s``.

    With :math:`w_n=a_n b_n` this evaluates :math:`a^\top(sI-\Lambda)^{-1}b`, the
    building block of every diagonal resolvent term in the S4 pipeline.
    """
    w, Lambda = _vec(w), _vec(Lambda)
    return np.sum(w / (s - Lambda))


def woodbury_resolvent_vector(s, Lambda, P, Q, B):
    r"""The vector :math:`(sI-A)^{-1}B` for :math:`A=\Lambda-PQ^*` at one node ``s``.

    Uses the Woodbury identity to avoid the :math:`N\times N` inverse:

    .. math::

        (sI-A)^{-1}
        =D_s-D_sP\,(I_r+Q^*D_sP)^{-1}Q^*D_s,
        \qquad D_s=(sI-\Lambda)^{-1},

    so only the diagonal :math:`D_s` and a small :math:`r\times r` inverse appear.
    """
    Lambda, B = _vec(Lambda), _vec(B)
    N, r = Lambda.shape[0], P.shape[1]
    Ds = 1.0 / (s - Lambda)  # diagonal of (sI - Lambda)^{-1}
    Qstar = Q.conj().T  # r x N
    DsB = Ds * B  # D_s B
    DsP = Ds[:, None] * P  # D_s P, shape (N, r)
    inner = np.eye(r, dtype=complex) + Qstar @ DsP  # I_r + Q* D_s P
    correction = DsP @ np.linalg.solve(inner, Qstar @ DsB)
    return DsB - correction


def generating_function(Lambda, P, Q, B, C, dt, L):
    r"""Evaluate the finite generating function at the ``L`` roots of unity.

    Returns the vector :math:`\hat G_L(\omega_j)=\sum_m \bar K_m\omega_j^m` for
    :math:`\omega_j=e^{-2\pi i j/L}`. Each sample is

    .. math::

        \hat G_L(\omega_j)
        =\tilde C\,\frac{2}{1+\omega_j}(s_jI-A)^{-1}B,
        \qquad \tilde C=C\,(I-\bar A^L),\quad s_j=\frac{2}{\mathrm{dt}}\frac{1-\omega_j}{1+\omega_j},

    with :math:`(s_jI-A)^{-1}B` supplied by :func:`woodbury_resolvent_vector`.
    The truncation row :math:`\tilde C` is formed once. The singular node
    :math:`\omega_j=-1` (present when ``L`` is even) is handled by the limit
    :math:`\tfrac{2}{1+z}(s(z)I-A)^{-1}B\to\tfrac{\mathrm{dt}}{2}B`.
    """
    Lambda, P, Q, B, C = _factors(Lambda, P, Q, B, C)
    Abar, _ = bilinear_discretise_dplr(Lambda, P, Q, B, dt)

    # Truncation row vector Ctilde = C (I - Abar^L). Formed once, O(N^3 log L).
    AbarL = np.linalg.matrix_power(Abar, L)
    Ctilde = C @ (np.eye(Abar.shape[0], dtype=complex) - AbarL)

    omega = np.exp(-2j * np.pi * np.arange(L) / L)  # L-th roots of unity
    G = np.empty(L, dtype=complex)
    for j in range(L):
        z = omega[j]
        if np.abs(1.0 + z) < 1e-12:
            # z = -1: the bilinear map s(z) blows up; use the finite limit.
            resB = (dt / 2.0) * B
        else:
            s = (2.0 / dt) * (1.0 - z) / (1.0 + z)  # bilinear map s(z)
            resB = (2.0 / (1.0 + z)) * woodbury_resolvent_vector(s, Lambda, P, Q, B)
        G[j] = Ctilde @ resB
    return G


def structured_kernel(Lambda, P, Q, B, C, dt, L):
    r"""The length-``L`` S4 kernel by the structured pipeline of section 3.4.

    Evaluates :func:`generating_function` at the roots of unity, then takes the
    inverse FFT. Because :math:`\hat G_L(\omega_j)=\sum_m\bar K_m\omega_j^m` with
    :math:`\omega_j=e^{-2\pi i j/L}` is exactly the forward DFT of the kernel, the
    inverse FFT returns :math:`(\bar K_0,\dots,\bar K_{L-1})`.
    """
    G = generating_function(Lambda, P, Q, B, C, dt, L)
    return np.fft.ifft(G)
