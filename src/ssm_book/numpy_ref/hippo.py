r"""The HiPPO-LegS matrix (book section 3.2).

The scaled Legendre (LegS) construction projects the rescaled history
:math:`f_t(r)=u(tr)` onto the shifted normalised Legendre basis on
:math:`[0,1]`. Online updating of the projection coefficients
:math:`c(t)` gives the linear system

.. math::

    c'(t)=-\tfrac{1}{t}H_N\,c(t)+\tfrac{1}{t}b_N\,u(t),

whose fixed parts are the positive lower-triangular matrix :math:`H_N` and the
input vector :math:`b_N`.

Following the chapter's sign convention, :func:`legs_matrix` returns the
*stable* state matrix

.. math::

    A_{\mathrm{HiPPO}}=-H_N,

so its eigenvalues have negative real part. The entries are

.. math::

    H_{nk}=\begin{cases}
        \sqrt{(2n+1)(2k+1)}, & n>k,\\
        n+1,                 & n=k,\\
        0,                   & n<k,
    \end{cases}

giving a diagonal of :math:`A_{\mathrm{HiPPO}}` equal to
:math:`-1,-2,\dots,-N`. The input vector has entries
:math:`(b_N)_n=\sqrt{2n+1}`.

Although the matrix is real-valued, it is assembled in ``complex128`` so the
NumPy, PyTorch, and JAX backends agree to tight tolerance, matching the rest of
the reference code.
"""
from __future__ import annotations

import numpy as np

__all__ = ["legs_hippo", "legs_matrix", "legs_input"]


def legs_hippo(N):
    r"""The positive HiPPO-LegS matrix :math:`H_N` of size ``(N, N)``.

    Lower triangular, with :math:`H_{nk}=\sqrt{(2n+1)(2k+1)}` below the diagonal
    and :math:`H_{nn}=n+1` on it.
    """
    n = np.arange(N)
    r = np.sqrt(2 * n + 1.0)
    H = r[:, None] * r[None, :]
    H = np.tril(H, k=-1)  # strictly-lower part only
    H[np.diag_indices(N)] = n + 1.0
    return H.astype(np.complex128)


def legs_matrix(N):
    r"""The stable HiPPO-LegS state matrix :math:`A_{\mathrm{HiPPO}}=-H_N`.

    The diagonal is :math:`-1,-2,\dots,-N`; the strictly-lower entries are
    :math:`-\sqrt{(2n+1)(2k+1)}`.
    """
    return -legs_hippo(N)


def legs_input(N):
    r"""The HiPPO-LegS input vector :math:`b_N`, with :math:`(b_N)_n=\sqrt{2n+1}`."""
    n = np.arange(N)
    return np.sqrt(2 * n + 1.0).astype(np.complex128)
