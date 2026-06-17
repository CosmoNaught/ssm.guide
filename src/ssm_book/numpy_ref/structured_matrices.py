r"""Diagonal-plus-low-rank matrices and the Woodbury resolvent (book section 3.3).

S4 replaces the dense HiPPO state matrix by a diagonal-plus-low-rank (DPLR) form

.. math::

    A=\diag(\Lambda)-PQ^*,
    \qquad
    \Lambda\in\C^{N},\ P,Q\in\C^{N\times r},\ r\ll N,

so the resolvent :math:`(sI-A)^{-1}` the kernel algorithm needs at many ``s``
reduces, via the Woodbury identity, to a diagonal inverse plus an
:math:`r\times r` solve:

.. math::

    (D+PQ^*)^{-1}
    =D^{-1}-D^{-1}P\,(I_r+Q^*D^{-1}P)^{-1}Q^*D^{-1},
    \qquad D=sI-\diag(\Lambda).

NumPy is the source of truth; the PyTorch and JAX ports mirror it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_dplr", "woodbury_resolvent", "hippo_nplr"]


def _diag(Lambda):
    return np.asarray(Lambda, dtype=complex).reshape(-1)


def _factor(P):
    P = np.asarray(P, dtype=complex)
    return P[:, None] if P.ndim == 1 else P


def make_dplr(Lambda, P, Q):
    r"""Dense DPLR matrix :math:`\diag(\Lambda)-PQ^*`.

    ``Lambda`` is ``(N,)`` and ``P``, ``Q`` are ``(N, r)`` (a 1-D vector is read
    as ``(N, 1)``). The star is the conjugate transpose.
    """
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    return np.diag(Lambda) - P @ Q.conj().T


def woodbury_resolvent(s, Lambda, P, Q):
    r"""Resolvent :math:`(sI-\diag(\Lambda)+PQ^*)^{-1}` via the Woodbury identity.

    Returns the dense ``(N, N)`` inverse of ``s*I - make_dplr(Lambda, P, Q)``.
    ``s`` must be off the spectrum so that :math:`s-\lambda_n\ne 0`.
    """
    Lambda = _diag(Lambda)
    P, Q = _factor(P), _factor(Q)
    N, r = P.shape

    d = 1.0 / (s - Lambda)            # diagonal of D^{-1}
    Dinv_P = d[:, None] * P           # D^{-1} P, shape (N, r)
    Qs_Dinv = Q.conj().T * d          # Q* D^{-1}, shape (r, N)
    core = np.eye(r, dtype=complex) + Qs_Dinv @ P  # I_r + Q* D^{-1} P
    correction = Dinv_P @ np.linalg.solve(core, Qs_Dinv)
    return np.diag(d) - correction


def hippo_nplr(N):
    r"""DPLR factors :math:`(\Lambda, \tilde p, \tilde q)` of the HiPPO-LegS matrix.

    The HiPPO-LegS state matrix :math:`A_{\HiPPO}=-H_N` is normal plus rank one,
    :math:`A_{\HiPPO}=S-pq^\top` with :math:`S=A_{\HiPPO}+pq^\top` normal and

    .. math::

        p_n=\tfrac12\sqrt{2n+1},\qquad q_n=\sqrt{2n+1}.

    Diagonalising the normal part unitarily, :math:`S=V\Lambda V^*`, moves the
    system into DPLR coordinates: ``make_dplr(Lambda, p_tilde, q_tilde)`` equals
    :math:`V^*A_{\HiPPO}V=\Lambda-\tilde p\tilde q^*` with
    :math:`\tilde p=V^*p`, :math:`\tilde q=V^*q`. Returns
    ``(Lambda (N,), p_tilde (N,), q_tilde (N,))``.
    """
    n = np.arange(N)
    # Positive lower-triangular HiPPO-LegS matrix H_N.
    row, col = np.meshgrid(n, n, indexing="ij")
    H = np.zeros((N, N), dtype=complex)
    H[row > col] = np.sqrt((2 * row[row > col] + 1) * (2 * col[row > col] + 1))
    H[row == col] = n + 1
    A = -H  # A_HiPPO

    p = 0.5 * np.sqrt(2 * n + 1.0)
    q = np.sqrt(2 * n + 1.0)
    S = A + np.outer(p, q)  # normal: symmetric part is -I/2, rest skew

    # S + I/2 is real skew-symmetric, so i*(S + I/2) is Hermitian and eigh
    # yields a numerically unitary V (more stable than eig on S directly).
    w = S + 0.5 * np.eye(N)                 # real skew-symmetric part
    eigvals, V = np.linalg.eigh(1j * w)     # Hermitian: V unitary, eigvals real
    Lambda = -0.5 + 1j * eigvals            # eigenvalues of S
    p_tilde = V.conj().T @ p.astype(complex)
    q_tilde = V.conj().T @ q.astype(complex)
    return Lambda, p_tilde, q_tilde
