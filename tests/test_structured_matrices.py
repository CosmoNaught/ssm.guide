r"""The DPLR resolvent of book section 3.3: the Woodbury identity reproduces the
dense inverse of ``s*I - (diag(Lambda) - P Q*)``, in every backend.

    woodbury_resolvent(s, ...) == inv(s*I - make_dplr(...))

and the three backends agree with one another. The HiPPO-LegS factors from
``hippo_nplr`` reconstruct the (transformed) HiPPO state matrix.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import structured_matrices as snp
from ssm_book.torch_ref import structured_matrices as st
from ssm_book.jax_ref import structured_matrices as sj


def random_dplr(N=6, r=2, seed=0):
    """Random complex DPLR factors and a few shift values off the spectrum."""
    rng = np.random.default_rng(seed)

    def cplx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    Lambda = cplx(N)
    P = cplx(N, r)
    Q = cplx(N, r)
    # Shifts kept away from the eigenvalues of diag(Lambda) - P Q*.
    A = snp.make_dplr(Lambda, P, Q)
    spec = np.linalg.eigvals(A)
    s_values = [3.0 + 2.0j, -4.0 - 1.0j, 0.5j, 10.0]
    for s in s_values:
        assert np.min(np.abs(s - spec)) > 1e-2
    return Lambda, P, Q, s_values


def test_woodbury_matches_dense_inverse_numpy():
    Lambda, P, Q, s_values = random_dplr()
    A = snp.make_dplr(Lambda, P, Q)
    N = A.shape[0]
    for s in s_values:
        dense = np.linalg.inv(s * np.eye(N) - A)
        assert np.allclose(snp.woodbury_resolvent(s, Lambda, P, Q), dense)


def test_woodbury_is_a_true_resolvent_numpy():
    # (s*I - A) @ resolvent == I
    Lambda, P, Q, s_values = random_dplr(seed=4)
    A = snp.make_dplr(Lambda, P, Q)
    N = A.shape[0]
    for s in s_values:
        R = snp.woodbury_resolvent(s, Lambda, P, Q)
        assert np.allclose((s * np.eye(N) - A) @ R, np.eye(N))


def test_rank_one_and_nonsquare_factors_numpy():
    # 1-D vectors are read as (N, 1); rank r != 1 also works.
    Lambda, _, _, s_values = random_dplr(seed=7)
    rng = np.random.default_rng(7)
    N = Lambda.shape[0]
    p = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    q = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    A = snp.make_dplr(Lambda, p, q)
    s = 5.0 + 5.0j
    dense = np.linalg.inv(s * np.eye(N) - A)
    assert np.allclose(snp.woodbury_resolvent(s, Lambda, p, q), dense)


def test_backends_agree_make_dplr_and_resolvent():
    Lambda, P, Q, s_values = random_dplr(seed=1)
    A_np = snp.make_dplr(Lambda, P, Q)
    A_t = to_numpy(st.make_dplr(Lambda, P, Q))
    A_j = np.asarray(sj.make_dplr(Lambda, P, Q))
    assert np.allclose(A_np, A_t)
    assert np.allclose(A_np, A_j)

    for s in s_values:
        R_np = snp.woodbury_resolvent(s, Lambda, P, Q)
        R_t = to_numpy(st.woodbury_resolvent(s, Lambda, P, Q))
        R_j = np.asarray(sj.woodbury_resolvent(s, Lambda, P, Q))
        assert np.allclose(R_np, R_t)
        assert np.allclose(R_np, R_j)


def test_hippo_nplr_reconstructs_transformed_state_matrix():
    # In DPLR coordinates, diag(Lambda) - p_tilde q_tilde* is V* A_HiPPO V,
    # which is similar to A_HiPPO and so has the same eigenvalues: -1,...,-N.
    for N in (4, 8):
        Lambda, p, q = snp.hippo_nplr(N)
        A_dplr = snp.make_dplr(Lambda, p, q)
        eig = np.sort(np.linalg.eigvals(A_dplr).real)
        assert np.allclose(eig, -(np.arange(N) + 1)[::-1], atol=1e-8)
        # Eigenvalues are real (Hermitian-derived), so imag parts vanish.
        assert np.max(np.abs(np.linalg.eigvals(A_dplr).imag)) < 1e-8


def test_hippo_nplr_backends_agree():
    for N in (4, 6):
        L_np, p_np, q_np = snp.hippo_nplr(N)
        L_t, p_t, q_t = st.hippo_nplr(N)
        L_j, p_j, q_j = sj.hippo_nplr(N)
        # The DPLR matrix is basis-dependent (eigenvector sign/order), but its
        # eigenvalues are not, so compare those.
        A_np = snp.make_dplr(L_np, p_np, q_np)
        A_t = to_numpy(st.make_dplr(L_t, p_t, q_t))
        A_j = np.asarray(sj.make_dplr(L_j, p_j, q_j))
        e_np = np.sort(np.linalg.eigvals(A_np).real)
        e_t = np.sort(np.linalg.eigvals(A_t).real)
        e_j = np.sort(np.linalg.eigvals(A_j).real)
        assert np.allclose(e_np, e_t)
        assert np.allclose(e_np, e_j)
