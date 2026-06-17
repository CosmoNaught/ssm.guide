r"""The HiPPO-LegS matrix of book section 3.2.

Checks that ``legs_matrix(N)`` reproduces the chapter's closed-form entries and
its explicit ``N=4`` matrix, and that the NumPy, PyTorch, and JAX backends all
agree.

Convention (from chapters/03-s4/02-hippo.qmd): with the positive lower-triangular

    H_{nk} = sqrt((2n+1)(2k+1))  for n > k,
    H_{nn} = n + 1,
    H_{nk} = 0                   for n < k,

the stable state matrix is A_HiPPO = -H_N, and the input vector is
(b_N)_n = sqrt(2n+1).
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import hippo as hnp
from ssm_book.torch_ref import hippo as ht
from ssm_book.jax_ref import hippo as hj


def closed_form_H(N):
    """Brute-force the chapter's closed form, entry by entry."""
    H = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                H[n, k] = np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                H[n, k] = n + 1
            else:
                H[n, k] = 0.0
    return H


def test_legs_matrix_matches_closed_form():
    for N in (1, 2, 4, 7):
        A = to_numpy(hnp.legs_matrix(N))
        assert A.shape == (N, N)
        # A_HiPPO = -H_N entry for entry.
        assert np.allclose(A, -closed_form_H(N))


def test_diagonal_is_minus_one_to_minus_N():
    N = 6
    A = to_numpy(hnp.legs_matrix(N))
    assert np.allclose(np.diag(A), -(np.arange(N) + 1.0))


def test_strictly_upper_triangle_is_zero():
    N = 5
    A = to_numpy(hnp.legs_matrix(N))
    assert np.allclose(np.triu(A, k=1), 0.0)


def test_explicit_N4_matrix_from_chapter():
    # H_4 exactly as printed in section 3.2.4.
    H4 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.sqrt(3), 2.0, 0.0, 0.0],
            [np.sqrt(5), np.sqrt(15), 3.0, 0.0],
            [np.sqrt(7), np.sqrt(21), np.sqrt(35), 4.0],
        ]
    )
    assert np.allclose(to_numpy(hnp.legs_hippo(4)), H4)
    # The stable state matrix is its negation.
    assert np.allclose(to_numpy(hnp.legs_matrix(4)), -H4)


def test_input_vector():
    N = 4
    # b_N = (sqrt(1), sqrt(3), sqrt(5), sqrt(7)).
    b = np.array([np.sqrt(1), np.sqrt(3), np.sqrt(5), np.sqrt(7)])
    assert np.allclose(to_numpy(hnp.legs_input(N)), b)


def test_eigenvalues_have_negative_real_part():
    # Stability claim of the chapter: A_HiPPO has eigenvalues with Re < 0.
    A = to_numpy(hnp.legs_matrix(8))
    assert np.all(np.real(np.linalg.eigvals(A)) < 0)


def test_numpy_torch_jax_agree():
    for N in (1, 3, 5, 9):
        A_np = to_numpy(hnp.legs_matrix(N))
        A_t = to_numpy(ht.legs_matrix(N))
        A_j = np.asarray(hj.legs_matrix(N))
        assert np.allclose(A_np, A_t)
        assert np.allclose(A_np, A_j)

        b_np = to_numpy(hnp.legs_input(N))
        b_t = to_numpy(ht.legs_input(N))
        b_j = np.asarray(hj.legs_input(N))
        assert np.allclose(b_np, b_t)
        assert np.allclose(b_np, b_j)
