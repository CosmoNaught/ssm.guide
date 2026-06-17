r"""The headline equivalence of book section 2.4: the recurrence and the
convolution compute the same input-output map, in every backend.

    ssm_recurrence == causal_conv(ssm_kernel) == fft_conv == toeplitz @ u

and the three backends agree with one another.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import kernels as knp
from ssm_book.torch_ref import kernels as kt
from ssm_book.jax_ref import kernels as kj


def random_stable_system(N=4, L=16, rho=0.6, seed=0):
    """A discrete system with spectral radius ``rho`` < 1, and a random input."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A = rho * A / max(np.abs(np.linalg.eigvals(A)))
    Bbar = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    u = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return A, Bbar, C, u, L


def test_numpy_recurrence_equals_all_convolution_forms():
    A, Bbar, C, u, L = random_stable_system()
    y_rec = knp.ssm_recurrence(A, Bbar, C, u)
    K = knp.ssm_kernel(A, Bbar, C, L)
    assert np.allclose(y_rec, knp.causal_conv(K, u))
    assert np.allclose(y_rec, knp.fft_conv(K, u))
    assert np.allclose(y_rec, knp.toeplitz(K, L) @ np.asarray(u, dtype=complex))


def test_torch_matches_numpy():
    A, Bbar, C, u, L = random_stable_system(seed=1)
    y_np = knp.ssm_recurrence(A, Bbar, C, u)
    y_t = to_numpy(kt.ssm_recurrence(A, Bbar, C, u))
    K_t = kt.ssm_kernel(A, Bbar, C, L)
    assert np.allclose(y_np, y_t)
    assert np.allclose(y_t, to_numpy(kt.causal_conv(K_t, u)))
    assert np.allclose(y_t, to_numpy(kt.fft_conv(K_t, u)))


def test_jax_matches_numpy():
    A, Bbar, C, u, L = random_stable_system(seed=2)
    y_np = knp.ssm_recurrence(A, Bbar, C, u)
    y_j = np.asarray(kj.ssm_recurrence(A, Bbar, C, u))
    K_j = kj.ssm_kernel(A, Bbar, C, L)
    assert np.allclose(y_np, y_j)
    assert np.allclose(y_j, np.asarray(kj.causal_conv(K_j, u)))
    assert np.allclose(y_j, np.asarray(kj.fft_conv(K_j, u)))


def test_kernel_zeroth_coefficient_is_C_B():
    A, Bbar, C, u, L = random_stable_system(seed=3)
    K = knp.ssm_kernel(A, Bbar, C, L)
    assert np.allclose(K[0], np.asarray(C, complex) @ np.asarray(Bbar, complex))
