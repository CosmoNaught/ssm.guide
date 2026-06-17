r"""The diagonal state space model of book section 3.5.

Checks that the diagonal recurrence equals the convolution with its Vandermonde
kernel, that a complex-conjugate pair of modes gives a real kernel, that the
S4D-Lin / S4D-Inv initialisations match their closed forms, and that the three
backends agree on ``diagonal_kernel``.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import diagonal as dnp
from ssm_book.numpy_ref.kernels import causal_conv
from ssm_book.torch_ref import diagonal as dt
from ssm_book.jax_ref import diagonal as dj


def random_diagonal_system(N=4, L=16, rho=0.7, seed=0):
    """A stable diagonal system (``|lambda_n| < 1``) with random weights and input."""
    rng = np.random.default_rng(seed)
    mag = rho * rng.uniform(0.1, 1.0, N)
    ang = rng.uniform(-np.pi, np.pi, N)
    lambdas = mag * np.exp(1j * ang)
    Bbar = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    u = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return lambdas, Bbar, C, u, L


def test_recurrence_equals_diagonal_convolution():
    lambdas, Bbar, C, u, L = random_diagonal_system()
    weights = C * Bbar  # w_n = C_n * Bbar_n
    K = dnp.diagonal_kernel(lambdas, weights, L)
    y_rec = dnp.diagonal_recurrence(lambdas, Bbar, C, u)
    assert np.allclose(y_rec, causal_conv(K, u))


def test_conjugate_pair_gives_real_kernel():
    # A mode and its conjugate, with conjugate weights, contribute a real kernel.
    lam = 0.6 * np.exp(1j * 0.7)
    w = 0.4 + 0.9j
    lambdas = np.array([lam, np.conj(lam)])
    weights = np.array([w, np.conj(w)])
    K = dnp.diagonal_kernel(lambdas, weights, 32)
    assert np.allclose(K.imag, 0.0)


def test_s4d_lin_init_closed_form():
    M = 8
    n = np.arange(M)
    expected = -0.5 + 1j * np.pi * n
    assert np.allclose(dnp.s4d_lin_init(M), expected)


def test_s4d_inv_init_closed_form():
    M = 8
    n = np.arange(M)
    expected = -0.5 + 1j * (M / np.pi) * (M / (2 * n + 1) - 1)
    assert np.allclose(dnp.s4d_inv_init(M), expected)


def test_backends_agree_on_diagonal_kernel():
    lambdas, Bbar, C, u, L = random_diagonal_system(seed=3)
    weights = C * Bbar
    K_np = dnp.diagonal_kernel(lambdas, weights, L)
    K_t = to_numpy(dt.diagonal_kernel(lambdas, weights, L))
    K_j = np.asarray(dj.diagonal_kernel(lambdas, weights, L))
    assert np.allclose(K_np, K_t)
    assert np.allclose(K_np, K_j)


def test_backends_agree_on_initialisations():
    M = 8
    assert np.allclose(dnp.s4d_lin_init(M), to_numpy(dt.s4d_lin_init(M)))
    assert np.allclose(dnp.s4d_lin_init(M), np.asarray(dj.s4d_lin_init(M)))
    assert np.allclose(dnp.s4d_inv_init(M), to_numpy(dt.s4d_inv_init(M)))
    assert np.allclose(dnp.s4d_inv_init(M), np.asarray(dj.s4d_inv_init(M)))
