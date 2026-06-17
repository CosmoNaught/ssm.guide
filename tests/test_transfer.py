r"""Generating function vs kernel (book section 2.5), across backends.

    G(z) = sum_m K_m z^m              (power series, inside convergence)
    K    = ifft(G on roots of unity)  (spectral recovery, up to aliasing)
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import kernels as knp
from ssm_book.numpy_ref import transfer as tnp
from ssm_book.torch_ref import transfer as tt
from ssm_book.jax_ref import transfer as tj


def random_stable_system(N=4, rho=0.6, seed=0):
    """A discrete system with spectral radius ``rho`` < 1."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A = rho * A / max(np.abs(np.linalg.eigvals(A)))
    Bbar = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return A, Bbar, C


def test_generating_function_equals_truncated_power_series():
    # Stable system; evaluate at z0 well inside the disc of convergence.
    A, Bbar, C = random_stable_system(rho=0.6, seed=10)
    rho = max(np.abs(np.linalg.eigvals(A)))
    z0 = 0.5 / rho  # |z0| < 1 / spectral_radius
    M = 200
    K = knp.ssm_kernel(A, Bbar, C, M + 1)
    series = sum(K[m] * z0 ** m for m in range(M + 1))
    G = tnp.generating_function(z0, A, Bbar, C)
    assert np.allclose(G, series)


def test_kernel_recovered_from_roots_of_unity():
    # Strongly stable: aliasing K_m + K_{m+L} + ... is negligible by L=256.
    A, Bbar, C = random_stable_system(rho=0.4, seed=20)
    L = 256
    samples = tnp.evaluate_on_roots_of_unity(A, Bbar, C, L)
    K_recovered = tnp.recover_kernel_by_ifft(samples)
    K_recurrence = knp.ssm_kernel(A, Bbar, C, L)
    assert np.allclose(K_recovered, K_recurrence, atol=1e-6)


def test_numpy_torch_jax_agree():
    A, Bbar, C = random_stable_system(N=3, rho=0.5, seed=30)
    z0 = 0.7

    R_np = tnp.resolvent(z0, A)
    R_t = to_numpy(tt.resolvent(z0, A))
    R_j = np.asarray(tj.resolvent(z0, A))
    assert np.allclose(R_np, R_t)
    assert np.allclose(R_np, R_j)

    G_np = tnp.generating_function(z0, A, Bbar, C)
    G_t = to_numpy(tt.generating_function(z0, A, Bbar, C))
    G_j = np.asarray(tj.generating_function(z0, A, Bbar, C))
    assert np.allclose(G_np, G_t)
    assert np.allclose(G_np, G_j)
