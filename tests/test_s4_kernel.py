r"""Book section 3.4: the structured S4 kernel must equal the dense kernel
:math:`\bar K_m=C\bar A^m\bar B` for a DPLR system, across all three backends.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import s4_reference as s4np
from ssm_book.torch_ref import s4 as s4t
from ssm_book.jax_ref import s4 as s4j


def random_dplr_system(N=8, r=1, L=16, seed=0):
    """Random DPLR S4 system; diagonal modes have negative real part (stable) and B, C are real."""
    rng = np.random.default_rng(seed)
    Lambda = -rng.uniform(0.3, 1.5, N) + 1j * rng.standard_normal(N)
    P = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
    Q = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
    B = rng.standard_normal(N)  # real B
    C = rng.standard_normal(N)  # real C
    dt = 0.1
    return Lambda, P, Q, B, C, dt, L


def test_structured_equals_dense_numpy():
    """Structured kernel == dense kernel (NumPy)."""
    Lambda, P, Q, B, C, dt, L = random_dplr_system()
    K_dense = s4np.dense_kernel(Lambda, P, Q, B, C, dt, L)
    K_struct = s4np.structured_kernel(Lambda, P, Q, B, C, dt, L)
    assert np.allclose(K_struct, K_dense, atol=1e-8)


def test_structured_equals_dense_higher_rank_and_odd_length():
    """Same equivalence for rank-2 and an odd length (no z = -1 node)."""
    Lambda, P, Q, B, C, dt, L = random_dplr_system(N=8, r=2, L=15, seed=4)
    K_dense = s4np.dense_kernel(Lambda, P, Q, B, C, dt, L)
    K_struct = s4np.structured_kernel(Lambda, P, Q, B, C, dt, L)
    assert np.allclose(K_struct, K_dense, atol=1e-8)


def test_torch_matches_numpy():
    """Torch dense and structured kernels match the NumPy reference."""
    Lambda, P, Q, B, C, dt, L = random_dplr_system(seed=1)
    K_dense_np = s4np.dense_kernel(Lambda, P, Q, B, C, dt, L)
    K_struct_np = s4np.structured_kernel(Lambda, P, Q, B, C, dt, L)
    K_dense_t = to_numpy(s4t.dense_kernel(Lambda, P, Q, B, C, dt, L))
    K_struct_t = to_numpy(s4t.structured_kernel(Lambda, P, Q, B, C, dt, L))
    assert np.allclose(K_dense_t, K_dense_np, atol=1e-8)
    assert np.allclose(K_struct_t, K_struct_np, atol=1e-8)
    # torch pipeline is self-consistent
    assert np.allclose(K_struct_t, K_dense_t, atol=1e-8)


def test_jax_matches_numpy():
    """JAX dense and structured kernels match the NumPy reference."""
    Lambda, P, Q, B, C, dt, L = random_dplr_system(seed=2)
    K_dense_np = s4np.dense_kernel(Lambda, P, Q, B, C, dt, L)
    K_struct_np = s4np.structured_kernel(Lambda, P, Q, B, C, dt, L)
    K_dense_j = np.asarray(s4j.dense_kernel(Lambda, P, Q, B, C, dt, L))
    K_struct_j = np.asarray(s4j.structured_kernel(Lambda, P, Q, B, C, dt, L))
    assert np.allclose(K_dense_j, K_dense_np, atol=1e-8)
    assert np.allclose(K_struct_j, K_struct_np, atol=1e-8)
    assert np.allclose(K_struct_j, K_dense_j, atol=1e-8)
