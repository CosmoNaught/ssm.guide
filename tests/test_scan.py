r"""Affine composition and parallel scan of book section 3.5.9.

The associative scan over affine steps :math:`(a_k,c_k)` must match the
left-to-right recurrence :math:`x_{k+1}=a_k x_k+c_k` in every backend, the
backends must agree, and the time-invariant case :math:`a_k=\bar\lambda`
reduces to the diagonal recurrence.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import scan as snp
from ssm_book.torch_ref import scan as st
from ssm_book.jax_ref import scan as sj


def random_steps(L=37, rho=0.9, seed=0):
    """Random per-step pairs ``(a, c)`` with ``|a_k| < rho < 1``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    a = rho * a / np.maximum(np.abs(a), 1.0)  # ensure |a_k| < rho
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return a, c


def constant_steps(L=37, lam=0.8 - 0.3j, seed=1):
    """Time-invariant multiplier ``a_k = lam`` (the diagonal recurrence)."""
    rng = np.random.default_rng(seed)
    a = np.full(L, lam, dtype=complex)
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return a, c


# (a) associative_scan == sequential_recurrence, time-varying and time-invariant.


def test_numpy_scan_equals_recurrence_time_varying():
    a, c = random_steps()
    assert np.allclose(snp.associative_scan(a, c), snp.sequential_recurrence(a, c))


def test_numpy_scan_equals_recurrence_time_invariant():
    a, c = constant_steps()
    x_scan = snp.associative_scan(a, c)
    x_rec = snp.sequential_recurrence(a, c)
    assert np.allclose(x_scan, x_rec)
    # With a constant multiplier the recurrence is the diagonal convolution
    # x_k = sum_{j<=k} lam^{k-j} c_j; check that closed form too.
    lam = a[0]
    L = len(a)
    direct = np.array([sum(lam ** (k - j) * c[j] for j in range(k + 1)) for k in range(L)])
    assert np.allclose(x_scan, direct)


def test_torch_scan_equals_recurrence():
    a, c = random_steps(seed=2)
    assert np.allclose(to_numpy(st.associative_scan(a, c)), to_numpy(st.sequential_recurrence(a, c)))
    a, c = constant_steps(seed=3)
    assert np.allclose(to_numpy(st.associative_scan(a, c)), to_numpy(st.sequential_recurrence(a, c)))


def test_jax_scan_equals_recurrence():
    a, c = random_steps(seed=4)
    assert np.allclose(np.asarray(sj.associative_scan(a, c)), np.asarray(sj.sequential_recurrence(a, c)))
    a, c = constant_steps(seed=5)
    assert np.allclose(np.asarray(sj.associative_scan(a, c)), np.asarray(sj.sequential_recurrence(a, c)))


# (b) numpy == torch == jax.


def test_backends_agree():
    for seed in (6, 7):
        a, c = random_steps(seed=seed)
        x_np = snp.associative_scan(a, c)
        x_t = to_numpy(st.associative_scan(a, c))
        x_j = np.asarray(sj.associative_scan(a, c))
        assert np.allclose(x_np, x_t)
        assert np.allclose(x_np, x_j)

    a, c = constant_steps(seed=8)
    x_np = snp.associative_scan(a, c)
    x_t = to_numpy(st.associative_scan(a, c))
    x_j = np.asarray(sj.associative_scan(a, c))
    assert np.allclose(x_np, x_t)
    assert np.allclose(x_np, x_j)


def test_affine_compose_matches_two_step_recurrence():
    # (a,c) bullet (a',c') applied to x_0 = 0 gives x_2 of the two-step loop.
    a, c = random_steps(L=2, seed=9)
    a_comb, c_comb = snp.affine_compose((a[1], c[1]), (a[0], c[0]))
    x = snp.sequential_recurrence(a, c)
    assert np.allclose(c_comb, x[1])  # offset is x_2 (state after both steps)
    assert np.allclose(a_comb, a[0] * a[1])


def test_empty_input():
    a = np.zeros(0, dtype=complex)
    c = np.zeros(0, dtype=complex)
    assert snp.associative_scan(a, c).shape == (0,)
    assert snp.sequential_recurrence(a, c).shape == (0,)
