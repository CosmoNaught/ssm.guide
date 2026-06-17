r"""Discretisation checks for book section 2.3.

* the zero-order hold reproduces the closed-form scalar formulas;
* a stable continuous system maps to a stable discrete system (|mu| < 1);
* the bilinear transform matches its closed form;
* NumPy, PyTorch, and JAX agree.
"""
import numpy as np

from ssm_book.common import to_numpy
from ssm_book.numpy_ref import discretisation as dnp
from ssm_book.torch_ref import discretisation as dt
from ssm_book.jax_ref import discretisation as dj


def test_zoh_scalar_closed_form():
    a, b, dt_ = -0.7, 1.3, 0.5
    Abar, Bbar = dnp.zoh_discretise([[a]], [[b]], dt_)
    assert np.allclose(Abar[0, 0], np.exp(a * dt_))
    assert np.allclose(Bbar[0, 0], (np.exp(a * dt_) - 1.0) / a * b)


def test_zoh_preserves_stability():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, 5))
    A = M - M.T - 3.0 * np.eye(5)  # skew-symmetric minus 3I => Re(lambda) = -3
    assert np.all(np.real(np.linalg.eigvals(A)) < 0)
    Abar, _ = dnp.zoh_discretise(A, np.ones((5, 1)), 0.3)
    assert np.all(np.abs(np.linalg.eigvals(Abar)) < 1.0)


def test_bilinear_closed_form():
    a, b, dt_ = -0.9, 2.0, 0.4
    Abar, Bbar = dnp.bilinear_discretise([[a]], [[b]], dt_)
    mu = (1 + dt_ / 2 * a) / (1 - dt_ / 2 * a)
    assert np.allclose(Abar[0, 0], mu)
    assert np.allclose(Bbar[0, 0], dt_ / (1 - dt_ / 2 * a) * b)


def test_zoh_eigenvalue_map():
    # mu = exp(lambda * dt) for ZOH.
    A = np.diag([-0.5, -1.0, -2.0]).astype(float)
    dt_ = 0.25
    Abar, _ = dnp.zoh_discretise(A, np.ones((3, 1)), dt_)
    assert np.allclose(np.sort(np.linalg.eigvals(Abar).real),
                       np.sort(np.exp(np.array([-0.5, -1.0, -2.0]) * dt_)))


def test_backends_agree():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4)) - 2.0 * np.eye(4)
    B = rng.standard_normal((4, 1))
    dt_ = 0.3
    for name, mod in (("torch", dt), ("jax", dj)):
        Az, Bz = dnp.zoh_discretise(A, B, dt_)
        Az2, Bz2 = mod.zoh_discretise(A, B, dt_)
        assert np.allclose(Az, to_numpy(Az2)), name
        assert np.allclose(Bz, to_numpy(Bz2)), name
        Ab, Bb = dnp.bilinear_discretise(A, B, dt_)
        Ab2, Bb2 = mod.bilinear_discretise(A, B, dt_)
        assert np.allclose(Ab, to_numpy(Ab2)), name
        assert np.allclose(Bb, to_numpy(Bb2)), name
