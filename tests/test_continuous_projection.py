r"""Checks for the early-chapter NumPy reference modules.

* :mod:`ssm_book.numpy_ref.state_memory` -- book section 2.1;
* :mod:`ssm_book.numpy_ref.continuous` -- book section 2.2;
* :mod:`ssm_book.numpy_ref.projection_memory` -- book section 3.1.
"""
import numpy as np

from ssm_book.numpy_ref.state_memory import scalar_memory
from ssm_book.numpy_ref.continuous import impulse_response, simulate
from ssm_book.numpy_ref.projection_memory import (
    legendre_project,
    legendre_reconstruct,
)


def test_scalar_memory_constant_input_steady_state():
    # x_{k+1} = a x_k + b u_k with u_k == 1 and |a| < 1 converges to b/(1-a).
    a, b, c = 0.9, 1.0, 1.0
    L = 400
    y = scalar_memory(np.ones(L), a=a, b=b, c=c)
    steady = c * b / (1.0 - a)
    assert np.isclose(y[-1].real, steady, rtol=1e-6)
    assert np.allclose(y.imag, 0.0)


def test_scalar_memory_matches_closed_form_partial_sum():
    # y_k = c * sum_{j<=k} a^(k-j) b u_j; for u == 1 this is the geometric sum.
    a, b, c = 0.5, 2.0, 1.5
    L = 20
    y = scalar_memory(np.ones(L), a=a, b=b, c=c)
    k = np.arange(L)
    expected = c * b * (1.0 - a ** (k + 1)) / (1.0 - a)
    assert np.allclose(y, expected)


def test_impulse_response_scalar_is_exponential():
    A, B, C = [[-1.0]], [[1.0]], [[1.0]]
    ts = np.linspace(0.0, 5.0, 60)
    h = impulse_response(A, B, C, ts).reshape(-1)
    assert np.allclose(h, np.exp(-ts))


def test_simulate_matches_zoh_recurrence():
    # simulate should reproduce a hand-rolled ZOH recurrence for a stable system.
    from ssm_book.numpy_ref.discretisation import zoh_discretise

    A = np.array([[-1.0, 0.5], [0.0, -2.0]])
    B = np.array([[1.0], [1.0]])
    C = np.array([[1.0, -1.0]])
    dt = 0.1
    rng = np.random.default_rng(0)
    u = rng.standard_normal(50)
    y = simulate(A, B, C, u, dt).reshape(-1)

    Abar, Bbar = zoh_discretise(A, B, dt)
    x = np.zeros((2, 1), dtype=complex)
    expected = []
    for uk in u:
        x = Abar @ x + Bbar * uk
        expected.append((C @ x).item())
    assert np.allclose(y, np.asarray(expected))


def test_legendre_reconstruction_improves_with_N():
    # A smooth test signal on [0, 1]; reconstruction error falls as N grows.
    M = 200
    t = np.linspace(0.0, 1.0, M)
    signal = np.sin(3.0 * t) + 0.5 * t ** 2

    def reconstruction_error(N):
        coeffs = legendre_project(signal, N)
        approx = legendre_reconstruct(coeffs, M)
        return np.linalg.norm(approx.real - signal) / np.sqrt(M)

    err3 = reconstruction_error(3)
    err8 = reconstruction_error(8)
    assert err8 < err3
    assert err8 < 1e-4  # 8 coefficients already fit this smooth signal tightly


def test_legendre_roundtrip_recovers_low_degree_signal():
    # A degree-2 polynomial in t is captured exactly by 3 Legendre coefficients.
    M = 50
    t = np.linspace(0.0, 1.0, M)
    signal = 1.0 - 2.0 * t + 3.0 * t ** 2
    coeffs = legendre_project(signal, 3)
    approx = legendre_reconstruct(coeffs, M)
    assert np.allclose(approx.real, signal)
