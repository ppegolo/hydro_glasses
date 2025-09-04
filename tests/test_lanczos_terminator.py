import numpy as np
import pytest

from hydroglass.lanczos import (
    estimate_tail_parameters,
    haydock_greens_function,
    haydock_spectral_density,
    square_root_terminator,
)


def semicircle_density(omega: np.ndarray, a_inf: float, b_inf: float) -> np.ndarray:
    """Analytic DOS for infinite constant tridiagonal: alpha=a_inf, beta=b_inf."""
    x = omega - a_inf
    support = 4.0 * (b_inf**2) - (x**2)
    rho = np.zeros_like(omega, dtype=float)
    mask = support > 0.0
    rho[mask] = 0.5 / (np.pi * (b_inf**2)) * np.sqrt(support[mask])
    return rho


@pytest.mark.parametrize("a_inf,b_inf,M", [(0.1, 0.7, 60), (0.0, 1.0, 80)])
def test_terminator_matches_semicircle(a_inf: float, b_inf: float, M: int) -> None:
    """
    With constant tail and the square-root terminator, Haydock DOS matches semicircle.
    """
    # Build constant-coefficient tridiagonal of depth M
    alpha = np.full(M, a_inf, dtype=float)
    beta = np.full(M - 1, b_inf, dtype=float) if M > 1 else np.empty(0, dtype=float)

    # Frequency grid spanning support with pad
    width = 2.0 * b_inf
    omega = np.linspace(a_inf - 1.2 * width, a_inf + 1.2 * width, 4001)
    eta = 1e-3  # small imaginary part to stabilize CF; density taken as limit eta->0+

    # CF with square-root tail
    def tail(zz: np.ndarray) -> np.ndarray:
        return square_root_terminator(z=zz, a_inf=a_inf, b_inf=b_inf)

    G = haydock_greens_function(
        alpha=alpha, beta=beta, omega=omega, eta=eta, terminator=tail
    )
    rho_cf = (-1.0 / np.pi) * np.imag(G)

    # Analytic semicircle
    rho_exact = semicircle_density(omega=omega, a_inf=a_inf, b_inf=b_inf)

    # Compare in L2 and L_inf norms on support
    mask = rho_exact > 1e-12
    assert np.max(np.abs(rho_cf[mask] - rho_exact[mask])) < 2e-2
    rel_l2 = np.linalg.norm(rho_cf[mask] - rho_exact[mask]) / np.linalg.norm(
        rho_exact[mask]
    )
    assert rel_l2 < 1.5e-2


def test_terminator_with_estimated_tail_parameters() -> None:
    """Estimate a_inf, b_inf from last few levels and check stability."""
    rng = np.random.default_rng(0)
    # Make a mildly varying tail around a true constant tail (a_inf=0.05, b_inf=0.8).
    a_inf_true, b_inf_true = 0.05, 0.8
    M = 80
    alpha = a_inf_true + 0.03 * rng.standard_normal(M)
    beta = np.abs(b_inf_true + 0.03 * rng.standard_normal(M - 1))
    # Use last 10 levels to estimate
    a_inf_est, b_inf_est = estimate_tail_parameters(
        alpha=alpha, beta=beta, tail_window=10
    )

    omega = np.linspace(-2.5, 2.5, 3001)
    eta = 1e-3

    def tail(zz: np.ndarray) -> np.ndarray:
        return square_root_terminator(z=zz, a_inf=a_inf_est, b_inf=b_inf_est)

    rho_est = haydock_spectral_density(
        alpha=alpha, beta=beta, omega=omega, eta=eta, terminator=tail
    )
    # If the estimate is reasonable, the DOS stays normalized to 1 on a wide grid.
    # (The small eta and wide window should capture nearly all mass.)
    integ = float(np.trapezoid(rho_est, omega))
    assert np.isclose(integ, 1.0, rtol=3e-2, atol=3e-2)
