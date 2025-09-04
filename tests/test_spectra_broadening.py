# tests/test_spectra_broadening.py
import numpy as np
import pytest

from hydroglass.spectra.broadening import gaussian_kernel, lorentzian_kernel

RTOL = 2e-3
ATOL = 2e-4


def _trapz_integrate_columns(
    kernel_matrix: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    """Integrate each column over omega using trapezoidal rule."""
    domega = np.diff(omega)
    # Average adjacent rows to approximate integral per column
    avg_rows = 0.5 * (kernel_matrix[:-1, :] + kernel_matrix[1:, :])
    return (avg_rows.T @ domega).T  # shape: (ncols,)


def test_gaussian_kernel_column_normalization_on_wide_window():
    # Wide window: ±8 sigma covers essentially all Gaussian mass.
    sigma = 0.2
    centers = np.array([-0.5, 0.0, 0.75], dtype=float)
    omega = np.linspace(-2.0, 2.0, 4001)
    K = gaussian_kernel(omega=omega, centers=centers, sigma=sigma)

    assert K.shape == (omega.size, centers.size)
    integrals = _trapz_integrate_columns(K, omega)  # ~1 for each column
    assert np.allclose(integrals, np.ones_like(integrals), rtol=RTOL, atol=ATOL)


def test_lorentzian_kernel_column_normalization_on_wide_window():
    # Finite window: choose wide enough range for good normalization.
    gamma = 0.15
    centers = np.array([-0.3, 0.0, 0.5], dtype=float)
    omega = np.linspace(-3.0, 3.0, 6001)
    K = lorentzian_kernel(omega=omega, centers=centers, gamma=gamma)

    assert K.shape == (omega.size, centers.size)
    integrals = _trapz_integrate_columns(K, omega)
    # Lorentzian has heavy tails, so allow slightly looser tolerance.
    assert np.allclose(integrals, np.ones_like(integrals), rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.25])
def test_gaussian_kernel_symmetry_and_peak_height(sigma: float):
    center = np.array([0.3], dtype=float)
    omega = np.linspace(-1.0, 1.0, 2001)
    K = gaussian_kernel(omega=omega, centers=center, sigma=sigma)
    # Symmetry around center in a discrete sense
    idx_center = np.argmin(np.abs(omega - center[0]))
    left = K[:idx_center, 0]
    right = K[idx_center + 1 :, 0][::-1]
    # Compare mirrored halves within a tolerance
    n = min(left.size, right.size)
    assert np.allclose(left[-n:], right[-n:], rtol=1e-2, atol=1e-3)
    # Peak near the center is maximal
    assert K[idx_center, 0] == K[:, 0].max()


@pytest.mark.parametrize("gamma", [0.05, 0.1, 0.3])
def test_lorentzian_kernel_symmetry_and_peak_height(gamma: float):
    center = np.array([-0.2], dtype=float)
    omega = np.linspace(-1.0, 1.0, 2001)
    K = lorentzian_kernel(omega=omega, centers=center, gamma=gamma)
    idx_center = np.argmin(np.abs(omega - center[0]))
    left = K[:idx_center, 0]
    right = K[idx_center + 1 :, 0][::-1]
    n = min(left.size, right.size)
    assert np.allclose(left[-n:], right[-n:], rtol=1e-2, atol=1e-3)
    assert K[idx_center, 0] == K[:, 0].max()
