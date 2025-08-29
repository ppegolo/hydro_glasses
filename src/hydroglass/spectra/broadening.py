# src/hydroglass/spectra/broadening.py
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

_TWO_PI_SQRT: Final[float] = float(np.sqrt(2.0 * np.pi))
_PI: Final[float] = float(np.pi)


def gaussian_kernel(
    omega: NDArray[np.floating],
    centers: NDArray[np.floating],
    sigma: float,
) -> NDArray[np.floating]:
    """Return Gaussian line-shape matrix evaluated on a frequency grid.

    The returned matrix has shape (omega.size, centers.size) with entries
    K[i, j] = N(omega[i] | centers[j], sigma). It is normalized in the
    continuous sense, and on a sufficiently wide grid the trapezoidal
    integral over omega of each column is approximately one.

    Args:
        omega: One dimensional frequency grid with shape (n,).
        centers: One dimensional array of peak centers with shape (m,).
        sigma: Positive standard deviation of the Gaussian.

    Returns:
        Matrix of shape (n, m) with nonnegative entries.

    Raises:
        ValueError: If sigma is not positive or inputs are not one dimensional.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if omega.ndim != 1:
        raise ValueError("omega must be one dimensional.")
    if centers.ndim != 1:
        raise ValueError("centers must be one dimensional.")

    # Broadcasted squared distances; shape (n, m).
    diff = omega[:, None] - centers[None, :]
    inv = 1.0 / (sigma * _TWO_PI_SQRT)
    return inv * np.exp(-0.5 * (diff / sigma) ** 2)


def lorentzian_kernel(
    omega: NDArray[np.floating],
    centers: NDArray[np.floating],
    gamma: float,
) -> NDArray[np.floating]:
    """Return Lorentzian line-shape matrix, renormalized on the given grid.

    The returned matrix has shape (omega.size, centers.size) with entries
    K[i, j] = L(omega[i] | centers[j], gamma). Because the Lorentzian has
    heavy tails, a finite omega window misses some mass. To ensure that
    each column integrates to one on the provided grid, the columns are
    renormalized using the trapezoidal rule.

    Args:
        omega: One dimensional frequency grid with shape (n,).
        centers: One dimensional array of peak centers with shape (m,).
        gamma: Positive half-width at half-maximum parameter.

    Returns:
        Matrix of shape (n, m) with nonnegative entries whose columns
        integrate to one (within numerical tolerance) on the given grid.

    Raises:
        ValueError: If gamma is not positive or inputs are not one dimensional.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    if omega.ndim != 1:
        raise ValueError("omega must be one dimensional.")
    if centers.ndim != 1:
        raise ValueError("centers must be one dimensional.")

    # Unnormalized Lorentzian; shape (n, m).
    diff = omega[:, None] - centers[None, :]
    K = (gamma / _PI) / (diff * diff + gamma * gamma)

    # Column-wise renormalization on the finite grid using trapezoidal rule.
    # This compensates for the heavy tails truncated by the finite window.
    integrals = np.trapezoid(K, x=omega, axis=0)  # shape (m,)
    # Avoid division by zero by only normalizing positive integrals.
    mask = integrals > 0.0
    K[:, mask] = K[:, mask] / integrals[mask]

    return K
