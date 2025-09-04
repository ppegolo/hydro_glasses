"""
Lanczos tridiagonalization and Haydock continued fraction utilities.

This module provides:
  - lanczos_tridiagonal: symmetric Lanczos with optional reorthogonalization
    that returns the tridiagonal coefficients and Krylov basis.
  - tridiagonal_eigh: eigenvalues of the tridiagonal matrix from (alpha, beta).
  - haydock_greens_function: continued-fraction evaluation of G(omega + i eta).
  - haydock_spectral_density: spectral density rho(omega) = -1/pi Im G.

All routines use dense numpy and optionally accept scipy.sparse matrices.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.sparse as sp

    SCIPY_OK = True
except Exception:  # pragma: no cover - optional path
    SCIPY_OK = False

logger = logging.getLogger(__name__)


def _matvec(matrix_or_linear_operator, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply a matrix or LinearOperator-like to a vector."""
    if SCIPY_OK and sp.issparse(matrix_or_linear_operator):
        return matrix_or_linear_operator @ x
    # numpy.ndarray or any object supporting @
    return matrix_or_linear_operator @ x


def lanczos_tridiagonal(
    matrix_or_linear_operator: object,
    initial_vector: NDArray[np.floating],
    num_steps: int,
    reorthogonalization: Literal["none", "partial", "full"] = "partial",
    atol_reorth: float = 1e-12,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Perform symmetric Lanczos and return tridiagonal coefficients and basis.

    The implementation assumes a real symmetric (or Hermitian with real data) operator.

    Args:
        matrix_or_linear_operator: Dense ndarray, scipy.sparse matrix, or any
            object that supports the matmul operator with a vector.
        initial_vector: Starting vector. It will be normalized internally.
        num_steps: Number of Lanczos steps m, with m >= 1.
        reorthogonalization: Strategy for numerical stability:
            - "none": no explicit reorthogonalization.
            - "partial": one pass against existing basis if loss detected.
            - "full": always orthogonalize against the entire basis built so far.
        atol_reorth: Threshold to trigger partial reorthogonalization.

    Returns:
        Tuple (alpha, beta, V) where:
            alpha: Diagonal entries of the tridiagonal matrix, shape (m,).
            beta:  Subdiagonal entries (nonnegative), shape (m - 1,).
            V:     Orthonormal Lanczos basis, shape (n, m).

    Raises:
        ValueError: If shapes are inconsistent or num_steps < 1.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    v = np.asarray(initial_vector, dtype=float).copy()
    n = v.size
    v_norm = float(np.linalg.norm(v))
    if not np.isfinite(v_norm) or v_norm == 0.0:
        raise ValueError("initial_vector must be finite and nonzero.")
    v /= v_norm

    V = np.zeros((n, num_steps), dtype=float)
    alpha = np.zeros((num_steps,), dtype=float)
    beta = np.zeros((num_steps - 1,), dtype=float)

    w_prev = np.zeros_like(v)
    for j in range(num_steps):
        V[:, j] = v
        w = _matvec(matrix_or_linear_operator, v)
        alpha_j = float(np.dot(v, w))
        alpha[j] = alpha_j

        # Three-term recurrence: w <- w - alpha_j v - beta_{j-1} v_{j-1}
        w = w - alpha_j * v
        if j > 0:
            w = w - beta[j - 1] * w_prev

        # Optional reorthogonalization against current basis
        if reorthogonalization == "full":
            # Gram-Schmidt against all V[:, :j+1]
            h = V[:, : j + 1].T @ w
            w = w - V[:, : j + 1] @ h
        elif reorthogonalization == "partial":
            loss = float(np.linalg.norm(V[:, : j + 1].T @ w, ord=2)) if j >= 0 else 0.0
            if loss > atol_reorth:
                h = V[:, : j + 1].T @ w
                w = w - V[:, : j + 1] @ h

        beta_j = float(np.linalg.norm(w))
        if j < num_steps - 1:
            beta[j] = beta_j

        # Prepare next vectors
        if j < num_steps - 1:
            if beta_j == 0.0:
                # Happy breakdown; pad remaining steps with zeros and reuse last basis
                # vector.
                logger.debug("Lanczos happy breakdown at step %d.", j + 1)
                for k in range(j + 1, num_steps):
                    V[:, k] = V[:, j]
                    alpha[k] = alpha[j]
                    if k < num_steps - 1:
                        beta[k] = 0.0
                break
            w_prev = v
            v = w / beta_j

    # Enforce nonnegative betas by sign flipping the next basis vector if needed
    # (here beta_j is a norm, so already nonnegative).
    return alpha, beta, V


def tridiagonal_eigh(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Eigenvalues of the symmetric tridiagonal defined by (alpha, beta).

    Args:
        alpha: Diagonal entries, shape (m,).
        beta: Subdiagonal entries, shape (m-1,).

    Returns:
        Eigenvalues in ascending order, shape (m,).
    """
    m = alpha.size
    if beta.size not in (0, m - 1):
        raise ValueError("beta must have shape (m-1,) or be empty when m=1.")
    # Build dense tridiagonal (small m in practice for Ritz estimates).
    T = np.zeros((m, m), dtype=float)
    np.fill_diagonal(T, alpha)
    if m > 1:
        off = beta.copy()
        T[np.arange(m - 1), np.arange(1, m)] = off
        T[np.arange(1, m), np.arange(m - 1)] = off
    evals = np.linalg.eigvalsh(T)
    return evals


def haydock_greens_function(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    omega: NDArray[np.floating],
    eta: float,
) -> NDArray[np.complexfloating]:
    """Evaluate the Haydock continued fraction Green's function.

    Computes G(z) = <v0 | (z I - H)^{-1} | v0> where z = omega + i eta.

    Args:
        alpha: Tridiagonal diagonal entries from Lanczos, shape (m,).
        beta: Tridiagonal subdiagonal entries, shape (m-1,).
        omega: Real frequency grid, shape (nomega,).
        eta: Positive imaginary shift for causal Green's function.

    Returns:
        Complex array G(omega + i eta) with shape (nomega,).

    Raises:
        ValueError: If eta is not positive or shapes are inconsistent.
    """
    if eta <= 0.0 or not np.isfinite(eta):
        raise ValueError("eta must be positive and finite.")
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    w = np.asarray(omega, dtype=float)
    m = a.size
    if b.size not in (0, m - 1):
        raise ValueError("beta must have shape (m-1,) or be empty when m=1.")
    z = w.astype(np.complex128) + 1j * float(eta)
    # Backward continued fraction evaluation
    g = np.zeros_like(z, dtype=np.complex128)
    for j in range(m - 1, -1, -1):
        if j == m - 1:
            g = 1.0 / (z - a[j])
        else:
            g = 1.0 / (z - a[j] - (b[j] ** 2) * g)
    return g


def haydock_spectral_density(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    omega: NDArray[np.floating],
    eta: float,
) -> NDArray[np.floating]:
    """Spectral density rho(omega) associated with the starting vector.

    rho(omega) = -1/pi * Im G(omega + i eta)

    Args:
        alpha: Tridiagonal diagonal entries from Lanczos, shape (m,).
        beta: Tridiagonal subdiagonal entries, shape (m-1,).
        omega: Real frequency grid, shape (nomega,).
        eta: Positive imaginary shift.

    Returns:
        rho(omega) with shape (nomega,). The integral over omega is approximately 1
        if the starting vector used in the Lanczos run was normalized.
    """
    G = haydock_greens_function(alpha=alpha, beta=beta, omega=omega, eta=eta)
    rho = -np.imag(G) / np.pi
    rho = np.clip(rho, a_min=0.0, a_max=None)
    return rho.astype(float, copy=False)
