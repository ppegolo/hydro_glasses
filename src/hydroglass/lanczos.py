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
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    import scipy as sp  # type: ignore
    from scipy.sparse.linalg import LinearOperator  # type: ignore
except Exception:  # pragma: no cover - SciPy missing
    sp = None  # type: ignore
    LinearOperator = None  # type: ignore


logger = logging.getLogger(__name__)


def _matvec(matrix_or_linear_operator: object, x: np.ndarray) -> np.ndarray:
    """
    Apply the operator `matrix_or_linear_operator` to vector `x`.

    Supported kinds (checked in this order):
      1) Callable: f(x) -> y
      2) SciPy LinearOperator (or duck-typed: has .matvec)
      3) SciPy sparse matrix (if SciPy is present)
      4) Dense ndarray / any object supporting '@'
    """
    # 1) Plain callable: f(x) -> y
    if callable(matrix_or_linear_operator):
        y = matrix_or_linear_operator(x)
        return np.asarray(y, dtype=float)

    # 2) LinearOperator-like: has .matvec
    if hasattr(matrix_or_linear_operator, "matvec"):
        y = matrix_or_linear_operator.matvec(x)  # type: ignore[attr-defined]
        return np.asarray(y, dtype=float)

    # 3) Sparse matrix (guarded so it never breaks when SciPy isn't available)
    try:
        import scipy.sparse as _sp  # local import to avoid hard dependency

        if hasattr(_sp, "issparse") and _sp.issparse(matrix_or_linear_operator):
            return np.asarray(matrix_or_linear_operator @ x, dtype=float)
    except Exception:
        pass  # SciPy not available or issparse missing -> skip

    # 4) Dense array-like (or anything that implements @)
    return np.asarray(matrix_or_linear_operator @ x, dtype=float)


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


def square_root_terminator(
    z: NDArray[np.complexfloating] | complex,
    a_inf: float,
    b_inf: float,
) -> NDArray[np.complexfloating] | complex:
    """Square-root terminator for Haydock tails with robust branch selection.

    Solves the constant-tail quadratic
        G = 1 / (z - a_inf - b_inf^2 * G)
    giving
        G(z) = ( (z - a_inf) - s(z) ) / (2 * b_inf^2),
    where s(z) = sqrt( (z - a_inf)^2 - 4 b_inf^2 ).
    The branch is chosen so that Im(s) has the same sign as Im(z), which
    ensures the retarded Green function has Im(G) < 0 when Im(z) > 0.

    Args:
        z: Complex frequency array or scalar (omega + i * eta).
        a_inf: Asymptotic alpha value.
        b_inf: Asymptotic beta value (non-negative).

    Returns:
        Terminator value G_tail(z) for an infinite constant-coefficient tail.

    Raises:
        ValueError: If inputs are not finite or b_inf is negative.
    """
    if not np.isfinite(a_inf) or not np.isfinite(b_inf):
        raise ValueError("a_inf and b_inf must be finite.")
    if b_inf < 0.0:
        raise ValueError("b_inf must be non-negative.")

    z_arr = np.asarray(z, dtype=np.complex128)
    delta = z_arr - a_inf
    s = np.lib.scimath.sqrt(delta * delta - 4.0 * (b_inf**2))

    # Choose branch so that Im(s) has the same sign as Im(z).
    im_z = np.imag(z_arr)
    im_s = np.imag(s)

    # Where sign(Im s) != sign(Im z), flip s -> -s.
    # Treat zeros in Im(z) as upper half-plane by default (retarded).
    desired_sign = np.where(im_z >= 0.0, 1.0, -1.0)
    flip_mask = np.sign(im_s) != desired_sign
    if np.any(flip_mask):
        s = np.where(flip_mask, -s, s)

    g = (delta - s) / (2.0 * (b_inf**2))
    return g


def estimate_tail_parameters(
    alpha: NDArray[np.floating], beta: NDArray[np.floating], tail_window: int = 8
) -> tuple[float, float]:
    """Estimate a_inf and b_inf from the last few Lanczos coefficients.

    Averages the last 'tail_window' entries of alpha and beta to produce a
    simple and robust estimate of the asymptotic tail used by the terminator.

    Args:
        alpha: Array of alpha coefficients with shape (M,).
        beta: Array of beta coefficients with shape (M - 1,).
        tail_window: Number of trailing entries to average.

    Returns:
        Tuple (a_inf, b_inf) with non-negative b_inf.

    Raises:
        ValueError: If input arrays are too short or contain non-finite values.
    """
    if alpha.ndim != 1 or (beta.ndim != 1 and beta.size != 0):
        raise ValueError("alpha must be (M,), beta must be (M-1,) or empty if M==1.")
    if alpha.size == 0:
        raise ValueError("alpha cannot be empty.")
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
        raise ValueError("alpha and beta must be finite.")

    n_alpha = max(1, min(tail_window, alpha.size))
    n_beta = max(1, min(tail_window, beta.size)) if beta.size > 0 else 0

    a_inf = float(np.mean(alpha[-n_alpha:]))
    b_inf = float(np.mean(np.abs(beta[-n_beta:]))) if n_beta > 0 else 0.0
    b_inf = max(0.0, b_inf)
    return a_inf, b_inf


def haydock_greens_function(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    omega: NDArray[np.floating],
    eta: float,
    terminator: (
        Callable[[NDArray[np.complexfloating]], NDArray[np.complexfloating]] | None
    ) = None,
    tail_coupling_beta: float | None = None,
) -> NDArray[np.complexfloating]:
    """Evaluate Haydock continued-fraction Green function G(omega + i * eta).

    If 'terminator' is provided, it is used as the complex tail G_tail(z) at the
    bottom of the finite continued fraction, improving accuracy for small M.
    The last level is then:
        g_{M-1}(z) = 1 / (z - alpha[M-1] - beta_tail^2 * G_tail(z)),
    where beta_tail defaults to beta[-1] when available, or can be passed
    explicitly through 'tail_coupling_beta'.

    Args:
        alpha: Array of alpha coefficients with shape (M,).
        beta: Array of beta coefficients with shape (M - 1,).
        omega: Real frequency grid with shape (Nw,).
        eta: Positive small imaginary part.
        terminator: Optional callable tail G_tail(z). It must accept a complex
            array z with shape (Nw,) and return an array of the same shape.
        tail_coupling_beta: Optional coupling used at the last level when a
            terminator is provided. If None and beta.size > 0, defaults to
            float(beta[-1]). If both None and beta.size == 0, defaults to 0.0.

    Returns:
        Complex array G(z) with shape (Nw,).

    Raises:
        ValueError: On invalid inputs.
    """
    if eta <= 0.0 or not np.isfinite(eta):
        raise ValueError("eta must be positive and finite.")
    if alpha.ndim != 1:
        raise ValueError("alpha must be one dimensional.")
    if beta.ndim != 1 and beta.size != 0:
        raise ValueError("beta must be one dimensional or empty.")
    if alpha.size == 0:
        raise ValueError("alpha cannot be empty.")
    if beta.size not in (0, alpha.size - 1):
        raise ValueError("beta must have length M-1 or be empty if M==1.")

    z = omega.astype(np.float64, copy=False) + 1j * float(eta)
    n_levels = alpha.size

    # Determine tail coupling for the last level when a terminator is used.
    if terminator is not None:
        if tail_coupling_beta is None:
            b_tail = float(beta[-1]) if beta.size > 0 else 0.0
        else:
            b_tail = float(tail_coupling_beta)
        g_next = terminator(z)  # this is G_tail(z)
    else:
        b_tail = 0.0
        g_next = np.zeros_like(z, dtype=np.complex128)

    # Backward recurrence for continued fraction:
    # g_j = 1 / (z - alpha_j - beta_j^2 * g_{j+1}), j = M-2, ..., 0
    # and at j = M-1 (last level):
    #   if terminator:
    #       g_{M-1} = 1 / (z - alpha_{M-1} - b_tail^2 * g_tail)
    #   else:
    #       g_{M-1} = 1 / (z - alpha_{M-1})
    for j in range(n_levels - 1, -1, -1):
        if j == n_levels - 1:
            if terminator is None:
                g = 1.0 / (z - alpha[j])
            else:
                g = 1.0 / (z - alpha[j] - (b_tail**2) * g_next)
        else:
            g = 1.0 / (z - alpha[j] - (beta[j] ** 2) * g_next)
        g_next = g

    return g_next


def haydock_spectral_density(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    omega: NDArray[np.floating],
    eta: float,
    terminator: (
        Callable[[NDArray[np.complexfloating]], NDArray[np.complexfloating]] | None
    ) = None,
    tail_coupling_beta: float | None = None,
) -> NDArray[np.floating]:
    """Return rho(omega) = -1/pi * Im G(omega + i * eta) using Haydock CF.

    Args:
        alpha: Array of alpha coefficients with shape (M,).
        beta: Array of beta coefficients with shape (M - 1,).
        omega: Real frequency grid with shape (Nw,).
        eta: Positive small imaginary part.
        terminator: Optional terminator callable passed to haydock_greens_function.
        tail_coupling_beta: Optional coupling used at the last level with terminator.

    Returns:
        Real array rho(omega) with shape (Nw,).
    """
    G = haydock_greens_function(
        alpha=alpha,
        beta=beta,
        omega=omega,
        eta=eta,
        terminator=terminator,
        tail_coupling_beta=tail_coupling_beta,
    )
    return (-1.0 / np.pi) * np.imag(G)
