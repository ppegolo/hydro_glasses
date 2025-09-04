import numpy as np
import numpy.linalg as npl
import pytest

try:
    import scipy.sparse as sp

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from hydroglass.lanczos import (
    haydock_greens_function,
    haydock_spectral_density,
    lanczos_tridiagonal,
    tridiagonal_eigh,
)


def _make_symmetric_matrix(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = 0.5 * (A + A.T)
    return A


def test_lanczos_betas_nonnegative_and_shapes():
    n = 20
    m = 10
    A = _make_symmetric_matrix(n, seed=1)
    v0 = np.ones(n, dtype=float)
    alpha, beta, V = lanczos_tridiagonal(
        matrix_or_linear_operator=A,
        initial_vector=v0,
        num_steps=m,
        reorthogonalization="partial",
        atol_reorth=1e-12,
    )
    assert alpha.shape == (m,)
    assert beta.shape == (m - 1,)
    assert V.shape == (n, m)
    assert np.all(beta >= 0.0)


def test_lanczos_basis_is_orthonormal_to_tight_tolerance():
    n = 40
    m = 20
    A = _make_symmetric_matrix(n, seed=2)
    v0 = np.arange(1, n + 1, dtype=float)
    alpha, beta, V = lanczos_tridiagonal(
        matrix_or_linear_operator=A,
        initial_vector=v0,
        num_steps=m,
        reorthogonalization="full",
        atol_reorth=1e-13,
    )
    gram = V.T @ V
    err = npl.norm(gram - np.eye(m), ord=2)
    assert err < 5e-12


def test_ritz_values_approximate_extremal_eigenvalues():
    n = 80
    m = 25
    A = _make_symmetric_matrix(n, seed=3)
    evals_full = np.sort(npl.eigvalsh(A))
    v0 = np.ones(n, dtype=float)
    alpha, beta, _ = lanczos_tridiagonal(
        matrix_or_linear_operator=A,
        initial_vector=v0,
        num_steps=m,
        reorthogonalization="partial",
        atol_reorth=1e-12,
    )
    theta = np.sort(tridiagonal_eigh(alpha, beta))
    # Compare extremal Ritz values against full spectrum edges.
    # Lanczos converges faster at the edges; allow modest tolerance.
    assert pytest.approx(theta[0], rel=0, abs=5e-2) == evals_full[0]
    assert pytest.approx(theta[-1], rel=0, abs=5e-2) == evals_full[-1]


def test_haydock_on_diagonal_matrix_matches_analytic_green_function():
    # A diagonal matrix has exact continued fraction when the starting
    # vector is a basis vector aligned with one diagonal entry.
    diag = np.array([1.0, 2.0, 3.0], dtype=float)
    A = np.diag(diag)
    n = A.shape[0]
    v0 = np.zeros(n, dtype=float)
    v0[1] = 1.0  # pick the eigenvector with eigenvalue 2.0

    alpha, beta, _ = lanczos_tridiagonal(
        matrix_or_linear_operator=A,
        initial_vector=v0,
        num_steps=3,
        reorthogonalization="none",
    )
    # Continued fraction Green's function G(z) = <v0|(zI - A)^{-1}|v0>.
    omega = np.linspace(0.0, 4.0, 401)
    eta = 1e-2
    G = haydock_greens_function(alpha, beta, omega=omega, eta=eta)
    # Exact diagonal case: G(z) = 1 / (z - 2) since v0 selects eigenvalue 2.
    z = omega + 1j * eta
    G_exact = 1.0 / (z - 2.0)
    err = np.max(np.abs(G - G_exact))
    assert err < 5e-3


def test_haydock_spectral_density_integrates_to_one_for_basis_vector():
    diag = np.array([1.0, 2.0, 3.0], dtype=float)
    A = np.diag(diag)
    n = A.shape[0]
    v0 = np.zeros(n, dtype=float)
    v0[0] = 1.0  # selects eigenvalue 1.0 with unit weight
    alpha, beta, _ = lanczos_tridiagonal(
        matrix_or_linear_operator=A,
        initial_vector=v0,
        num_steps=3,
        reorthogonalization="none",
    )
    omega = np.linspace(-1.0, 5.0, 3001)
    eta = 1e-2
    rho = haydock_spectral_density(alpha, beta, omega=omega, eta=eta)
    # Spectral density integrates to 1 when starting vector is normalized.
    integral = np.trapezoid(rho, omega)
    assert pytest.approx(integral, rel=5e-3, abs=5e-3) == 1.0


@pytest.mark.skipif(not SCIPY_OK, reason="scipy.sparse not available")
def test_sparse_matrix_path_yields_same_tridiagonalization():
    A_dense = _make_symmetric_matrix(30, seed=11)
    A_sparse = sp.csr_matrix(A_dense)
    v0 = np.linspace(1.0, 2.0, A_dense.shape[0])
    m = 12

    alpha_d, beta_d, _ = lanczos_tridiagonal(A_dense, v0, m)
    alpha_s, beta_s, _ = lanczos_tridiagonal(A_sparse, v0, m)

    assert np.allclose(alpha_d, alpha_s, rtol=1e-12, atol=1e-12)
    assert np.allclose(beta_d, beta_s, rtol=1e-12, atol=1e-12)
