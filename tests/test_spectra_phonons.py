import numpy as np
import pytest

from hydroglass.spectra.phonon import Mesh1D, PhononModel


def _make_single_atom_modes():
    # Three Cartesian modes, orthonormal columns; frequencies 1, 2, 3.
    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    eigvecs = np.eye(3, dtype=float)  # shape (3, 3), columns are ex, ey, ez
    return freqs, eigvecs


def _trapz(y, x):
    return np.trapezoid(y, x=x)


def test_dsf_longitudinal_single_atom_peak_and_integral_gaussian():
    """With L projection along x, only the ex mode contributes.
    The integral equals (n_B + 1) * w / (2 * omega) = 1 / (2 * 1) = 0.5 at T=0.
    """
    freqs, eigvecs = _make_single_atom_modes()
    model = PhononModel(
        frequencies=freqs,
        eigenvectors=eigvecs,
        masses=None,
        q_vectors=None,
        omega_mesh=Mesh1D(points=np.linspace(0.0, 4.0, 4001)),
    )

    q = np.array([1.0, 0.0, 0.0], dtype=float)
    omega, sqw = model.dynamic_structure_factor(
        q_vectors=q,  # shape (3,)
        temperature=0.0,
        polarization="longitudinal",
        broadening={"kind": "gaussian", "sigma": 0.02},
    )

    # Peak near 1.0
    idx_max = np.argmax(sqw)
    assert np.isclose(omega[idx_max], 1.0, atol=omega[1] - omega[0])

    # Integral equals 1/(2*1) because only the x mode contributes with weight 1.
    integral = _trapz(sqw, omega)
    assert np.isclose(integral, 0.5, rtol=2e-3, atol=2e-3)


def test_dsf_transverse_single_atom_integral_gaussian():
    """With T projection when q is along x, the y and z modes contribute.
    The integral is sum_m (n_B+1) * w_m / (2*omega_m) = 1/(2*2) + 1/(2*3).
    """
    freqs, eigvecs = _make_single_atom_modes()
    omega_mesh = Mesh1D(points=np.linspace(0.0, 4.0, 4001))
    model = PhononModel(
        frequencies=freqs,
        eigenvectors=eigvecs,
        masses=None,
        q_vectors=None,
        omega_mesh=omega_mesh,
    )

    q = np.array([1.0, 0.0, 0.0], dtype=float)

    omegaT, sqwT = model.dynamic_structure_factor(
        q_vectors=q,  # shape (3,)
        temperature=0.0,
        polarization="transverse",
        broadening={"kind": "gaussian", "sigma": 0.02},
    )

    # Expected integral: y and z only, with weights 1 each.
    expected = 1.0 / (2.0 * 2.0) + 1.0 / (2.0 * 3.0)  # 0.25 + 0.166666...
    integralT = _trapz(sqwT, omegaT)
    assert np.isclose(integralT, expected, rtol=2e-3, atol=2e-3)


def test_dsf_lorentzian_longitudinal_integral_with_equal_L_weights():
    """q at equal components gives identical L weights for the three Cartesian modes.
    The integral equals sum_m w_m/(2*omega_m) with w_m = 1/3 for each mode.
    """
    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    eigvecs = np.eye(3, dtype=float)
    omega = np.linspace(0.0, 4.0, 8001)
    model = PhononModel(
        frequencies=freqs,
        eigenvectors=eigvecs,
        masses=None,
        q_vectors=None,
        omega_mesh=Mesh1D(points=omega),
    )
    q = np.array([1.0, 1.0, 1.0], dtype=float)
    q /= np.linalg.norm(q)

    omega_out, sqw = model.dynamic_structure_factor(
        q_vectors=q,  # shape (3,)
        temperature=0.0,
        polarization="longitudinal",
        broadening={"kind": "lorentzian", "gamma": 0.03},
    )

    # Each mode has L weight 1/3. Integral is sum ( (1/3) / (2*omega_m) ).
    expected = (1.0 / 3.0) * (1.0 / (2.0 * 1.0) + 1.0 / (2.0 * 2.0) + 1.0 / (2.0 * 3.0))
    integral = _trapz(sqw, omega_out)
    assert np.isclose(integral, expected, rtol=1e-2, atol=2e-3)


def test_dsf_raises_without_mesh():
    """Calling DSF without an omega mesh in the model or call should raise."""
    freqs, eigvecs = _make_single_atom_modes()
    model = PhononModel(
        frequencies=freqs,
        eigenvectors=eigvecs,
        masses=None,
        q_vectors=None,
        omega_mesh=None,
    )
    with pytest.raises(ValueError):
        model.dynamic_structure_factor(
            q_vectors=np.array([1.0, 0.0, 0.0], dtype=float),
            temperature=0.0,
            polarization="longitudinal",
            broadening={"kind": "gaussian", "sigma": 0.02},
        )
