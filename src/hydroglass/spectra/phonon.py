# src/hydroglass/spectra/phonon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .broadening import gaussian_kernel, lorentzian_kernel


@dataclass(slots=True)
class Mesh1D:
    """Simple one dimensional mesh descriptor.

    Attributes:
        points: One dimensional array of grid points. Must be monotonic.
        spacing: Optional uniform spacing. If None, no assumption is made.
    """

    points: NDArray[np.floating]
    spacing: float | None = None


@dataclass(slots=True)
class PhononModel:
    """Phonon-like object with eigenvalues and eigenvectors.

    This container stores harmonic modes and exposes methods to compute
    projections and the dynamic structure factor on a frequency grid.

    Shape conventions:
        - Let natoms be the number of atoms.
        - Degrees of freedom ndof = 3 * natoms.
        - eigenvectors has shape (ndof, nmodes) with columns being modes.
        - frequencies has shape (nmodes,).

    Mass convention:
        - eigenvectors_kind = "raw_cartesian": physical displacements are
          e_{a,alpha}(m) / sqrt(M_a) if masses are provided.
        - eigenvectors_kind = "mass_weighted": eigenvectors are already
          mass-weighted displacements; no additional 1/sqrt(M) factor.

    Eigenvalue convention:
        - eigenvalue_kind = "omega": provided frequencies are ω >= 0.
        - eigenvalue_kind = "omega_squared": provided eigenvalues are ω^2 >= 0,
          and the model converts to ω by taking a safe square root.

    Positions and phases:
        - If positions is provided (shape (natoms, 3)), the DSF uses phase
          factors exp(i q dot r_a) per atom. Optional real scattering_weights
          of shape (natoms,) weight atomic contributions (for example,
          coherent scattering lengths).

    Attributes:
        frequencies: Mode frequencies ω_m >= 0 with shape (nmodes,).
        eigenvectors: Eigenvector matrix with shape (ndof, nmodes).
        masses: Optional mass vector with shape (natoms,).
        positions: Optional Cartesian positions with shape (natoms, 3).
        scattering_weights: Optional per-atom real weights with shape (natoms,).
        q_vectors: Optional preset q-points with shape (nq, 3).
        omega_mesh: Optional one dimensional frequency mesh.
        eigenvectors_kind: "raw_cartesian" or "mass_weighted".
        eigenvalue_kind: "omega" or "omega_squared".
    """

    frequencies: NDArray[np.floating]
    eigenvectors: NDArray[np.floating]
    masses: NDArray[np.floating] | None = None
    positions: NDArray[np.floating] | None = None
    scattering_weights: NDArray[np.floating] | None = None
    q_vectors: NDArray[np.floating] | None = None
    omega_mesh: Mesh1D | None = None
    eigenvectors_kind: Literal["raw_cartesian", "mass_weighted"] = "raw_cartesian"
    eigenvalue_kind: Literal["omega", "omega_squared"] = "omega"

    def __post_init__(self) -> None:
        # Validate eigenvectors
        if self.eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must have shape (ndof, nmodes).")
        ndof, nmodes = self.eigenvectors.shape

        # q_vectors
        if self.q_vectors is not None:
            if self.q_vectors.ndim != 2 or self.q_vectors.shape[1] != 3:
                raise ValueError("q_vectors must have shape (nq, 3) when provided.")

        # Masses and positions tie ndof to natoms
        if ndof % 3 != 0:
            raise ValueError("eigenvectors first dimension must be a multiple of 3.")
        natoms_from_vecs = ndof // 3
        if self.masses is not None:
            if self.masses.ndim != 1:
                raise ValueError("masses must be one dimensional when provided.")
            if self.masses.shape[0] != natoms_from_vecs:
                raise ValueError(
                    "masses length must equal eigenvectors first dimension / 3."
                )
        if self.positions is not None:
            if self.positions.ndim != 2 or self.positions.shape[1] != 3:
                raise ValueError("positions must have shape (natoms, 3) when provided.")
            if self.positions.shape[0] != natoms_from_vecs:
                raise ValueError(
                    "positions length must equal eigenvectors first dimension / 3."
                )
            if self.scattering_weights is not None:
                if self.scattering_weights.ndim != 1:
                    raise ValueError("scattering_weights must be one dimensional.")
                if self.scattering_weights.shape[0] != self.positions.shape[0]:
                    raise ValueError("scattering_weights must have shape (natoms,).")

        # Frequencies: accept ω or λ=ω^2 and convert to ω
        freqs = np.asarray(self.frequencies, dtype=float)
        if freqs.ndim != 1:
            raise ValueError("frequencies must be one dimensional.")
        if self.eigenvalue_kind == "omega_squared":
            freqs = np.sqrt(np.clip(freqs, a_min=0.0, a_max=None))
        if np.any(freqs < 0.0) or not np.all(np.isfinite(freqs)):
            raise ValueError("frequencies must be nonnegative and finite.")
        if freqs.shape[0] != nmodes:
            raise ValueError(
                "frequencies length must equal number of eigenvector columns."
            )
        self.frequencies = freqs  # store back a validated ω array

        # Omega mesh sanity
        if self.omega_mesh is not None:
            if self.omega_mesh.points.ndim != 1:
                raise ValueError("omega_mesh.points must be one dimensional.")

        # eigenvectors_kind sanity
        if self.eigenvectors_kind not in ("raw_cartesian", "mass_weighted"):
            raise ValueError(
                "eigenvectors_kind must be 'raw_cartesian' or 'mass_weighted'."
            )

    def _natoms_ndof_nmodes(self) -> tuple[int, int, int]:
        """Return natoms, ndof, nmodes and validate 3D blocks."""
        ndof, nmodes = self.eigenvectors.shape
        if ndof % 3 != 0:
            raise ValueError("eigenvectors first dimension must be a multiple of 3.")
        natoms = ndof // 3
        return natoms, ndof, nmodes

    def _displacements(self) -> NDArray[np.floating]:
        """Return physical displacements d_{a,alpha,m} with shape (natoms, 3, nmodes).

        If eigenvectors_kind is 'mass_weighted', eigenvectors are already displacements.
        If 'raw_cartesian' and masses are provided, apply 1/sqrt(M) per atom.
        If masses are None under 'raw_cartesian', treat eigenvectors as displacements
        but note the missing mass normalization in intensities.
        """
        natoms, _ndof, _nmodes = self._natoms_ndof_nmodes()
        vecs = self.eigenvectors.reshape(natoms, 3, -1)

        if self.eigenvectors_kind == "mass_weighted":
            return vecs

        # raw_cartesian
        if self.masses is not None:
            inv_sqrt_m = 1.0 / np.sqrt(self.masses)  # (natoms,)
            vecs = vecs * inv_sqrt_m[:, None, None]
        return vecs

    def _phase_factors(
        self, q_vector: NDArray[np.floating]
    ) -> NDArray[np.complexfloating] | None:
        """
        Return exp(i q dot r_a) per atom if positions are available; otherwise None.
        """
        if self.positions is None:
            return None
        q_dot_r = self.positions @ q_vector  # shape (natoms,)
        return np.exp(1j * q_dot_r)

    def _per_mode_vector_amplitudes(
        self, q_vector: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Return B_m(q) = sum_a b_a d_{a,m} exp(i q dot r_a) as a (3, nmodes) array.

        If positions are not available, phases are omitted and a phase-less sum is
        returned. Scattering weights default to ones if not provided.
        """
        d = self._displacements()  # (natoms, 3, nmodes)
        natoms, _three, _nmodes = d.shape

        if self.scattering_weights is not None:
            w = self.scattering_weights.astype(float, copy=False)  # (natoms,)
        else:
            w = np.ones(natoms, dtype=float)

        phase = self._phase_factors(q_vector)
        if phase is None:
            B = (d * w[:, None, None]).sum(axis=0).astype(np.complex128)  # (3, nmodes)
        else:
            B = (d * w[:, None, None] * phase[:, None, None]).sum(
                axis=0
            )  # (3, nmodes), complex
        return B

    def longitudinal_projection(
        self, q_vector: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Phase-less longitudinal diagnostic: |sum_a d_{a,m}| projected onto q_hat,
        squared.

        This is a local polarization measure; it is not the DSF amplitude.
        """
        if q_vector.shape != (3,):
            raise ValueError("q_vector must have shape (3,).")
        q_norm = float(np.linalg.norm(q_vector)) + 1e-12
        if not np.isfinite(q_norm) or q_norm <= 0.0:
            raise ValueError("q_vector must be nonzero and finite.")
        q_hat = q_vector / q_norm

        d = self._displacements()  # (natoms, 3, nmodes)
        d_total = d.sum(axis=0)  # (3, nmodes)
        proj = np.tensordot(q_hat, d_total, axes=([0], [0]))  # (nmodes,)
        w = np.real_if_close(proj * proj.conjugate()).astype(float)
        return w

    @staticmethod
    def _bose_factor(
        frequency: NDArray[np.floating], temperature: float
    ) -> NDArray[np.floating]:
        """Return Bose-Einstein occupation with k_B = 1 and hbar = 1 units."""
        if temperature <= 0.0:
            return np.zeros_like(frequency, dtype=float)
        x = np.clip(frequency / temperature, 0.0, 700.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            n = 1.0 / (np.exp(x) - 1.0)
        n[~np.isfinite(n)] = 0.0
        return n.astype(float)

    # inside src/hydroglass/spectra/phonon.py

    def dynamic_structure_factor(
        self,
        q_vectors: NDArray[np.floating] | NDArray[np.float64],
        temperature: float,
        polarization: Literal["longitudinal", "transverse"] = "longitudinal",
        broadening: dict[str, float] | None = None,
        chunk_n_modes: int | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute one-phonon S(q, omega) for one or many q vectors.

        This is a vectorized interface. If q_vectors has shape (3,), the return
        has shape (nomega,). If q_vectors has shape (nq, 3), the return has
        shape (nomega, nq).

        Model:
            S(q, omega) = sum_m [(n_B + 1) / (2 * omega_m)] * W_m(q) * K(omega|omega_m)

        Projections:
            longitudinal: W_m(q) = | q dot B_m(q) |^2
            transverse:   W_m(q) = | B_m(q) - q_hat (q_hat dot B_m(q)) |^2

        Args:
            q_vectors: Array of shape (3,) or (nq, 3). All q must be finite and nonzero.
            temperature: Absolute temperature (k_B = 1).
            polarization: 'longitudinal' or 'transverse'.
            broadening: Kernel specification dict, e.g. {
                                                            'kind': 'gaussian',
                                                            'sigma': ...
                                                        }
                or {'kind': 'lorentzian', 'gamma': ...}. If None, a default Gaussian
                width proportional to the spectral span is used.
            chunk_n_modes: Optional number of modes per chunk to limit memory use.

        Returns:
            Tuple (omega, sqw) where omega has shape (nomega,) and sqw has shape:
              - (nomega,) if q_vectors was shape (3,)
              - (nomega, nq) if q_vectors was shape (nq, 3)

        Raises:
            ValueError: On invalid shapes, missing omega mesh, or invalid kernel params.
        """
        # Omega mesh
        if (
            self.omega_mesh is None
            or self.omega_mesh.points.ndim != 1
            or self.omega_mesh.points.size == 0
        ):
            raise ValueError("PhononModel must have a nonempty omega mesh.")
        omega = np.asarray(self.omega_mesh.points, dtype=float)
        nomega = omega.size

        # Normalize q input to (nq, 3) for computation
        q_vectors = np.asarray(q_vectors, dtype=float)
        single_q_input = False
        if q_vectors.shape == (3,):
            q_vectors = q_vectors[None, :]
            single_q_input = True
        if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
            raise ValueError("q_vectors must have shape (3,) or (nq, 3).")
        nq = q_vectors.shape[0]

        q_norms = np.linalg.norm(q_vectors, axis=1) + 1e-12
        if not np.all(np.isfinite(q_norms)) or np.any(q_norms <= 0.0):
            raise ValueError("All q vectors must be nonzero and finite.")
        q_hat = q_vectors / q_norms[:, None]  # for transverse decomposition

        # Broadening kernel matrix K(omega | omega_m)
        kind = None if broadening is None else broadening.get("kind", None)
        if kind == "gaussian":
            sigma = float(broadening.get("sigma", 0.0))
            if sigma <= 0.0:
                raise ValueError("Gaussian broadening requires positive 'sigma'.")
            K = gaussian_kernel(omega=omega, centers=self.frequencies, sigma=sigma)
        elif kind == "lorentzian":
            gamma = float(broadening.get("gamma", 0.0))
            if gamma <= 0.0:
                raise ValueError("Lorentzian broadening requires positive 'gamma'.")
            K = lorentzian_kernel(omega=omega, centers=self.frequencies, gamma=gamma)
        else:
            freq_scale = (
                float(np.max(self.frequencies)) if self.frequencies.size else 1.0
            )
            sigma = 0.02 * max(freq_scale, 1.0)
            K = gaussian_kernel(omega=omega, centers=self.frequencies, sigma=sigma)

        # Thermal prefactor (n_B + 1) / (2 * omega_m)
        n_bose = self._bose_factor(self.frequencies, float(temperature))
        safe_omega = np.where(self.frequencies > 1e-12, self.frequencies, np.inf)
        mode_prefactor = (n_bose + 1.0) / (2.0 * safe_omega)  # (nmodes,)

        # Precompute phases
        if self.positions is None:
            phase = None
        else:
            # shape (natoms, nq)
            phase = np.exp(1j * (self.positions @ q_vectors.T))

        # Displacements and per-atom weights
        D = self._displacements()  # (natoms, 3, nmodes)
        natoms, _three, nmodes = D.shape
        if self.scattering_weights is not None:
            w = self.scattering_weights.astype(float, copy=False)  # (natoms,)
        else:
            w = np.ones(natoms, dtype=float)

        # Chunk size heuristic
        if chunk_n_modes is None:
            bytes_per_complex = 16  # complex128
            target_bytes = 128 * 1024 * 1024
            m_blk = max(1, int(target_bytes / max(1, 3 * nq * bytes_per_complex)))
            chunk_n_modes = min(m_blk, nmodes)

        sqw_grid = np.zeros((nomega, nq), dtype=float)

        for m0 in range(0, nmodes, chunk_n_modes):
            m1 = min(nmodes, m0 + chunk_n_modes)
            D_blk = D[:, :, m0:m1]  # (natoms, 3, m_blk)
            DW = D_blk * w[:, None, None]

            # Build B(alpha, m, j) = sum_a DW(a, alpha, m) * phase(a, j)
            if phase is None:
                B = np.empty((3, m1 - m0, nq), dtype=np.complex128)
                ones = np.ones(natoms, dtype=float)
                for alpha in range(3):
                    tmp = DW[:, alpha, :].T @ ones  # (m_blk,)
                    B[alpha, :, :] = tmp[:, None]
            else:
                B = np.empty((3, m1 - m0, nq), dtype=np.complex128)
                for alpha in range(3):
                    B[alpha, :, :] = DW[:, alpha, :].T @ phase  # (m_blk, nq)

            # Project
            if polarization == "longitudinal":
                # Full q projection: A_L = q ⋅ B, shape (m_blk, nq)
                A_L = np.einsum("ja,amj->mj", q_vectors, B, optimize=True)
                weights_blk = np.real(A_L * A_L.conjugate()).astype(
                    float
                )  # (m_blk, nq)
            else:
                # Remove component along unit q_hat
                A_L_hat = np.einsum(
                    "ja,amj->mj", q_hat, B, optimize=True
                )  # (m_blk, nq)
                # B_T[alpha, m, j] = B[alpha, m, j] - q_hat[j, alpha] * A_L_hat[m, j]
                B_T = B - (q_hat.T[:, None, :] * A_L_hat[None, :, :])
                weights_blk = np.real(np.sum(B_T * B_T.conjugate(), axis=0)).astype(
                    float
                )
                weights_blk = np.clip(weights_blk, a_min=0.0, a_max=None)

            amps_blk = mode_prefactor[m0:m1][:, None] * weights_blk  # (m_blk, nq)
            sqw_grid += K[:, m0:m1] @ amps_blk

        # Clamp tiny negative round-off
        np.maximum(sqw_grid, 0.0, out=sqw_grid)

        if single_q_input:
            return omega, sqw_grid[:, 0]
        return omega, sqw_grid
