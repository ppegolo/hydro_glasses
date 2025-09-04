# hydroglass

Lightweight Python package for harmonic and anharmonic lattice dynamics in amorphous solids.

## Install

```bash
pip install -e .
```

Optional extras for docs:

```bash
pip install -e .[docs]
```

## Quick links

- API: [Lanczos](api/lanczos.md)
- API: [Phonons](api/phonon.md)

## Minimal example (Lanczos)

```python
import numpy as np
from hydroglass.lanczos import lanczos_tridiagonal, tridiagonal_eigh, haydock_spectral_density

rng = np.random.default_rng(7)
A = rng.normal(size=(60, 60)); A = 0.5 * (A + A.T)
v0 = rng.normal(size=60); v0 /= np.linalg.norm(v0)

alpha, beta, _ = lanczos_tridiagonal(A, v0, num_steps=20)
eta = 0.02
theta = tridiagonal_eigh(alpha, beta)
omega = np.linspace(theta.min() - 10*eta, theta.max() + 10*eta, 4001)
rho = haydock_spectral_density(alpha, beta, omega, eta)
```