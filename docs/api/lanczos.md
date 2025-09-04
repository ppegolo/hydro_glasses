# Lanczos

This module implements symmetric Lanczos tridiagonalization and the Haydock continued fraction.

- `lanczos_tridiagonal`: build the Krylov basis and tridiagonal coefficients `(alpha, beta)`.
- `tridiagonal_eigh`: eigenvalues of the implied tridiagonal matrix.
- `haydock_greens_function`: evaluate the continued-fraction Green’s function.
- `haydock_spectral_density`: `rho(omega) = -1/pi * Im G(omega + i * eta)`.

!!! note
    The routines accept dense NumPy arrays or `scipy.sparse` matrices.  
    Docstrings follow Google style.

## Quick start

```python
import numpy as np
from hydroglass.lanczos import (
    lanczos_tridiagonal,
    tridiagonal_eigh,
    haydock_greens_function,
    haydock_spectral_density,
)

rng = np.random.default_rng(7)
A = rng.normal(size=(60, 60)); A = 0.5 * (A + A.T)
v0 = rng.normal(size=60); v0 /= np.linalg.norm(v0)

alpha, beta, _ = lanczos_tridiagonal(A, v0, num_steps=20, reorthogonalization="partial")

eta = 0.02
theta = tridiagonal_eigh(alpha, beta)
omega = np.linspace(theta.min() - 10*eta, theta.max() + 10*eta, 4001)

G = haydock_greens_function(alpha, beta, omega, eta)
rho = haydock_spectral_density(alpha, beta, omega, eta)
```

## API reference

::: hydroglass.lanczos.lanczos_tridiagonal

::: hydroglass.lanczos.tridiagonal_eigh

::: hydroglass.lanczos.haydock_greens_function

::: hydroglass.lanczos.haydock_spectral_density