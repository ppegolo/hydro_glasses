# Phonons

High-level tools for vibrational spectra.

- `Mesh1D`: one-dimensional frequency mesh.
- `PhononModel`: container with frequencies, eigenvectors, positions, masses, and optional scattering weights.  
  Includes projections (L/T) and dynamic structure factor.

## Example

```python
import numpy as np
from hydroglass.spectra.phonon import Mesh1D, PhononModel

frequencies = np.array([1.0, 2.0, 3.0])
eigenvectors = np.eye(3)
omega_mesh = Mesh1D(points=np.linspace(0.0, 4.0, 4001))

model = PhononModel(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=None,
    q_vectors=None,
    omega_mesh=omega_mesh,
)

q = np.array([1.0, 0.0, 0.0])
omega, sqw = model.dynamic_structure_factor(
    q_vectors=q[None, :],
    temperature=0.0,
    polarization="longitudinal",
    broadening={"kind": "gaussian", "sigma": 0.05},
)
```

## API reference

::: hydroglass.spectra.phonon.Mesh1D

::: hydroglass.spectra.phonon.PhononModel