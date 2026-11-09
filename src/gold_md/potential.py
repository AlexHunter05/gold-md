import numpy as np

def potential(positions, cutoff, F_spline, density_spline, rphi_spline, F_derivative, density_derivative, rphi_derivative):
    natoms = len(positions)
  
    forces = np.zeros_like(positions)
    rho = np.zeros(natoms)

    # Get all unique atom pairs
    i, j = np.triu_indices(natoms, k=1)

    # Calculate all separation vectors and distances
    rij = positions[i] - positions[j]
    r2 = np.sum(rij**2, axis=1)

    # Apply cutoff
    mask = r2 < cutoff**2

    i = i[mask]
    j = j[mask]
    rij = rij[mask]
    r = np.sqrt(r2[mask])

    # Calculate local electron densities
    density = density_spline(r)

    np.add.at(rho, i, density)
    np.add.at(rho, j, density)

    # Calculate embedding energy and derivatives
    embedding_energy = np.sum(F_spline(rho))
    embedding_derivatives = F_derivative(rho)

    # Calculate pair energy
    rphi = rphi_spline(r)
    phi = rphi / r

    pair_energy = np.sum(phi)

    # Calculate force derivatives
    dphi = rphi_derivative(r) / r - rphi / r**2
    drho = density_derivative(r)

    dEdr = (embedding_derivatives[i] + embedding_derivatives[j]) * drho + dphi

    # Calculate pair forces
    fij = -dEdr[:, None] * rij / r[:, None]

    np.add.at(forces, i, fij)
    np.add.at(forces, j, -fij)

    # Total potential energy
    energy = embedding_energy + pair_energy

    return energy, forces

def kinetic(vel, masses, accel_conversion):
    return 0.5 * np.sum(masses[:, None] * vel**2) / accel_conversion
