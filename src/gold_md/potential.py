
import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

# - load_eam()

def load_eam(filename):
    filename = Path(filename)
    
    if not filename.is_absolute():
        filename = Path(__file__).parent / filename

    with open(filename, "r") as file:
        lines = file.readlines()

    # First three lines are comments
    header = lines[3].split()
    nelements = int(header[0])

    if nelements != 1 or header[1] != "Au":
        raise ValueError("This code requires a single-element Au EAM file.")

    # Line 5 contains grid information
    grid = lines[4].split()

    nrho = int(grid[0])
    drho = float(grid[1])
    nr = int(grid[2])
    dr = float(grid[3])
    cutoff = float(grid[4])

    # Line 6 contains gold atomic information
    element_info = lines[5].split()

    atomic_number = int(element_info[0])
    mass = float(element_info[1])

    # Read the tabulated data after the six header lines
    data = np.array([
        float(value)
        for line in lines[6:]
        for value in line.split()
    ])

    # Extract the three tabulated functions
    F_values = data[:nrho]
    density_values = data[nrho:nrho + nr]
    rphi_values = data[nrho + nr:nrho + 2 * nr]

    if len(F_values) != nrho or len(density_values) != nr or len(rphi_values) != nr:
        raise ValueError("Unexpected EAM file format or incomplete data.")

    rho_grid = np.arange(nrho) * drho
    r_grid = np.arange(nr) * dr

    F_spline = CubicSpline(rho_grid, F_values)
    density_spline = CubicSpline(r_grid, density_values)
    rphi_spline = CubicSpline(r_grid, rphi_values)

    return mass, cutoff, F_spline, density_spline, rphi_spline

# - potential(positions)
def potential(positions):
    natoms = len(positions)

    energy = 0.0
    forces = np.zeros_like(positions)
    rho = np.zeros(natoms)

    max_pairs = natoms * (natoms - 1) // 2
    pairs = np.zeros((max_pairs, 5))
    npairs = 0

    # Calculating the local electron density
    for i in range(natoms):
        for j in range(i+1, natoms):

            # Separation vector from atom i to atom j
            rij = positions[i] - positions[j]
            r2 = np.sum(rij**2)

            if r2 < cutoff**2:
                r = np.sqrt(r2)
                pairs[npairs] = [i, j, rij[0], rij[1], rij[2]]
                npairs += 1

                density_contribution = density_spline(r)
                rho[i] += density_contribution
                rho[j] += density_contribution

    # Calculating the embedding energy
    embedding_derivatives = np.zeros(natoms)

    for i in range(natoms):
        energy += F_spline(rho[i])
        embedding_derivatives[i] = F_spline.derivative()(rho[i])

    # Calculate pair energy and forces
    for k in range(npairs):
        i = int(pairs[k, 0])
        j = int(pairs[k, 1])
        rij = positions[i] - positions[j]
        r = np.sqrt(np.sum(rij**2))

        rphi = rphi_spline(r)
        phi = rphi / r
        dphi = rphi_spline.derivative()(r) / r - rphi / r**2
        energy += phi

        # Calculate the embedding contribution to the force
        drho = density_spline.derivative()(r)
        dEdr = (embedding_derivatives[i] + embedding_derivatives[j]) * drho + dphi
        fij = -dEdr * rij / r

        # Forces on the two atoms are equal and opposite
        forces[i] += fij
        forces[j] -= fij

    return energy, forces

def kinetic(vel, masses, accel_conversion):
    return 0.5 * np.sum(masses[:, None] * vel**2) / accel_conversion

# Load the potential
mass, cutoff, F_spline, density_spline, rphi_spline = (load_eam("Au-Grochola-JCP05.eam.alloy"))
