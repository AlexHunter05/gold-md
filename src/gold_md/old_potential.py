import numpy as np

def old_potential(positions, cutoff, F_spline, density_spline, rphi_spline):
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