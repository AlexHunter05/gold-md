
import numpy as np
from .potential import potential

# Paste your kinetic() function here.

# Paste your AMDC class here.
class AMDC:
    def __init__(self):
        # Simulation settings
        self.stride = 10
        self.timestep = 1.0
        self.temperature = 300.0
        self.friction = 0.01

        # Initial atomic configuration
        self.positions = np.array(initial_positions, dtype=float)
        self.natoms = len(self.positions)
        self.masses = np.full(self.natoms, mass)
        self.moving_atoms = np.ones(self.natoms)
        self.non_conservative_atoms = np.zeros(self.natoms)

        # Physical constants
        self.kB = 8.617333262e-5
        self.accel_conversion = 0.0096485332

        # Initial thermal velocities
        self.rng = np.random.default_rng(42)
        velocity_sigma = np.sqrt(self.kB * self.temperature * self.accel_conversion / self.masses[0])
        self.vel = self.rng.normal(0, velocity_sigma, size=(self.natoms, 3))
        self.vel -= np.mean(self.vel, axis=0)

        # Langevin thermostat coefficients
        delta = self.timestep / 2
        self.decay = np.exp(-self.friction * delta)
        self.random_factor = np.sqrt(self.kB * self.temperature * self.accel_conversion / self.masses[0] * (1 - np.exp(-2 * self.friction * delta)))

        # Track energy exchanged
        self.therm = 0.0

        # Initial forces and energy
        self.eng, self.forces = potential(self.positions)

        self.trajectory = []
        self.potential_energies = []
        self.kinetic_energies = []
        self.total_energies = []

    def set_position(self, positions):
        self.positions = np.array(positions, dtype=float)
        self.eng, self.forces = potential(self.positions)

    def md(self, nsteps):
        # Velocity Verlet algorithm
        for step in range(nsteps):
            self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
            self.vel = self.vel * self.decay + self.random_factor * np.random.normal(size=(self.natoms, 3))
            self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

            self.vel += (0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep)
            self.positions += self.vel * self.timestep
            self.eng, self.forces = potential(self.positions)
            self.vel += (0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep)

            self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
            self.vel = self.vel * self.decay + self.random_factor * np.random.normal(size=(self.natoms, 3))
            self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

            # Record energies
            if step % self.stride == 0:
                KE = kinetic(self.vel, self.masses, self.accel_conversion)
                self.trajectory.append(self.positions.copy())
                self.potential_energies.append(self.eng)
                self.kinetic_energies.append(KE)
                self.total_energies.append(self.eng + KE + self.therm)

        self.trajectory = np.array(self.trajectory)
