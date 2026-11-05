import numpy as np
from .load_eam import load_eam
from .potential import potential, kinetic

class AMDC:
    def __init__(self, eam_file, positions, ensemble="NVT", seed=42):
        # Load EAM potential
        self.mass, self.cutoff, self.F_spline, self.density_spline, self.rphi_spline = load_eam(eam_file)

        # Select ensemble
        self.ensemble = ensemble.upper()
        if self.ensemble not in ["NVE", "NVT"]:
            raise ValueError("Ensemble must be either 'NVE' or 'NVT'")

        # Simulation settings
        self.steps = 0
        self.stride = 10
        self.timestep = 1.0
        self.temperature = 300.0
        self.friction = 0.01

        # Physical constants
        self.kB = 8.617333262e-5
        self.accel_conversion = 0.0096485332

        # Initial atomic configuration
        self.positions = np.array(positions, dtype=float)
        self.natoms = len(self.positions)
        self.masses = np.full(self.natoms, self.mass)
        self.moving_atoms = np.ones(self.natoms)

        # Non-conservative force settings
        self.K = 1.0
        self.centre = np.zeros(3)
        self.non_conservative_atoms = np.zeros(self.natoms)

        # Initial thermal velocities
        self.rng = np.random.default_rng(seed)
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
        self.calculate_forces()

        self.trajectory = []
        self.potential_energies = []
        self.kinetic_energies = []
        self.total_energies = []

    def fix_atoms(self, atoms):
        self.moving_atoms[atoms] = 0
        self.vel[atoms] = 0

    def apply_non_conservative_force(self, atoms):
        self.non_conservative_atoms[atoms] = 1

    def non_conservative_force(self, K, centre):
        x0, y0, z0 = centre

        forces = np.zeros((self.natoms, 3))

        dx = self.positions[:, 0] - x0
        dy = self.positions[:, 1] - y0

        forces[:, 0] = -K * dy
        forces[:, 1] = K * dx
        return forces * self.non_conservative_atoms[:, None]

    def calculate_forces(self):
        self.eng, self.forces = potential(
            self.positions,
            self.cutoff,
            self.F_spline,
            self.density_spline,
            self.rphi_spline
        )
        self.forces += self.non_conservative_force(self.K, self.centre)

    def md(self, nsteps=None, until_escape=False, max_steps=25000):
        # Automatically calculate escape radius
        if until_escape:
            escape_radius = np.max(np.linalg.norm(self.positions, axis=1)) + 2.88
            nsteps = max_steps

        # Velocity Verlet algorithm
        for step in range(nsteps):
            if self.ensemble == "NVT":
                self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
                self.vel = (self.vel * self.decay + self.random_factor * self.rng.normal(size=(self.natoms, 3))) * self.moving_atoms[:, None]
                self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

            self.vel += (0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep) * self.moving_atoms[:, None]
            self.positions += self.vel * self.timestep * self.moving_atoms[:, None]
            self.calculate_forces()
            self.vel += (0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep) * self.moving_atoms[:, None]

            if self.ensemble == "NVT":
                self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
                self.vel = (self.vel * self.decay + self.random_factor * self.rng.normal(size=(self.natoms, 3))) * self.moving_atoms[:, None]
                self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

            self.steps += 1
            escaped = until_escape and np.linalg.norm(self.positions[0]) > escape_radius

            # Record energies and trajectory
            if self.steps % self.stride == 0 or escaped:
                KE = kinetic(self.vel, self.masses, self.accel_conversion)
                self.trajectory.append(self.positions.copy())
                self.potential_energies.append(self.eng)
                self.kinetic_energies.append(KE)
                self.total_energies.append(self.eng + KE + self.therm)

            if escaped:
                return self.steps * self.timestep

        return None