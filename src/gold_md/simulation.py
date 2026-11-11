import os
import numpy as np
from .load_eam import load_eam
from .potential import potential, kinetic

class AMDC:
    def __init__(self, eam_file, positions, ensemble="NVT", seed=42):
        # Load EAM potential
        self.mass, self.cutoff, self.F_spline, self.density_spline, self.rphi_spline, self.F_derivative, self.density_derivative, self.rphi_derivative = load_eam(eam_file)
        
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

        # Compute escape threshold based on initial configuration
        self.initial_distances = np.linalg.norm(self.positions[1:] - self.positions[0], axis=1)
        self.escape_threshold = 1.1 * np.min(self.initial_distances)

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

        self.potential_energies = []
        self.kinetic_energies = []
        self.total_energies = []

    def fix_atoms(self, atoms):
        self.moving_atoms[atoms] = 0
        self.vel[atoms] = 0

    def apply_non_conservative_force(self, atoms):
        self.non_conservative_atoms[atoms] = 1
        self.calculate_forces()

    def non_conservative_force(self, K):
        # Calculate centre of mass, excluding central atom
        centre_of_mass = np.average(self.positions[1:], axis=0, weights=self.masses[1:])

        x0, y0, z0 = centre_of_mass

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
            self.rphi_spline,
            self.F_derivative,
            self.density_derivative,
            self.rphi_derivative)
        
        if np.any(self.non_conservative_atoms):
            self.forces += self.non_conservative_force(self.K)

    def write_xyz_frame(self, file):
        file.write(f"{self.natoms}\n")
        file.write(f"Step {self.steps}, Time {self.steps * self.timestep:.2f} fs\n")
        for position in self.positions:
            file.write(f"Au {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n")

    def md(self, nsteps=None, trajectory_file=None, until_escape=False, max_steps=100000):

        if until_escape:
            nsteps = max_steps
        elif nsteps is None:
            raise ValueError("Specify number of steps or set until_escape=True")

        # Open trajectory file if a filename is provided
        xyz_file = None
        self.trajectory_file = None
        last_saved_step = None

        if trajectory_file is not None:
            print(f"Saving trajectory to {trajectory_file}\n")
            trajectory_file = f"{trajectory_file}.xyz"
            print(f"Saving trajectory to {trajectory_file}\n")
            os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)
            self.trajectory_file = trajectory_file
            xyz_file = open(trajectory_file, "w")
            self.write_xyz_frame(xyz_file)
            last_saved_step = self.steps

        escaped = False

        try:
            # Velocity Verlet algorithm
            for _ in range(nsteps):
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

                # Check whether the central atom has escaped
                if until_escape:
                    distances = np.linalg.norm(self.positions[1:] - self.positions[0], axis=1)
                    escaped = np.all(distances > self.escape_threshold)

                # Record energies and trajectory
                if self.steps % self.stride == 0 or escaped:
                    KE = kinetic(self.vel, self.masses, self.accel_conversion)
                    self.potential_energies.append(self.eng)
                    self.kinetic_energies.append(KE)
                    self.total_energies.append(self.eng + KE + self.therm)

                    if xyz_file is not None and self.steps != last_saved_step:
                        self.write_xyz_frame(xyz_file)
                        last_saved_step = self.steps

                if escaped:
                    break

            # Record final configuration if it has not already been saved
            if xyz_file is not None and self.steps != last_saved_step:
                self.write_xyz_frame(xyz_file)

        finally:
            if xyz_file is not None:
                xyz_file.close()

        if escaped:
            return self.steps * self.timestep

        return None