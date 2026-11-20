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
        self.initial_positions = self.positions.copy()
        self.natoms = len(self.positions)
        self.masses = np.full(self.natoms, self.mass)
        self.moving_atoms = np.ones(self.natoms)

        # Non-conservative force settings
        self.K = 1.0
        self.centre = self.positions[0].copy()
        self.non_conservative_atoms = np.zeros(self.natoms)
        self.force_mode = "always_off"

        # Event and energy records
        self.shell_exit_times = {}
        self.first_return_time = None
        self.escape_time = None
        self.escape_kinetic_energy = None
        self.energy_times = []
        self.potential_energies = []
        self.kinetic_energies = []
        self.total_energies = []

        # Detection thresholds
        self.initial_distances = np.linalg.norm(self.positions[1:] - self.positions[0], axis=1)
        self.first_shell_radius = np.min(self.initial_distances)
        self.shell_exit_thresholds = self.get_shell_exit_thresholds()
        self.escape_threshold = self.cutoff

        # Initial thermal velocities
        self.rng = np.random.default_rng(seed)
        velocity_sigma = np.sqrt(self.kB * self.temperature * self.accel_conversion / self.masses[0])
        self.vel = self.rng.normal(0, velocity_sigma, size=(self.natoms, 3))
        self.vel -= np.mean(self.vel, axis=0)

        # Langevin thermostat coefficients
        delta = self.timestep / 2
        self.decay = np.exp(-self.friction * delta)
        self.random_factor = np.sqrt(self.kB * self.temperature * self.accel_conversion / self.masses[0] * (1 - np.exp(-2 * self.friction * delta)))

        self.therm = 0.0
        self.calculate_forces()

    def get_shell_exit_thresholds(self):
        sorted_distances = np.sort(self.initial_distances)
        thresholds = {}
        start = 0
        shell = 2

        while start < len(sorted_distances):
            atom_count = 10 * (shell - 1)**2 + 2
            shell_distances = sorted_distances[start:start + atom_count]

            if len(shell_distances) != atom_count:
                raise ValueError("The atomic configuration does not contain complete icosahedral shells.")

            thresholds[shell] = 1.1 * np.min(shell_distances)
            start += atom_count
            shell += 1

        return thresholds

    def fix_atoms(self, atoms):
        self.moving_atoms[atoms] = 0
        self.vel[atoms] = 0

    def set_force_mode(self, force_mode):
        valid_modes = ["always_on", "always_off", "off_after_shell_2"]

        if force_mode not in valid_modes:
            raise ValueError(f"Invalid force_mode '{force_mode}'. Choose from {valid_modes}.")

        self.force_mode = force_mode
        self.calculate_forces()

    def apply_non_conservative_force(self, atoms, force_mode="always_on"):
        self.non_conservative_atoms[atoms] = 1
        self.set_force_mode(force_mode)

    def force_is_active(self):
        if self.force_mode == "always_on":
            return True
        if self.force_mode == "off_after_shell_2":
            return 2 not in self.shell_exit_times
        return False

    def non_conservative_force(self, K):
        centre_of_mass = np.average(self.positions[1:], axis=0, weights=self.masses[1:])
        dx = self.positions[:, 0] - centre_of_mass[0]
        dy = self.positions[:, 1] - centre_of_mass[1]
        forces = np.zeros((self.natoms, 3))
        forces[:, 0] = -K * dy
        forces[:, 1] = K * dx
        return forces * self.non_conservative_atoms[:, None]

    def calculate_forces(self):
        self.eng, self.forces = potential(self.positions, self.cutoff, self.F_spline, self.density_spline, self.rphi_spline, self.F_derivative, self.density_derivative, self.rphi_derivative)

        if self.force_is_active() and np.any(self.non_conservative_atoms):
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

        xyz_file = None
        self.trajectory_file = None
        last_saved_step = None

        if trajectory_file is not None:
            self.trajectory_file = f"{trajectory_file}.xyz"
            os.makedirs(os.path.dirname(self.trajectory_file) or ".", exist_ok=True)
            xyz_file = open(self.trajectory_file, "w")
            self.write_xyz_frame(xyz_file)
            last_saved_step = self.steps

        # Update thermostat coefficients using the current settings
        delta = self.timestep / 2
        self.decay = np.exp(-self.friction * delta)
        self.random_factor = np.sqrt(self.kB * self.temperature * self.accel_conversion / self.masses[0] * (1 - np.exp(-2 * self.friction * delta)))

        # Reset event records for this run
        self.shell_exit_times = {}
        self.first_return_time = None
        self.escape_time = None
        self.escape_kinetic_energy = None
        self.outcome = "unresolved"
        self.calculate_forces()

        escaped = False

        # Moving average of central atom position for escape detection
        window_size = 10
        position_buffer = np.tile(self.positions[0], (window_size, 1))
        position_sum = np.sum(position_buffer, axis=0)
        buffer_index = 0

        # Return detection: remain within half the first-shell radius for 200 fs
        return_radius = 0.5 * self.first_shell_radius
        return_duration = 200.0
        radial_buffer = np.zeros(window_size)
        radial_sum = 0.0
        radial_buffer_index = 0
        return_counter = 0
        return_start_time = None

        try:
            for _ in range(nsteps):
                # Langevin thermostat half-step
                if self.ensemble == "NVT":
                    self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
                    self.vel = (self.vel * self.decay + self.random_factor * self.rng.normal(size=(self.natoms, 3))) * self.moving_atoms[:, None]
                    self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

                # Velocity Verlet algorithm
                self.vel += 0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep * self.moving_atoms[:, None]
                self.positions += self.vel * self.timestep * self.moving_atoms[:, None]

                # Detect shell exits
                if until_escape:
                    radial_distance = np.linalg.norm(self.positions[0] - self.centre)

                    for shell, threshold in self.shell_exit_thresholds.items():
                        if shell not in self.shell_exit_times and radial_distance > threshold:
                            exit_time = (self.steps + 1) * self.timestep
                            self.shell_exit_times[shell] = exit_time

                            if shell == 2:
                                radial_buffer.fill(radial_distance)
                                radial_sum = np.sum(radial_buffer)
                                print(f"Shell 2 exited at {exit_time} fs.")

                self.calculate_forces()
                self.vel += 0.5 * self.accel_conversion * self.forces / self.masses[:, None] * self.timestep * self.moving_atoms[:, None]

                # Langevin thermostat second half-step
                if self.ensemble == "NVT":
                    self.therm += kinetic(self.vel, self.masses, self.accel_conversion)
                    self.vel = (self.vel * self.decay + self.random_factor * self.rng.normal(size=(self.natoms, 3))) * self.moving_atoms[:, None]
                    self.therm -= kinetic(self.vel, self.masses, self.accel_conversion)

                self.steps += 1

                # Update moving average of central atom position
                position_sum -= position_buffer[buffer_index]
                position_buffer[buffer_index] = self.positions[0]
                position_sum += position_buffer[buffer_index]
                buffer_index = (buffer_index + 1) % window_size
                average_position = position_sum / window_size

                if until_escape and 2 in self.shell_exit_times:
                    # Escape: central atom is beyond the EAM cutoff from every other atom
                    distances = np.linalg.norm(self.positions[1:] - average_position, axis=1)
                    escaped = np.all(distances > self.escape_threshold)

                    # Return: moving-average radial distance stays below the return radius
                    if self.first_return_time is None:
                        radial_distance = np.linalg.norm(self.positions[0] - self.centre)
                        radial_sum += radial_distance - radial_buffer[radial_buffer_index]
                        radial_buffer[radial_buffer_index] = radial_distance
                        radial_buffer_index = (radial_buffer_index + 1) % window_size
                        average_radial_distance = radial_sum / window_size

                        if average_radial_distance < return_radius:
                            if return_counter == 0:
                                return_start_time = self.steps * self.timestep
                            return_counter += 1

                            if return_counter * self.timestep >= return_duration:
                                self.first_return_time = return_start_time
                                print(f"Central atom returned at {self.first_return_time} fs.")
                        else:
                            return_counter = 0
                            return_start_time = None

                # Record energies and trajectory
                if self.steps % self.stride == 0 or escaped:
                    KE = kinetic(self.vel, self.masses, self.accel_conversion)
                    self.potential_energies.append(self.eng)
                    self.kinetic_energies.append(KE)
                    self.total_energies.append(self.eng + KE + self.therm)
                    self.energy_times.append(self.steps * self.timestep)

                    if escaped:
                        self.escape_kinetic_energy = kinetic(self.vel[:1], self.masses[:1], self.accel_conversion)

                    if xyz_file is not None and self.steps != last_saved_step:
                        self.write_xyz_frame(xyz_file)
                        last_saved_step = self.steps

                if escaped:
                    break

            if xyz_file is not None and self.steps != last_saved_step:
                self.write_xyz_frame(xyz_file)

        finally:
            if xyz_file is not None:
                xyz_file.close()
                print(f"Saved trajectory to {self.trajectory_file}")

        if escaped:
            self.outcome = "escaped"
            self.escape_time = self.steps * self.timestep
        elif self.first_return_time is not None:
            self.outcome = "returned"

        return self.escape_time