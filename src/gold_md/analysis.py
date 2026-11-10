import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Time axis (fs)
def get_time(simulation):
    print("\n")
    return np.arange(len(simulation.total_energies)) * simulation.stride * simulation.timestep


def get_trajectory(simulation):
    trajectory = read(simulation.trajectory_file, index=":")
    return np.array([atoms.get_positions() for atoms in trajectory])


def plot_energy_conservation(simulation):
    time = get_time(simulation)
    energy_change = np.array(simulation.total_energies) - simulation.total_energies[0]

    plt.figure(figsize=(10, 6))
    plt.plot(time, energy_change)
    plt.xlabel("Time (fs)")
    plt.ylabel("Change in Conserved Energy (eV)")
    plt.title("Gold EAM Simulation - Energy Conservation")
    plt.grid(True)
    plt.show()


def plot_PE(simulation):
    time = get_time(simulation)

    plt.figure(figsize=(10, 6))
    plt.plot(time, simulation.potential_energies, label="Potential Energy")
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Potential Energy for {simulation.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()


def plot_KE(simulation, equilibration_steps=1000):
    time = get_time(simulation)
    ave_KE = np.mean(np.array(simulation.kinetic_energies)[equilibration_steps // simulation.stride:])
    expected_KE = 1.5 * np.sum(simulation.moving_atoms) * simulation.kB * simulation.temperature

    plt.figure(figsize=(10, 6))
    plt.plot(time, simulation.kinetic_energies, label="Kinetic Energy")
    if simulation.ensemble == "NVT":
        plt.axhline(ave_KE, color="red", linestyle="--", label=f"Average KE = {ave_KE:.3f}")
        plt.axhline(expected_KE, color="green", linestyle="--", label=f"Expected KE = {expected_KE:.3f}")
        plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Kinetic Energy for {simulation.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()


def plot_temp(simulation, equilibration_steps=1000):
    time = get_time(simulation)
    degrees_of_freedom = 3 * np.sum(simulation.moving_atoms)
    temps = (2 * np.array(simulation.kinetic_energies) / (simulation.kB * degrees_of_freedom))
    ave_temps = np.mean(temps[equilibration_steps // simulation.stride:])

    plt.figure(figsize=(10, 6))
    plt.plot(time, temps, label="Temperature")
    if simulation.ensemble == "NVT":
        plt.axhline(ave_temps, color="red", linestyle="--", label=f"Average Temp = {ave_temps:.2f}")
        plt.axhline(simulation.temperature, color="green", linestyle="--", label=f"Target Temp = {simulation.temperature:.2f}")
        plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (Kelvin)")
    plt.title(f"Temperature for {simulation.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()


def plot_displacement(simulation, trajectory_file):
    time = np.arange(len(get_trajectory(trajectory_file))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(trajectory_file)
    centre_positions = trajectory[:, 0, :]
    distances = np.linalg.norm(centre_positions, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(time, distances, label="Central atom displacement")
    plt.axhline(y=simulation.escape_threshold, color="red", linestyle="--", label="Escape threshold")
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance from centre (Å)")
    plt.title("Central Atom Displacement")
    plt.grid(True)
    plt.show()


def plot_displacement_squared(simulation, trajectory_file):
    time = np.arange(len(get_trajectory(trajectory_file))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(trajectory_file)
    centre_positions = trajectory[:, 0, :]
    distances_squared = (np.linalg.norm(centre_positions, axis=1))**2

    plt.figure(figsize=(10, 5))
    plt.plot(time, distances_squared)
    plt.xlabel("Time (fs)")
    plt.ylabel("Squared Displacement (Å²)")
    plt.title("Central Atom Squared Displacement")
    plt.grid(True)
    plt.show()


def plot_azimuthal(simulation, trajectory_file):
    time = np.arange(len(get_trajectory(trajectory_file))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(trajectory_file)
    centre_positions = trajectory[:, 0, :]

    x = centre_positions[:, 0]
    y = centre_positions[:, 1]
    angle = np.arctan2(y, x)
    angle_degrees = np.degrees(angle)

    plt.figure(figsize=(10, 5))
    plt.plot(time, angle_degrees)
    plt.xlabel("Time (fs)")
    plt.ylabel("Azimuthal Angle (degrees)")
    plt.title("Central Atom Azimuthal Angle")
    plt.grid(True)
    plt.show()


def plot_radial_distance(simulation, trajectory_file):
    time = np.arange(len(get_trajectory(trajectory_file))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(trajectory_file)
    centre_positions = trajectory[:, 0, :]

    x = centre_positions[:, 0]
    y = centre_positions[:, 1]
    radial_distance = np.sqrt(x**2 + y**2)

    plt.figure(figsize=(10, 5))
    plt.plot(time, radial_distance)
    plt.xlabel("Time (fs)")
    plt.ylabel("Radial Distance (Å)")
    plt.title("Central Atom Radial Distance")
    plt.grid(True)
    plt.show()


def plot_angular_velocity(simulation, trajectory_file):
    time = np.arange(len(get_trajectory(trajectory_file))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(trajectory_file)
    centre_positions = trajectory[:, 0, :]

    x = centre_positions[:, 0]
    y = centre_positions[:, 1]
    angle = np.unwrap(np.arctan2(y, x))
    angular_velocity = np.gradient(angle, time)
    angular_velocity_degrees = np.degrees(angular_velocity)

    plt.figure(figsize=(10, 5))
    plt.plot(time, angular_velocity_degrees)
    plt.xlabel("Time (fs)")
    plt.ylabel("Angular Velocity (degrees/fs)")
    plt.title("Central Atom Angular Velocity")
    plt.grid(True)
    plt.show()