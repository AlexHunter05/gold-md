import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from types import SimpleNamespace

# Time axis (fs)
def get_time(simulation):
    print("\n")
    return np.arange(len(simulation.total_energies)) * simulation.stride * simulation.timestep


def get_trajectory(simulation):
    trajectory = read(simulation.trajectory_file, index=":")
    return np.array([atoms.get_positions() for atoms in trajectory])


def moving_average(data, window_size=10):
    return np.array([np.mean(data[i:i + window_size], axis=0) for i in range(len(data) - window_size + 1)])


def load_simulation(filename, stride=10, timestep=1.0):
    positions = np.array([atoms.get_positions() for atoms in read(filename, index=":")])
    simulation = SimpleNamespace()
    simulation.trajectory_file = filename
    simulation.stride = stride
    simulation.timestep = timestep
    simulation.escape_threshold = 1.1 * np.min(np.linalg.norm(positions[0, 1:] - positions[0, 0], axis=1))
    return simulation


def plot_energy_conservation(simulation, window_size=10):
    time = get_time(simulation)
    energy_change = np.array(simulation.total_energies) - simulation.total_energies[0]

    energy_change = moving_average(energy_change, window_size)
    time = time[:len(energy_change)]

    plt.figure(figsize=(10, 6))
    plt.plot(time, energy_change, label="Smoothed Energy Change")
    plt.xlabel("Time (fs)")
    plt.ylabel("Change in Conserved Energy (eV)")
    plt.title("Gold EAM Simulation - Energy Conservation")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_PE(simulation, window_size=10):
    time = get_time(simulation)
    PE = moving_average(np.array(simulation.potential_energies), window_size)
    time = time[:len(PE)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, PE, label="Total System Potential Energy")
    plt.xlabel("Time (fs)")
    plt.ylabel("Potential Energy (eV)")
    plt.title(f"Potential Energy for {simulation.natoms}-atom Gold Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_KE(simulation, equilibration_steps=1000, window_size=10):
    time = get_time(simulation)
    KE = moving_average(np.array(simulation.kinetic_energies), window_size)
    time = time[:len(KE)]
    ave_KE = np.mean(KE[equilibration_steps // simulation.stride:])
    expected_KE = 1.5 * np.sum(simulation.moving_atoms) * simulation.kB * simulation.temperature

    plt.figure(figsize=(10, 6))
    plt.plot(time, KE, label="Kinetic Energy")
    if simulation.ensemble == "NVT":
        plt.axhline(ave_KE, color="red", linestyle="--", label=f"Average KE = {ave_KE:.3f}")
        plt.axhline(expected_KE, color="green", linestyle="--", label=f"Expected KE = {expected_KE:.3f}")
        plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Kinetic Energy for {simulation.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()


def plot_temp(simulation, equilibration_steps=1000, window_size=10):
    time = get_time(simulation)
    degrees_of_freedom = 3 * np.sum(simulation.moving_atoms)
    temps = (2 * np.array(simulation.kinetic_energies) / (simulation.kB * degrees_of_freedom))
    temps = moving_average(temps, window_size)
    time = time[:len(temps)]
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


def plot_displacement(simulation, window_size=10):
    time = np.arange(len(get_trajectory(simulation))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(simulation)

    centre_positions = moving_average(trajectory[:, 0, :], window_size)
    distances = np.linalg.norm(centre_positions, axis=1)
    time = time[:len(centre_positions)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, distances, label="Smoothed central atom displacement")
    plt.axhline(y=simulation.escape_threshold, color="red", linestyle="--", label="Escape threshold")
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance from centre (Å)")
    plt.title("Central Atom Displacement")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_displacement_squared(simulation, window_size=10):
    time = np.arange(len(get_trajectory(simulation))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(simulation)

    centre_positions = moving_average(trajectory[:, 0, :], window_size)
    distances_squared = (np.linalg.norm(centre_positions, axis=1))**2
    time = time[:len(centre_positions)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, distances_squared)
    plt.xlabel("Time (fs)")
    plt.ylabel("Squared Displacement (Å²)")
    plt.title("Central Atom Squared Displacement")
    plt.grid(True)
    plt.show()


def plot_radial_distance(simulation, window_size=10):
    time = np.arange(len(get_trajectory(simulation))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(simulation)
    centre_positions = moving_average(trajectory[:, 0, :], window_size)
    time = time[:len(centre_positions)]

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


def plot_azimuthal(simulation, window_size=10):
    time = np.arange(len(get_trajectory(simulation))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(simulation)
    centre_positions = moving_average(trajectory[:, 0, :], window_size)
    time = time[:len(centre_positions)]

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


def plot_angular_velocity(simulation, window_size=10):
    time = np.arange(len(get_trajectory(simulation))) * simulation.stride * simulation.timestep
    trajectory = get_trajectory(simulation)
    centre_positions = moving_average(trajectory[:, 0, :], window_size)
    time = time[:len(centre_positions)]

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