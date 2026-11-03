import numpy as np
import matplotlib.pyplot as plt

# Time axis (fs)
def get_time(my_md):
    print("\n")
    return np.arange(len(my_md.total_energies)) * my_md.stride * my_md.timestep

def plot_energy_conservation(my_md):
    time = get_time(my_md)
    energy_change = np.array(my_md.total_energies) - my_md.total_energies[0]

    plt.figure(figsize=(10, 6))
    plt.plot(time, energy_change)
    plt.xlabel("Time (fs)")
    plt.ylabel("Change in Conserved Energy (eV)")
    plt.title("Gold EAM Simulation - Energy Conservation")
    plt.grid(True)
    plt.show()

def plot_PE(my_md):
    time = get_time(my_md)

    plt.figure(figsize=(10, 6))
    plt.plot(time, my_md.potential_energies, label="Potential Energy")
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Potential Energy for {my_md.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()

def plot_KE(my_md, equilibration_steps=1000):
    time = get_time(my_md)
    ave_KE = np.mean(np.array(my_md.kinetic_energies)[equilibration_steps // my_md.stride:])
    expected_KE = 1.5 * np.sum(my_md.moving_atoms) * my_md.kB * my_md.temperature

    plt.figure(figsize=(10, 6))
    plt.plot(time, my_md.kinetic_energies, label="Kinetic Energy")
    if my_md.ensemble == "NVT":
        plt.axhline(ave_KE, color="red", linestyle="--", label=f"Average KE = {ave_KE:.3f}")
        plt.axhline(expected_KE, color="green", linestyle="--", label=f"Expected KE = {expected_KE:.3f}")
        plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Kinetic Energy for {my_md.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()

def plot_temp(my_md, equilibration_steps=1000):
    time = get_time(my_md)
    degrees_of_freedom = 3 * np.sum(my_md.moving_atoms)
    temps = (2 * np.array(my_md.kinetic_energies) / (my_md.kB * degrees_of_freedom))
    ave_temps = np.mean(temps[equilibration_steps // my_md.stride:])

    plt.figure(figsize=(10, 6))
    plt.plot(time, temps, label="Temperature")
    if my_md.ensemble == "NVT":
        plt.axhline(ave_temps, color="red", linestyle="--", label=f"Average Temp = {ave_temps:.2f}")
        plt.axhline(my_md.temperature, color="green", linestyle="--", label=f"Target Temp = {my_md.temperature:.2f}")
        plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (Kelvin)")
    plt.title(f"Temperature for {my_md.natoms}-atom Gold Cluster")
    plt.grid(True)
    plt.show()

def plot_displacement(my_md):
    time = get_time(my_md)
    trajectory = np.array(my_md.trajectory)
    centre_positions = trajectory[:, 0, :]
    distances = np.linalg.norm(centre_positions, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(time, distances)
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance from centre (Å)")
    plt.title("Central Atom Displacement")
    plt.grid(True)
    plt.show()
