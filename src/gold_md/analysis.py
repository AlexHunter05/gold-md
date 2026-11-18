import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

# Time axis (fs)
def get_time(simulation):
    return simulation.energy_times


def get_trajectory(simulation):
    trajectory = read(simulation.trajectory_file, index=":")
    return np.array([atoms.get_positions() for atoms in trajectory])


def moving_average(data, window_size=10):
    return np.array([np.mean(data[i:i + window_size], axis=0) for i in range(len(data) - window_size + 1)])


def load_simulation(filename, stride=10, timestep=1.0, number_of_simulations=None):
    path = Path(filename)

    if path.is_dir():
        files = sorted(path.glob("simulation_*.xyz"))

        if number_of_simulations is not None:
            files = files[:number_of_simulations]

        return [load_simulation(file, stride, timestep) for file in files]

    positions = np.array([atoms.get_positions() for atoms in read(filename, index=":")])
    simulation = SimpleNamespace()
    simulation.natoms = positions.shape[1]
    simulation.trajectory_file = filename
    simulation.stride = stride
    simulation.timestep = timestep
    simulation.escape_threshold = 1.1 * np.min(np.linalg.norm(positions[0, 1:] - positions[0, 0], axis=1))

    energy_file = path.parent / "Energies" / f"{path.stem}.npz"

    with np.load(energy_file) as energies:
        simulation.kinetic_energies = energies["KE"].copy()
        simulation.potential_energies = energies["PE"].copy()
        simulation.energy_times = energies["time"].copy()

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
    PE = np.array(simulation.potential_energies)
    PE = PE - PE[0]
    PE = moving_average(PE, window_size)
    time = time[:len(PE)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, PE)
    plt.xlabel("Time (fs)")
    plt.ylabel("Change in Potential Energy (eV)")
    plt.title(f"Potential Energy for {simulation.natoms}-atom Gold Cluster")
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
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance from centre (Å)")
    plt.title("Central Atom Displacement")
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


def plot_nearest_distance(simulation, window_size=10, escape_distance=5.5):
    trajectory = get_trajectory(simulation)
    time = np.arange(len(trajectory)) * simulation.stride * simulation.timestep
    separations = np.linalg.norm(trajectory[:, 1:, :] - trajectory[:, 0:1, :], axis=2)
    nearest = np.min(separations, axis=1)
    nearest = moving_average(nearest, window_size)
    time = time[:len(nearest)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, nearest, label="Smoothed distance to nearest atom")
    plt.axhline(y=escape_distance, color="red", linestyle="--", label="Escape distance")
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance to nearest atom (Å)")
    plt.title("Central Atom Distance to Nearest Atom")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_energy_data(simulations, align=None, fixed=None, force_off=None):
    results = None

    if align is not None or fixed is not None or force_off is not None:
        results = pd.read_csv("Outputs/Simulation results.csv")

        results["fixed"] = results["fixed"].astype(str).str.lower().map({"true": True, "false": False})
        results["force_off"] = results["force_off"].astype(str).str.lower().map({"true": True, "false": False})

        if fixed is not None:
            results = results[results["fixed"] == fixed]

        if force_off is not None:
            results = results[results["force_off"] == force_off]

    aligned = []

    for sim in simulations:
        time = get_time(sim)
        seed = int(Path(sim.trajectory_file).stem.split("_")[1])
        PE = np.array(sim.potential_energies) - sim.potential_energies[0]

        if align is not None:
            row = results[(results["seed"] == seed) & (results["natoms"] == sim.natoms)]

            if len(row) != 1:
                print(f"Skipping simulation {seed}: found {len(row)} matching CSV rows")
                continue

            event_time = pd.to_numeric(row.iloc[0][align], errors="coerce")

            if pd.isna(event_time):
                print(f"Skipping simulation {seed}: no {align} recorded")
                continue

            time = time - event_time

        aligned.append((time, np.array(sim.kinetic_energies), PE))

    print(f"Prepared {len(aligned)} simulations.")
    return aligned


def plot_energy_spaghetti(simulations, align=None, fixed=None, force_off=None):
    aligned = prepare_energy_data(simulations, align, fixed, force_off)
    colours = plt.cm.turbo(np.linspace(0, 1, len(aligned)))
    alignment_label = align.replace("_", " ").title() if align else "Simulation Start"

    # Kinetic Energy
    plt.figure(figsize=(10, 5))
    for i, (time, KE, PE) in enumerate(aligned):
        plt.plot(time, KE, color=colours[i], alpha=0.7)

    if align:
        plt.axvline(0, color="black", linestyle="--", label=alignment_label)
        plt.legend()

    plt.xlabel("Time (fs)" if align is None else f"Time relative to {alignment_label} (fs)")
    plt.ylabel("Kinetic Energy (eV)")
    plt.title(f"Kinetic Energy – Individual Simulations ({alignment_label})")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Potential Energy
    plt.figure(figsize=(10, 5))
    for i, (time, KE, PE) in enumerate(aligned):
        plt.plot(time, PE, color=colours[i], alpha=0.7)

    if align:
        plt.axvline(0, color="black", linestyle="--", label=alignment_label)
        plt.legend()

    plt.xlabel("Time (fs)" if align is None else f"Time relative to {alignment_label} (fs)")
    plt.ylabel("Change in Potential Energy (eV)")
    plt.title(f"Potential Energy – Individual Simulations ({alignment_label})")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_mean_energy(simulations, align=None, fixed=None, force_off=None):
    aligned = prepare_energy_data(simulations, align, fixed, force_off)

    if not aligned:
        print("No simulations available to plot.")
        return

    dt = aligned[0][0][1] - aligned[0][0][0]
    min_time = min(time[0] for time, KE, PE in aligned)
    max_time = max(time[-1] for time, KE, PE in aligned)
    time = np.arange(min_time, max_time + 0.5 * dt, dt)

    KE = np.array([np.interp(time, t, k, left=np.nan, right=np.nan) for t, k, p in aligned])
    PE = np.array([np.interp(time, t, p, left=np.nan, right=np.nan) for t, k, p in aligned])

    count_KE = np.sum(~np.isnan(KE), axis=0)
    count_PE = np.sum(~np.isnan(PE), axis=0)

    mean_KE = np.nanmean(KE, axis=0)
    mean_PE = np.nanmean(PE, axis=0)

    std_KE = np.nanstd(KE, axis=0, ddof=1)
    std_PE = np.nanstd(PE, axis=0, ddof=1)

    std_KE[count_KE < 2] = np.nan
    std_PE[count_PE < 2] = np.nan

    alignment_label = align.replace("_", " ").title() if align else "Simulation Start"

    # Mean Kinetic Energy
    plt.figure(figsize=(10, 5))
    plt.plot(time, mean_KE, color="blue", label="Mean KE")
    plt.fill_between(time, mean_KE - std_KE, mean_KE + std_KE, color="blue", alpha=0.2, label="±1 SD")

    if align:
        plt.axvline(0, color="black", linestyle="--", label=alignment_label)
        plt.legend()

    plt.xlabel("Time (fs)" if align is None else f"Time relative to {alignment_label} (fs)")
    plt.ylabel("Kinetic Energy (eV)")
    plt.title(f"Mean Kinetic Energy ({alignment_label})")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Mean Potential Energy
    plt.figure(figsize=(10, 5))
    plt.plot(time, mean_PE, color="red", label="Mean PE")
    plt.fill_between(time, mean_PE - std_PE, mean_PE + std_PE, color="red", alpha=0.2, label="±1 SD")

    if align:
        plt.axvline(0, color="black", linestyle="--", label=alignment_label)
        plt.legend()

    plt.xlabel("Time (fs)" if align is None else f"Time relative to {alignment_label} (fs)")
    plt.ylabel("Change in Potential Energy (eV)")
    plt.title(f"Mean Potential Energy ({alignment_label})")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()