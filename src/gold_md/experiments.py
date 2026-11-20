import numpy as np
import os
import csv
import shutil
from pathlib import Path
from datetime import datetime, timezone
from .simulation import AMDC


def run_escape_time(potential, positions, output_folder, seed, fixed, force_mode="always_on", max_steps=None):
    simulation = AMDC(potential, positions, "NVE", seed=seed)
    natoms = len(positions)

    if fixed:
        simulation.fix_atoms(range(1, natoms))

    simulation.apply_non_conservative_force(0, force_mode=force_mode)

    trajectory_file = f"{output_folder}/simulation_{seed}"

    if max_steps is None:
        escape_time = simulation.md(trajectory_file=trajectory_file, until_escape=True)
    else:
        escape_time = simulation.md(trajectory_file=trajectory_file, until_escape=True, max_steps=max_steps)

    energy_folder = Path(output_folder) / "Energies"
    energy_folder.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(energy_folder / f"simulation_{seed}.npz", KE=simulation.kinetic_energies, PE=simulation.potential_energies, time=simulation.energy_times)

    return simulation, escape_time


def repeat_simulations(potential, positions, output_folder, nruns, fixed, force_mode="always_on", max_steps=None):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(nruns):
        # Find lowest available simulation number
        sim_number = 1
        while os.path.exists(f"{output_folder}/simulation_{sim_number}.xyz"):
            sim_number += 1

        current_seed = sim_number
        simulation, escape_time = run_escape_time(potential, positions, output_folder, current_seed, fixed, force_mode=force_mode, max_steps=max_steps)
        save_result(simulation, len(positions), fixed, current_seed, escape_time, max_steps)

        print(f"Run {i+1}/{nruns} complete,   Seed = {current_seed},   Escape time = {escape_time} fs,   {simulation.outcome},   Force mode = {force_mode}.\n")

    return simulation


def save_result(simulation, natoms, fixed, seed, escape_time, max_steps):
    results_file = Path("Outputs/Simulation results.csv")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    shells = {13: 2, 55: 3, 147: 4, 309: 5, 561: 6, 923: 7, 1415 : 8}

    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "shells": shells.get(natoms),
        "natoms": natoms,
        "fixed": fixed,
        "force_mode": simulation.force_mode,
        "outcome": simulation.outcome,
        "shell_2_exit_time": simulation.shell_exit_times.get(2),
        "shell_3_exit_time": simulation.shell_exit_times.get(3),
        "shell_4_exit_time": simulation.shell_exit_times.get(4),
        "shell_5_exit_time": simulation.shell_exit_times.get(5),
        "shell_6_exit_time": simulation.shell_exit_times.get(6),
        "first_return_time": simulation.first_return_time,
        "escape_time": escape_time,
        "escape_kinetic_energy": round(simulation.escape_kinetic_energy, 3) if simulation.escape_kinetic_energy is not None else None,
        "max_steps": max_steps,
        "temperature": simulation.temperature,
        "force_constant": simulation.K,
        "ensemble": simulation.ensemble
    }

    file_exists = results_file.exists() and results_file.stat().st_size > 0

    with open(results_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)