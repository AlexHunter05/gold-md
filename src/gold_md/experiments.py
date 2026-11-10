import numpy as np
import os
from .simulation import AMDC
from .trajectory import save_xyz
import csv
from pathlib import Path
from datetime import datetime, timezone


def run_escape_time(potential, positions, output_folder, seed, fixed, max_steps=None):
    simulation = AMDC(potential, positions, "NVE", seed=seed)
    natoms = len(positions)

    if fixed:
        simulation.fix_atoms(range(1, natoms))
    simulation.apply_non_conservative_force(0)

    sim_number = 1
    while os.path.exists(f"{output_folder}/simulation_{sim_number}.xyz"):
        sim_number += 1

    trajectory_file = f"{output_folder}/simulation_{sim_number}.xyz"

    if max_steps is None:
        escape_time = simulation.md(trajectory_file, until_escape=True)
    else:
        escape_time = simulation.md(trajectory_file, until_escape=True, max_steps=max_steps)

    print(f"Saved trajectory to {trajectory_file}")

    return simulation, escape_time


def repeat_simulations(potential, positions, output_folder, nruns, fixed, max_steps=None):

    os.makedirs(output_folder, exist_ok=True)

    # Find seeds already used
    used_seeds = []
    for filename in os.listdir(output_folder):
        if filename.startswith("simulation_") and filename.endswith(".xyz"):
            seed = int(filename.replace("simulation_", "").replace(".xyz", ""))
            used_seeds.append(seed)

    # Find next available seed
    seed = max(used_seeds, default=0) + 1

    for i in range(nruns):
        current_seed = seed + i
        simulation, escape_time = run_escape_time(potential, positions, output_folder, current_seed, fixed, max_steps)
        save_result(simulation, len(positions), fixed, current_seed, escape_time, max_steps)
        print(f"Run {i+1}/{nruns} complete,     Seed = {current_seed},  Escape time = {escape_time} fs")

    return simulation


def save_result(simulation, natoms, fixed, seed, escape_time, max_steps):
    results_file = Path("Outputs/Simulation results.csv")

    shells = {13: 2, 55: 3, 147: 4, 309: 5, 561: 6}

    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "natoms": natoms,
        "shells": shells.get(natoms),
        "fixed": fixed,
        "seed": seed,
        "escape_time (fs)": escape_time,
        "escaped": escape_time is not None,
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