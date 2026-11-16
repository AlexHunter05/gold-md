import numpy as np
import os
from .simulation import AMDC
import csv
from pathlib import Path
from datetime import datetime, timezone



def run_escape_time(potential, positions, output_folder, seed, fixed, turn_force_off=False, max_steps=None):
    simulation = AMDC(potential, positions, "NVE", seed=seed)
    natoms = len(positions)

    if fixed:
        simulation.fix_atoms(range(1, natoms))
    simulation.apply_non_conservative_force(0)

    if turn_force_off:
        simulation.disable_force_after_shell_exit()

    trajectory_file = f"{output_folder}/simulation_{seed}"

    if max_steps is None:
        escape_time = simulation.md(trajectory_file=trajectory_file, until_escape=True)
    else:
        escape_time = simulation.md(trajectory_file=trajectory_file, until_escape=True, max_steps=max_steps)

    return simulation, escape_time


def repeat_simulations(potential, positions, output_folder, nruns, fixed, max_steps=None, turn_force_off=False):

    os.makedirs(f"{output_folder}", exist_ok=True)

    for i in range(nruns):
        # Find lowest available simulation number
        sim_number = 1
        while os.path.exists(f"{output_folder}/simulation_{sim_number}.xyz"):
            sim_number += 1

        current_seed = sim_number
        simulation, escape_time = run_escape_time(potential, positions, output_folder, current_seed, fixed, turn_force_off=turn_force_off, max_steps=max_steps)
        save_result(simulation, len(positions), fixed, turn_force_off, current_seed, escape_time, max_steps)
        print(f"Run {i+1}/{nruns} complete,   Seed = {current_seed},   Escape time = {escape_time} fs,   {simulation.outcome}.\n")

    return simulation


def save_result(simulation, natoms, fixed, turn_force_off, seed, escape_time, max_steps):
    results_file = Path("Outputs/Simulation results.csv")

    shells = {13: 2, 55: 3, 147: 4, 309: 5, 561: 6, 923: 7}

    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "shells": shells.get(natoms),
        "natoms": natoms,
        "fixed": fixed,
        "force_off": turn_force_off,
        "outcome": simulation.outcome,
        "first_shell_exit_time": simulation.first_shell_exit_time,
        "first_return_time": simulation.first_return_time,
        "escape_time": escape_time,
        "max_steps": max_steps,
        "temperature": simulation.temperature,
        "force_constant": simulation.K,
        "ensemble": simulation.ensemble
    }

    fieldnames = list(result.keys())
    rows = []

    if results_file.exists() and results_file.stat().st_size > 0:
        with open(results_file, "r", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = list(reader)

    rows = [
        row for row in rows
        if not (
            int(row["seed"]) == seed
            and int(row["natoms"]) == natoms
            and row["fixed"].lower() == str(fixed).lower()
            and row["force_off"].lower() == str(turn_force_off).lower()
        )
    ]

    rows.append(result)

    with open(results_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

import pandas as pd
def rerun_old_returns(potential, positions, output_folder, shells, fixed, turn_force_off=False, max_steps=None):

    results_file = Path("Outputs/Simulation results.csv")
    results = pd.read_csv(results_file)

    # Identify old simulations incorrectly classified as returned
    old_false = results["outcome"].astype(str).str.lower() == "false"
    first_shell_exited = pd.to_numeric(results["first_shell_exit_time"], errors="coerce").notna()

    candidates = results[
        old_false
        & first_shell_exited
        & (results["shells"] == shells)
        & (results["fixed"] == fixed)
        & (results["force_off"] == turn_force_off)
    ]

    if candidates.empty:
        print(f"No simulations to rerun for {shells} shells, fixed={fixed}, force_off={turn_force_off}.")
        return

    os.makedirs(output_folder, exist_ok=True)

    seeds = candidates["seed"].tolist()

    print(f"Rerunning {len(seeds)} simulations for {shells} shells.")
    print(f"Fixed = {fixed}, Force turns off = {turn_force_off}\n")

    for i, seed in enumerate(seeds):
        simulation, escape_time = run_escape_time(
            potential, positions, output_folder, seed, fixed,
            turn_force_off=turn_force_off, max_steps=max_steps
        )

        save_result(simulation, len(positions), fixed, turn_force_off, seed, escape_time, max_steps)

        print(f"Run {i+1}/{len(seeds)} complete, Seed = {seed}, Escape time = {escape_time} fs, Outcome = {simulation.outcome}.\n")

    return simulation