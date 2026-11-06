import numpy as np
import os
from .simulation import AMDC
from .trajectory import save_xyz


def run_escape_time(potential,positions, output_folder, seed, max_steps=None):
    my_md = AMDC(potential, positions, "NVE", seed=seed)
    natoms = len(positions)

    my_md.fix_atoms(range(1, natoms))
    my_md.apply_non_conservative_force(0)

    if max_steps is None:
        escape_time = my_md.md(until_escape=True)
    else:
        escape_time = my_md.md(until_escape=True, max_steps=max_steps)

    os.makedirs(output_folder, exist_ok=True)
    save_xyz(my_md, f"{output_folder}/simulation_{seed}")

    return escape_time


def repeat_simulations(potential, positions, output_folder, nruns, max_steps=None):
    escape_times = []

    for seed in range(1, nruns + 1):
        escape_time = run_escape_time(potential, positions, output_folder, seed, max_steps)
        escape_times.append(escape_time)
        print(f"Run {seed}/{nruns} complete. Escape time = {escape_time} fs")

    return np.array(escape_times)