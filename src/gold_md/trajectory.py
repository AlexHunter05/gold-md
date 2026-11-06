# Save the trajectory in XYZ format for VMD
def save_xyz(simulation, filename):
    with open(f"{filename}.xyz", "w") as file:
        for frame in simulation.trajectory:
            file.write(f"{simulation.natoms}\n\n")
            for x, y, z in frame:
                file.write(f"Au {x:.8f} {y:.8f} {z:.8f}\n")
    print(f"Trajectory saved as \"{filename}.xyz\"")