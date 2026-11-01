def load_eam(filename):
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"EAM potential file not found: {filename}")

    with open(filename, "r") as file:
        lines = file.readlines()

    # First three lines are comments
    header = lines[3].split()
    nelements = int(header[0])

    if nelements != 1 or header[1] != "Au":
        raise ValueError("This code requires a single-element Au EAM file.")

    # Line 5 contains grid information
    grid = lines[4].split()

    nrho = int(grid[0])
    drho = float(grid[1])
    nr = int(grid[2])
    dr = float(grid[3])
    cutoff = float(grid[4])

    # Line 6 contains gold atomic information
    element_info = lines[5].split()

    atomic_number = int(element_info[0])
    mass = float(element_info[1])

    # Read the tabulated data after the six header lines
    data = np.array([
        float(value)
        for line in lines[6:]
        for value in line.split()
    ])

    # Extract the three tabulated functions
    F_values = data[:nrho]
    density_values = data[nrho:nrho + nr]
    rphi_values = data[nrho + nr:nrho + 2 * nr]

    if len(F_values) != nrho or len(density_values) != nr or len(rphi_values) != nr:
        raise ValueError("Unexpected EAM file format or incomplete data.")

    rho_grid = np.arange(nrho) * drho
    r_grid = np.arange(nr) * dr

    F_spline = CubicSpline(rho_grid, F_values)
    density_spline = CubicSpline(r_grid, density_values)
    rphi_spline = CubicSpline(r_grid, rphi_values)

    return mass, cutoff, F_spline, density_spline, rphi_spline
