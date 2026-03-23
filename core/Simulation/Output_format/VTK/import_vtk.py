import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter




# -------------------------------
# 1️⃣ Load all VTK files
# -------------------------------
vtk_dir = "../../../../out1/"

if not os.path.isdir(vtk_dir):
    raise FileNotFoundError(f"Directory not found: {vtk_dir}")

vtk_files = sorted(f for f in os.listdir(vtk_dir) if f.lower().endswith(".vtk"))
if len(vtk_files) == 0:
    raise RuntimeError("No VTK files found.")

data_by_time = {}

for vtk_file in vtk_files:
    file_path = os.path.join(vtk_dir, vtk_file)

    # Extract time from filename (last number before .vtk)
    try:
        time_part = vtk_file.split('.')[-2]
        time = float(time_part)
    except (ValueError, IndexError):
        time = vtk_file

    print(f"Loading {vtk_file} (time={time})")
    mesh = pv.read(file_path)

    data_by_time[time] = {
        "mesh": mesh,
        "node_data": {name: np.array(arr) for name, arr in mesh.point_data.items()},
        "cell_data": {name: np.array(arr) for name, arr in mesh.cell_data.items()}
    }

print("\nAll VTK files loaded.")

# -------------------------------
# 2️⃣ Build grid for plotting
# -------------------------------
def build_grid(time):
    """
    Returns a PyVista mesh (PolyData or UnstructuredGrid) ready for plotting.
    """
    data = data_by_time[time]
    mesh = data["mesh"]

    # If mesh has no cells, treat as PolyData point cloud
    if mesh.n_cells == 0:
        poly = pv.PolyData(mesh.points)
        for name, arr in mesh.point_data.items():
            poly[name] = arr
        return poly
    else:
        # Use the original mesh as-is (UnstructuredGrid)
        return mesh.copy()



# -------------------------------
# 3️⃣ Plot a variable
# -------------------------------
def plot_variable_at_a_time(var_name, time, cmap="viridis", show_edges=False, scale=(1,1,1)):
    grid = build_grid(time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found in mesh at time {time}.")

    # Extract surface for plotting tetrahedral/volumetric mesh
    surf = grid.extract_geometry()
    surf.points *= scale

    p = pv.Plotter()
    p.add_mesh(
        surf,
        scalars=var_name,
        cmap=cmap,
        show_edges=show_edges
    )
    p.add_scalar_bar(title=var_name)
    p.add_text(f"{var_name} at time {time}", font_size=24, position="upper_edge")
    p.show()

# -------------------------------
# 4️⃣ Plot a cross-section slice
# -------------------------------
def plot_cross_section_slice(var_name, time, origin, normal, cmap="viridis", show_edges=False, scale=(1,1,1)):
    grid = build_grid(time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    slice_mesh = grid.slice(origin=origin, normal=normal)
    slice_mesh.points *= scale

    p = pv.Plotter()
    render_kwargs = {}
    if isinstance(slice_mesh, pv.PolyData) and slice_mesh.n_cells == 0:
        render_kwargs = {"render_points_as_spheres": True, "point_size": 6}

    p.add_mesh(slice_mesh, scalars=var_name, cmap=cmap, show_edges=show_edges, **render_kwargs)
    p.add_scalar_bar(title=var_name, vertical=False)
    p.add_text(f"{var_name} cross-section at time {time}", font_size=20, position="upper_edge")
    p.show()

# -------------------------------
# 5️⃣ Plot variable along a line
# -------------------------------
def plot_variable_along_line(var_name, time, p0, p1, n_samples=200):
    grid = build_grid(time)
    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    line = pv.Line(p0, p1, resolution=n_samples)
    sampled = line.sample(grid)

    if sampled.n_points == 0:
        raise RuntimeError("Line sampling produced no points – check p0/p1")

    values = sampled[var_name]
    distances = sampled["Distance"]

    plt.figure(figsize=(7,4))
    plt.plot(distances, values, "-k")
    plt.xlabel("Distance along line")
    plt.ylabel(var_name)
    plt.title(f"{var_name} along line at time {time}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6️⃣ Value at a point
# -------------------------------
def print_variable_at_point(var_name, time, point):
    grid = build_grid(time)
    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    pt = pv.PolyData(np.array([point]))
    sampled = pt.sample(grid)

    if sampled.n_points == 0:
        print(f"⚠️ Point {point} is outside mesh bounds {grid.bounds}")
        return None

    value = sampled[var_name][0]
    print(f"{var_name} at point {point} at time {time}: {value}")
    return value

# -------------------------------
# 7️⃣ Time series at a point
# -------------------------------
def plot_variable_time_series(var_name, point, decimals=3):
    times = sorted(data_by_time.keys())
    values = []

    for t in times:
        grid = build_grid(t)
        if var_name not in grid.point_data and var_name not in grid.cell_data:
            raise ValueError(f"Variable '{var_name}' not found at time {t}")
        pt = pv.PolyData(np.array([point]))
        sampled = pt.sample(grid)
        if sampled.n_points == 0:
            values.append(np.nan)
        else:
            values.append(sampled[var_name][0])

    values = np.array(values)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(times, values, '-o', color='tab:red')
    ax.set_xlabel("Time")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name} at point {point} over time")
    ax.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
    plt.tight_layout()
    plt.show()

# -------------------------------
# ✅ Examples
# -------------------------------
time0 = 0
plot_variable_at_a_time("T", time0, cmap="coolwarm", scale=(1,1,1))
plot_cross_section_slice("p", time0, origin=(540,20,100), normal=(1,0,0))
plot_variable_along_line("p", time0, p0=(21500,20,0), p1=(500,20,400))
print_variable_at_point("p", time0, point=(500,20,500))
plot_variable_time_series("p", point=(500,20,500))

# --------------------------------------------------
# Sanity check of keys
# --------------------------------------------------
for t in sorted(data_by_time.keys()):
    print(f"\n⏱ Time {t}")
    print("  Node data keys:", list(data_by_time[t]["node_data"].keys()))
    print("  Cell data keys:", list(data_by_time[t]["cell_data"].keys()))


# Example: Plot ObjectId at time = 420
plot_variable_at_a_time("T", 0, cmap="tab20", scale=(1,1,1))
# Example: Plot temperature at time = 420
plot_variable_at_a_time("p", 0, cmap="coolwarm", scale=(1,1,1))

plot_cross_section_slice(
    var_name="p",
    time=0,
    origin=(540, 20, 100),
    normal=(0, 0, 1),
    cmap="coolwarm", scale=(1,1,1)
)

plot_variable_along_line(
    var_name="p",
    time=0,
    p0=(21500, 20, 0),
    p1=(500, 20, 400),
    n_samples=300
)

# Temperature at a specific point
print_variable_at_point(
    var_name="p",
    time=0,
    point=(500, 20, 500)
)

# Pore pressure at a specific point
print_variable_at_point(
    var_name="T",
    time=0,
    point=(500, 20, 500)
)


# Temperature time series at a point
plot_variable_time_series(
    var_name="p",
    point=(200, 20, 500)
)

# Pore pressure time series at a point
plot_variable_time_series(
    var_name="p",
    point=(500, 20, 500)
)
