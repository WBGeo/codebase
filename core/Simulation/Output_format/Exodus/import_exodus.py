import os
import numpy as np
import pyvista as pv
from vtkmodules.vtkIOExodus import vtkExodusIIReader
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
import logging

# Suppress non-fatal VTK warnings
logging.getLogger('vtkmodules').setLevel(logging.ERROR)

# iterate over leaf blocks
def iter_leaf_blocks(block):
    """Recursively yield all leaf meshes in a (possibly nested) MultiBlock"""
    if isinstance(block, pv.MultiBlock):
        for child in block:
            if child is not None:
                yield from iter_leaf_blocks(child)
    else:
        yield block

# File path and reader setup
#filename = "../Output_examples/unstructured.exo"
filename = "../Output_examples/transient_rect_out.e"
reader = vtkExodusIIReader()
reader.SetFileName(filename)
reader.UpdateInformation()

# Enable all nodal and cell variables
for i in range(reader.GetNumberOfPointResultArrays()):
    reader.SetPointResultArrayStatus(reader.GetPointResultArrayName(i), 1)
for i in range(reader.GetNumberOfElementResultArrays()):
    reader.SetElementResultArrayStatus(reader.GetElementResultArrayName(i), 1)

# Generate IDs
reader.SetGenerateGlobalNodeIdArray(True)
reader.SetGenerateGlobalElementIdArray(True)
reader.SetGenerateObjectIdCellArray(True)

# Get available timesteps
info = reader.GetOutputInformation(0)
time_steps = list(info.Get(vtkStreamingDemandDrivenPipeline.TIME_STEPS()))
print("Available time steps:", time_steps)

# Containers for cached data
data_by_time = {}

# Loop over timesteps
for t in time_steps:
    print(f"Processing time {t}")
    info.Set(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(), t)
    reader.Update()

    mb = pv.wrap(reader.GetOutput())

    # Collect all non-empty leaf meshes
    leaf_meshes = [leaf for leaf in iter_leaf_blocks(mb) if leaf is not None and leaf.n_points > 0]

    if len(leaf_meshes) == 0:
        raise ValueError(f"No valid leaf meshes found at time {t}")

    # Combine leaf meshes safely
    combined_mesh = pv.MultiBlock(leaf_meshes).combine()

    # Store data
    node_data = {name: np.array(arr) for name, arr in combined_mesh.point_data.items()}
    cell_data = {name: np.array(arr) for name, arr in combined_mesh.cell_data.items()}

    data_by_time[t] = {
        "points": combined_mesh.points.copy(),
        "cells": combined_mesh.cells.copy(),
        "celltypes": combined_mesh.celltypes.copy(),
        "node_data": node_data,
        "cell_data": cell_data
    }

print("Data cached successfully for all timesteps")

# --------------------------------------------------
# Sanity check of keys
# --------------------------------------------------
for t in sorted(data_by_time.keys()):
    print(f"\n⏱ Time {t}")
    print("  Node data keys:", list(data_by_time[t]["node_data"].keys()))
    print("  Cell data keys:", list(data_by_time[t]["cell_data"].keys()))





# Example1: get pore_pressure at t=420 and plot
t1 = 32400.0
# get points and pore_pressure from cached data
points = data_by_time[t1]["points"]
pp = data_by_time[t1]["node_data"]["pore_pressure"]
# create a PyVista point cloud
cloud = pv.PolyData(points)
# attach the scalar data
cloud["pore_pressure"] = pp
# plot
cloud.plot(
    scalars="pore_pressure",
    cmap="plasma",
    point_size=3,
    render_points_as_spheres=True,
    show_scalar_bar=True
)


# Example2: print temperature at t=32400.0
t2 = 32400.0
# get temperature nodal values
temperature = data_by_time[t2]["node_data"]["temp"]
# print some info
print(f"Temperature at time {t2}:")
print("Shape:", temperature.shape)
print("First 10 values:", temperature[:10])



# Example3: print temperature at t=32400.0
t3 = 32400.0
# get temperature nodal values
Ob= data_by_time[t3]["cell_data"]["ObjectId"]
# print some info
print(f"ObjectId at time {t3}:")
print("Shape:", Ob.shape)
print("First 10 values:", Ob[:10])



# Function: Reconstruct grid for a given timestep
def build_grid(time):
    data = data_by_time[time]
    grid = pv.UnstructuredGrid(
        data["cells"],
        data["celltypes"],
        data["points"]
    )

    # Attach node and cell data
    for name, arr in data["node_data"].items():
        grid.point_data[name] = arr
    for name, arr in data["cell_data"].items():
        grid.cell_data[name] = arr

    return grid

# Function: Plot any variable at a given timestep
def plot_variable_at_a_time(var_name, time, cmap="viridis", show_edges=False):
    grid = build_grid(time)

    if var_name in grid.point_data:
        scalars_type = "point"
    elif var_name in grid.cell_data:
        scalars_type = "cell"
    else:
        raise ValueError(f"Variable '{var_name}' not found in grid at time {time}.")

    grid.plot(
        scalars=var_name,
        cmap=cmap,
        show_edges=show_edges
    )

def plot_cross_section_slice(
    var_name,
    time,
    origin,
    normal,
    cmap="viridis",
    show_edges=False
):
    """
    Plot a variable on a planar cross-section.

    Parameters
    ----------
    var_name : str
        Name of variable to plot (point or cell data)
    time : float
        Time step
    origin : tuple (x, y, z)
        A point on the slicing plane
    normal : tuple (nx, ny, nz)
        Normal vector of the slicing plane
    """

    grid = build_grid(time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    # Perform slicing
    slice_mesh = grid.slice(origin=origin, normal=normal)

    if slice_mesh.n_points == 0:
        raise RuntimeError("Slice produced no points – check origin/normal")

    slice_mesh.plot(
        scalars=var_name,
        cmap=cmap,
        show_edges=show_edges
    )


def plot_variable_along_line(
    var_name,
    time,
    p0,
    p1,
    n_samples=200
):
    """
    Plot a variable sampled along a line.

    Parameters
    ----------
    var_name : str
        Name of variable (point or cell data)
    time : float
        Time step
    p0, p1 : tuple (x, y, z)
        Start and end points of the line
    n_samples : int
        Number of sampling points
    """

    grid = build_grid(time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    # Create line
    line = pv.Line(p0, p1, resolution=n_samples)

    # Sample grid onto line
    sampled = line.sample(grid)

    if sampled.n_points == 0:
        raise RuntimeError("Line sampling produced no points – check p0/p1")

    values = sampled[var_name]
    distances = sampled["Distance"]

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(distances, values, "-k")
    plt.xlabel("Distance along line")
    plt.ylabel(var_name)
    plt.title(f"{var_name} along line (t = {time})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_variable_at_point(
    var_name,
    time,
    point
):
    """
    Get and plot (or print) the value of a variable at a specific point in space.

    Parameters
    ----------
    var_name : str
        Name of the variable (point or cell data)
    time : float
        Time step
    point : tuple (x, y, z)
        3D coordinates of the location
    """

    grid = build_grid(time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}")

    # Create a tiny point cloud at the location
    pt = pv.PolyData(np.array([point]))

    # Sample the variable at this point
    sampled = pt.sample(grid)

    if sampled.n_points == 0:
        print(f"⚠️ Point {point} is outside the mesh bounds {grid.bounds}")
        return None

    value = sampled[var_name][0]  # sampled returns an array, only one point

    # Print or return
    print(f"{var_name} at point {point} at time {time}: {value}")
    return value

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_variable_time_series(var_name, point, decimals=3):
    """
    Plot the value of a variable at a specific point over all cached timesteps,
    with axis tick labels formatted to a fixed number of decimals.

    Parameters
    ----------
    var_name : str
        Name of variable (point or cell data)
    point : tuple (x, y, z)
        3D coordinates of the location
    decimals : int
        Number of decimal places for Y-axis tick labels
    """

    times = sorted(data_by_time.keys())
    values = []

    for t in times:
        grid = build_grid(t)

        if var_name not in grid.point_data and var_name not in grid.cell_data:
            raise ValueError(f"Variable '{var_name}' not found at time {t}")

        # Sample at the point
        pt = pv.PolyData(np.array([point]))
        sampled = pt.sample(grid)

        if sampled.n_points == 0:
            print(f"⚠️ Point {point} is outside the mesh at time {t}, appending NaN")
            values.append(np.nan)
        else:
            values.append(sampled[var_name][0])

    values = np.array(values)

    # Plot time series
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, values, '-o', color='tab:red')
    ax.set_xlabel("time")               # X-axis label
    ax.set_ylabel(var_name)          # Y-axis label
    ax.set_title(f"{var_name} at point {point} over time")
    ax.grid(True)

    # Force float format on both axes
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))

    plt.tight_layout()
    plt.show()

    return times, values


# --------------------------------------------------
# Sanity check of keys
# --------------------------------------------------
for t in sorted(data_by_time.keys()):
    print(f"\n⏱ Time {t}")
    print("  Node data keys:", list(data_by_time[t]["node_data"].keys()))
    print("  Cell data keys:", list(data_by_time[t]["cell_data"].keys()))


# Example: Plot ObjectId at time = 420
plot_variable_at_a_time("ObjectId", 32400.0, cmap="tab20")
# Example: Plot temperature at time = 420
plot_variable_at_a_time("temp",32400.0, cmap="coolwarm")

plot_cross_section_slice(
    var_name="temp",
    time=30600.0,
    origin=(-12.7194375992,	204.407546997,	-4139.53271484),
    normal=(0, 0, 1),
    cmap="coolwarm"
)

plot_variable_along_line(
    var_name="temp",
    time=30600.0,
    p0=(884, 184, -4090),
    p1=(884, 184, -3895),
    n_samples=300
)

# Temperature at a specific point
print_variable_at_point(
    var_name="temp",
    time=30600.0,
    point=(-12.7194375992,	204.407546997,	-4139.53271484)
)

# Pore pressure at a specific point
print_variable_at_point(
    var_name="pore_pressure",
    time=30600.0,
    point=(884, 184, -4090)
)


# Temperature time series at a point
plot_variable_time_series(
    var_name="temp",
    point=(-12.7194375992,	204.407546997,	-4139.53271484)
)

# Pore pressure time series at a point
plot_variable_time_series(
    var_name="pore_pressure",
    point=(-12.7194375992,	204.407546997,	-4139.53271484)
)
