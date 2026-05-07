# Import your SimulationResults class
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_inspector

from core.object_components import SimulationResults  # adjust this path

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# -------------------------------
# Build PyVista grid from SimulationResults
# -------------------------------
def build_grid_from_class(sim: SimulationResults, time: float):
    if time not in sim.nodes_by_time:
        raise ValueError(f"Time {time} not found in simulation results.")

    nodes = sim.nodes_by_time[time]
    cells = sim.cells_by_time.get(time, None)
    celltypes = sim.celltypes_by_time.get(time, None)

    if cells is None or len(cells) == 0:
        mesh = pv.PolyData(nodes)
    else:
        mesh = pv.UnstructuredGrid(cells, celltypes, nodes)

    # Add node and cell data
    for name, arr in sim.node_data_by_time.get(time, {}).items():
        mesh.point_data[name] = arr
    for name, arr in sim.cell_data_by_time.get(time, {}).items():
        mesh.cell_data[name] = arr

    return mesh

# -------------------------------
#  Plot variable at a given time
# -------------------------------
def plot_variable_at_a_time(sim: SimulationResults, var_name, time, cmap="viridis", show_edges=False, scale=(1,1,1)):
    # Build the grid from your tests_simulation_components class
    grid = build_grid_from_class(sim, time)

    # Check if the variable exists
    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}.")

    # Extract surface for plotting
    surf = grid.extract_geometry()
    surf.points *= scale  # scale the coordinates if needed

    # Create PyVista plotter with axes/grid visible
    p = pv.Plotter()

    # Add the surface with the variable
    p.add_mesh(surf, scalars=var_name, cmap=cmap, show_edges=show_edges)

    # Add scalar bar and text
    p.add_scalar_bar(title=var_name)
    p.add_text(f"{var_name} at time {time}", font_size=20, position="upper_edge")

    # Show axes grid
    p.show_grid()
    p.show()

# -------------------------------
# Plot cross-section slice
# -------------------------------
def plot_cross_section(sim: SimulationResults, var_name, time, origin, normal, cmap="viridis", show_edges=False, scale=(1,1,1)):
    grid = build_grid_from_class(sim, time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}.")

    # Create the cross-section slice
    slice_mesh = grid.slice(origin=origin, normal=normal)
    slice_mesh.points *= scale

    p = pv.Plotter()

    # If the slice has no cells, render points as spheres
    render_kwargs = {}
    if isinstance(slice_mesh, pv.PolyData) and slice_mesh.n_cells == 0:
        render_kwargs = {"render_points_as_spheres": True, "point_size": 6}

    # Add the slice mesh
    p.add_mesh(slice_mesh, scalars=var_name, cmap=cmap, show_edges=show_edges, **render_kwargs)

    # Add scalar bar and text
    p.add_scalar_bar(title=var_name, vertical=False)
    p.add_text(f"{var_name} cross-section at time {time}", font_size=20, position="upper_edge")

    # Show axes grid
    p.show_grid()

    # Show plot
    p.show()

# -------------------------------
# Plot variable along a line
# -------------------------------
def plot_variable_along_line(sim: SimulationResults, var_name, time, p0, p1, n_samples=200):
    grid = build_grid_from_class(sim, time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}.")

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
# Print variable at a point
# -------------------------------
def print_variable_at_point(sim: SimulationResults, var_name, time, point):
    grid = build_grid_from_class(sim, time)

    if var_name not in grid.point_data and var_name not in grid.cell_data:
        raise ValueError(f"Variable '{var_name}' not found at time {time}.")

    pt = pv.PolyData(np.array([point]))
    sampled = pt.sample(grid)

    if sampled.n_points == 0:
        print(f"⚠️ Point {point} is outside mesh bounds {grid.bounds}")
        return None

    value = sampled[var_name][0]
    print(f"{var_name} at point {point} at time {time}: {value}")
    return value

# -------------------------------
# Plot time series at a point
# -------------------------------
def plot_variable_time_series(sim: SimulationResults, var_name, point, decimals=3):
    times = sorted(sim.nodes_by_time.keys())
    values = []

    for t in times:
        grid = build_grid_from_class(sim, t)
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


@wbgeo_component(identifier='wbgeo::inspect_sim_plot_variable_at_a_time',
                 title='Plot Variable p at a time',
                 description='...')
@wbgeo_inspector()
async def vis_plot_variable_at_a_time(
    sim: SimulationResults
):
  # todo: inspect windows currently do not offer inputs. Discuss if we need/want them
  plot_variable_at_a_time(sim, var_name='p', time=0, cmap='coolwarm', scale=(1, 1, 1))


@wbgeo_component(identifier='wbgeo::inspect_sim_plot_cross_section',
                 title='Plot Variable T along a cross_section',
                 description='...')
@wbgeo_inspector()
async def vis_plot_cross_section(
    sim: SimulationResults
):
  # todo: inspect windows currently do not offer inputs. Discuss if we need/want them
  plot_cross_section(sim,"T", 0, origin=(540,20,100), normal=(1,0,0))



@wbgeo_component(identifier='wbgeo::inspect_sim_plot_variable_along_line',
                 title='Plot Variable p along a line',
                 description='...')
@wbgeo_inspector()
async def vis_plot_variable_along_line(
    sim: SimulationResults
):
  # todo: inspect windows currently do not offer inputs. Discuss if we need/want them
  plot_variable_along_line(sim,"p", 0, p0=(500,20,50), p1=(500,20,1000))


