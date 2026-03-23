
import os
import shutil
import pyvista as pv
from core.object_components import SimulationResults
import numpy as np
import tempfile

def load_exodus_results(output_dir) -> SimulationResults:
    output_dir = os.path.abspath(output_dir)  # ensure absolute path

    exo_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".exo") or f.endswith(".e"):
                exo_files.append(os.path.join(root, f))

    exo_files.sort()
    print(f"Found {len(exo_files)} Exodus files in {output_dir}")

    results = SimulationResults()

    for file_path in exo_files:
        print(f"Reading Exodus file: {file_path}")

        mesh = pv.read(file_path)

        # Check if time-dependent
        if hasattr(mesh, "n_times") and mesh.n_times > 1:
            for i in range(mesh.n_times):
                mesh.set_active_time_point(i)
                time = mesh.time_values[i]

                results.nodes_by_time[time] = mesh.points.copy()
                results.cells_by_time[time] = mesh.cells.copy()
                results.celltypes_by_time[time] = mesh.celltypes.copy()
                results.node_data_by_time[time] = {
                    k: np.array(v) for k, v in mesh.point_data.items()
                }
                results.cell_data_by_time[time] = {
                    k: np.array(v) for k, v in mesh.cell_data.items()
                }
        else:
            # Single timestep fallback
            time = 0.0

            results.nodes_by_time[time] = mesh.points.copy()
            results.cells_by_time[time] = mesh.cells.copy()
            results.celltypes_by_time[time] = mesh.celltypes.copy()
            results.node_data_by_time[time] = {
                k: np.array(v) for k, v in mesh.point_data.items()
            }
            results.cell_data_by_time[time] = {
                k: np.array(v) for k, v in mesh.cell_data.items()
            }

    print("All Exodus files loaded successfully.")

    # Automatically remove only if the directory is inside the system temporary directory
    temp_dir = tempfile.gettempdir()
    if os.path.commonpath([output_dir, temp_dir]) == temp_dir:
        try:
            shutil.rmtree(output_dir)
            print(f"Removed temporary directory: {output_dir}")
        except Exception as e:
            print(f"Could not remove directory {output_dir}: {e}")
    else:
        print(f"Directory is not temporary, kept: {output_dir}")

    return results
