import os
import shutil
import pyvista as pv
from core.object_components import SimulationResults
import numpy as np
import tempfile

def load_vtk_results(output_dir) -> SimulationResults:
    output_dir = os.path.abspath(output_dir)  # ensure absolute path
    vtk_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".vtk"):
                vtk_files.append(os.path.join(root, f))
    vtk_files.sort()
    print(f"Found {len(vtk_files)} VTK files in {output_dir}")

    results = SimulationResults()

    for file_path in vtk_files:
        vtk_file = os.path.basename(file_path)
        try:
            time = float(vtk_file.split('.')[-2])
        except:
            time = vtk_file

        mesh = pv.read(file_path)

        results.nodes_by_time[time] = mesh.points.copy()
        results.cells_by_time[time] = mesh.cells.copy()
        results.celltypes_by_time[time] = mesh.celltypes.copy()
        results.node_data_by_time[time] = {k: np.array(v) for k, v in mesh.point_data.items()}
        results.cell_data_by_time[time] = {k: np.array(v) for k, v in mesh.cell_data.items()}

    print("All VTK files loaded successfully.")

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
