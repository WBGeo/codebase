import os
import shutil
import pyvista as pv
import numpy as np
import re

from py_api_wbgeo.nodesapi import wbgeo_component
from core.object_components import SimulationResults
from core.Simulation.Simulation_packages.Sfepy.simulation_run import SfepySimulationOutput


# -------------------------------------------------
# SAFE TIME EXTRACTOR (UNCHANGED)
# -------------------------------------------------
def extract_time(filename: str):
    nums = re.findall(r"\d+\.?\d*", filename)
    return float(nums[-1]) if nums else None


# =====================================================
# LOAD VTK RESULTS (FIXED + CONSISTENT CONTRACT)
# =====================================================
@wbgeo_component(
    description='load VTK results',
    title='Load VTK results',
    color='#cc9999',
    border_color='#000000',
    group='Meshing',
    identifier='Load_VTK_results',
    return_name='Loaded VTK results',
)
def load_vtk_results(sim_output: SfepySimulationOutput) -> SimulationResults:

    output_dir = sim_output.output_dir

    vtk_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".vtk"):
                vtk_files.append(os.path.join(root, f))

    vtk_files.sort()

    print(f"[INFO] Found {len(vtk_files)} VTK files")

    results = SimulationResults()

    for file_path in vtk_files:
        mesh = pv.read(file_path)
        vtk_file = os.path.basename(file_path)

        time = extract_time(vtk_file)

        # -------------------------------------------------
        # enforce old behavior: float time keys only
        # -------------------------------------------------
        if time is None:
            continue

        time = float(time)

        # -------------------------------------------------
        # NODE DATA (UNCHANGED)
        # -------------------------------------------------
        results.nodes_by_time[time] = mesh.points.copy()

        # -------------------------------------------------
        # FIXED CELL ACCESS (IMPORTANT)
        # PyVista compatibility: NOT all datasets have .cells
        # -------------------------------------------------
        if hasattr(mesh, "cells") and mesh.cells is not None:
            cells = mesh.cells.copy()
        else:
            # fallback for PolyData
            cells = mesh.faces.copy() if hasattr(mesh, "faces") else None

        results.cells_by_time[time] = cells

        # -------------------------------------------------
        # CELL TYPES (SAFE)
        # -------------------------------------------------
        results.celltypes_by_time[time] = getattr(mesh, "celltypes", None)

        # -------------------------------------------------
        # POINT DATA (UNCHANGED)
        # -------------------------------------------------
        results.node_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.point_data.items()
        }

        # -------------------------------------------------
        # CELL DATA (UNCHANGED)
        # -------------------------------------------------
        results.cell_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.cell_data.items()
        }

    # -------------------------------------------------
    # CLEANUP POLICY (UNCHANGED)
    # -------------------------------------------------
    if sim_output.is_temp:
        try:
            shutil.rmtree(output_dir)
            print(f"[INFO] Removed temp output: {output_dir}")
        except Exception as e:
            print(f"[WARNING] cleanup failed: {e}")
    else:
        print(f"[INFO] Kept user output: {output_dir}")

    return results
