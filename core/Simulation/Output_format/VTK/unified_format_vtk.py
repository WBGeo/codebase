import os
import shutil
import pyvista as pv
import numpy as np
import re
import typing
from core.Simulation.Simulation_packages.Sfepy.simulation_run import SfepyOutputType
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from core.object_components import SimulationResults





# -------------------------------------------------
# SAFE TIME EXTRACTOR
# -------------------------------------------------
def extract_time(filename: str):
    nums = re.findall(r"\d+\.?\d*", filename)
    return float(nums[-1]) if nums else None


# =====================================================
# LOAD VTK RESULTS (WBGeo + SAFE + CONSISTENT)
# =====================================================
@wbgeo_component(
    description='load VTK results',
    title='Load VTK results',
    color="#e5d016",
    border_color='#000000',
    group='Simulation',
    identifier='Load_VTK_results',
    return_name='Loaded VTK results',
)
def load_vtk_results(sim_output: SfepyOutputType) -> SimulationResults:

    # -------------------------------------------------
    # ACCESS DICT (NEW STRUCTURE)
    # -------------------------------------------------
    output_dir = sim_output["output_dir"]

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
        # enforce numeric time keys (OLD BEHAVIOR)
        # -------------------------------------------------
        if time is None:
            continue

        time = float(time)

        # -------------------------------------------------
        # NODE DATA
        # -------------------------------------------------
        results.nodes_by_time[time] = mesh.points.copy()

        # -------------------------------------------------
        # SAFE CELL HANDLING (PyVista robust)
        # -------------------------------------------------
        if hasattr(mesh, "cells") and mesh.cells is not None:
            try:
                cells = mesh.cells.copy()
            except:
                cells = np.array(mesh.cells)
        elif hasattr(mesh, "faces") and mesh.faces is not None:
            try:
                cells = mesh.faces.copy()
            except:
                cells = np.array(mesh.faces)
        else:
            cells = None

        results.cells_by_time[time] = cells

        # -------------------------------------------------
        # CELL TYPES (SAFE)
        # -------------------------------------------------
        results.celltypes_by_time[time] = getattr(mesh, "celltypes", None)

        # -------------------------------------------------
        # POINT DATA
        # -------------------------------------------------
        results.node_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.point_data.items()
        }

        # -------------------------------------------------
        # CELL DATA
        # -------------------------------------------------
        results.cell_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.cell_data.items()
        }

    # -------------------------------------------------
    # CLEANUP POLICY
    # -------------------------------------------------
    if sim_output.get("is_temp", False):
        try:
            shutil.rmtree(output_dir)
            print(f"[INFO] Removed temp output: {output_dir}")
        except Exception as e:
            print(f"[WARNING] cleanup failed: {e}")
    else:
        print(f"[INFO] Kept user output: {output_dir}")

    return results
