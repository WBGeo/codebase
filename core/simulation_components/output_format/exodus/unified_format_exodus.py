import os
import shutil
import numpy as np
import pyvista as pv
from typing import Union
from core.object_components import SimulationResults
from py_api_wbgeo.nodesapi import wbgeo_component
from vtkmodules.vtkIOExodus import vtkExodusIIReader
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline


##################
# LOAD EXODUS FILE
#################
@wbgeo_component(
    description='load single Exodus results file (time-aware)',
    title='Load Exodus file',
    color="#e5d016",
    border_color='#000000',
    group='simulation_components',
    identifier='wbgeo::load_exodus_file',
    return_name='results',
)
def load_exodus_results(sim_input: Union[str, dict]) -> SimulationResults:

    # INPUT NORMALIZATION
    if isinstance(sim_input, str):
        file_path = sim_input
        is_temp = False
    else:
        file_path = sim_input["output_dir"]
        is_temp = sim_input.get("is_temp", False)

    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Exodus file not found: {file_path}")

    print(f"[INFO] Reading Exodus file: {file_path}")

    # EXODUS READER SETUP
    reader = vtkExodusIIReader()
    reader.SetFileName(file_path)
    reader.UpdateInformation()

    # ENABLE RESULT ARRAYS (CRITICAL)
    for i in range(reader.GetNumberOfPointResultArrays()):
        reader.SetPointResultArrayStatus(
            reader.GetPointResultArrayName(i), 1
        )

    for i in range(reader.GetNumberOfElementResultArrays()):
        reader.SetElementResultArrayStatus(
            reader.GetElementResultArrayName(i), 1
        )

    reader.SetGenerateGlobalNodeIdArray(True)
    reader.SetGenerateGlobalElementIdArray(True)
    reader.SetGenerateObjectIdCellArray(True)

    # TIME STEPS
    info = reader.GetOutputInformation(0)
    time_steps = list(info.Get(vtkStreamingDemandDrivenPipeline.TIME_STEPS()))

    print(f"[INFO] Found {len(time_steps)} time steps")

    results = SimulationResults()

    # LOOP OVER TIME STEPS (MATCH YOUR WORKING SCRIPT)
    for t in time_steps:

        print(f"Processing time {t}")

        info.Set(
            vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(), t
        )
        reader.Update()

        mb = pv.wrap(reader.GetOutput())

        # MATCH WORKING LEAF EXTRACTION
        leaf_meshes = list(iter_leaf_blocks(mb))
        leaf_meshes = [m for m in leaf_meshes if m is not None and getattr(m, "n_points", 0) > 0]

        if len(leaf_meshes) == 0:
            print(f"[WARNING] No valid leaf meshes at time {t}")
            continue

        combined_mesh = pv.MultiBlock(leaf_meshes).combine()

        if combined_mesh is None or not hasattr(combined_mesh, "points"):
            print(f"[WARNING] Skipping empty mesh at time {t}")
            continue

        time = float(t)

        # STORE RESULTS
        results.nodes_by_time[time] = combined_mesh.points.copy()

        results.cells_by_time[time] = combined_mesh.cells.copy()

        results.celltypes_by_time[time] = combined_mesh.celltypes.copy()

        results.node_data_by_time[time] = {
            k: np.array(v) for k, v in combined_mesh.point_data.items()
        }

        results.cell_data_by_time[time] = {
            k: np.array(v) for k, v in combined_mesh.cell_data.items()
        }

        # Debug (same style as yours)
        print(f"\n⏱ Time {time}")
        print(f"  Node data keys: {list(combined_mesh.point_data.keys())}")
        print(f"  Cell data keys: {list(combined_mesh.cell_data.keys())}")

    print("[INFO] Exodus file loaded successfully.")

    # CLEANUP
    if not isinstance(sim_input, str) and is_temp:
        try:
            shutil.rmtree(os.path.dirname(file_path))
            print("[INFO] Removed temp directory")
        except Exception as e:
            print(f"[WARNING] cleanup failed: {e}")
    else:
        print(f"[INFO] Kept user file: {file_path}")

    return results


#########
# HELPER
#########
def iter_leaf_blocks(block):
    if isinstance(block, pv.MultiBlock):
        for child in block:
            if child is not None:
                yield from iter_leaf_blocks(child)
    else:
        yield block
