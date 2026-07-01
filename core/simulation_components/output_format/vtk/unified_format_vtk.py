import logging
import os
import shutil
import pyvista as pv
import numpy as np
import re
from core.simulation_components.simulation_packages.sfepy.simulation_run import SfepyOutputType
from py_api_wbgeo.nodesapi import wbgeo_component
from core.object_components import SimulationResults
from typing import Optional, Union

logger = logging.getLogger(__name__)


####################
# SAFE TIME EXTRACTOR
####################
def extract_time(filename: str) -> Optional[float]:

    match = re.search(r"([\d\.]+)\.vtk$", filename)

    if not match:
        return None

    s = match.group(1)

    parts = s.split('.')

    if len(parts) >= 3:
        return float(parts[-2] + '.' + parts[-1])

    elif len(parts) == 2:
        return int(parts[-1])

    else:
        return int(parts[0])


####################
# LOAD VTK RESULTS
###################
@wbgeo_component(
    description='load VTK results',
    title='Load VTK results',
    color="#e5d016",
    border_color='#000000',
    group='simulation_components',
    identifier='wbgeo::load_VTK_results',
    return_name='results',
)
def load_vtk_results(sim_input: Union[SfepyOutputType, str]) -> SimulationResults:

    # NORMALIZE INPUT (DICT OR DIRECT PATH)
    if isinstance(sim_input, str):
        output_dir = sim_input
        is_temp = False
    else:
        output_dir = sim_input["output_dir"]
        is_temp = sim_input.get("is_temp", False)

    vtk_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".vtk"):
                vtk_files.append(os.path.join(root, f))

    vtk_files.sort()
    logger.info("Found %d VTK files", len(vtk_files))

    results = SimulationResults()

    for file_path in vtk_files:
        mesh = pv.read(file_path)
        vtk_file = os.path.basename(file_path)

        time = extract_time(vtk_file)
        if time is None:
            continue

        time = float(time)

        # NODE DATA
        results.nodes_by_time[time] = mesh.points.copy()

        # CELL HANDLING
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

        # CELL TYPES
        results.celltypes_by_time[time] = getattr(mesh, "celltypes", None)

        # POINT DATA
        results.node_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.point_data.items()
        }

        # CELL DATA
        results.cell_data_by_time[time] = {
            k: np.array(v) for k, v in mesh.cell_data.items()
        }

    # CLEANUP POLICY (only if dict input)
    if not isinstance(sim_input, str) and is_temp:
        try:
            shutil.rmtree(output_dir)
            logger.info("Removed temp output: %s", output_dir)
        except Exception as e:
            logger.warning("Cleanup failed: %s", e)
    else:
        logger.info("Kept user output: %s", output_dir)

    return results

