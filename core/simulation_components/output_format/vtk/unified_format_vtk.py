import logging
import os
import shutil
import pyvista as pv
import numpy as np
import re
from dataclasses import dataclass
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type
from core.object_components import SimulationResults
from typing import Optional, Union

logger = logging.getLogger(__name__)


@wbgeo_type(name='SfepyOutputType', color='pink', identifier='SfepyOutputType')
@dataclass
class SfepyOutputType:
    """
    Where a completed SfePy run's raw VTK output lives, and whether
    load_vtk_results() should delete it after loading -- the shape
    _run_sfepy_input_file() returns and load_vtk_results() consumes.

    stdout: the raw sfepy-run output for this run, so callers of
    _run_sfepy_input_file (not just load_vtk_results, which only cares
    about the VTK files) can carry it forward -- see
    run_simulation_sfepy()'s SimulationResults.sfepy_stdout.
    """
    output_dir: str
    is_temp: bool = False
    stdout: str = ""


####################
# SAFE TIME EXTRACTOR
####################
def extract_time(filename: str) -> Optional[Union[float, int]]:

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


def _geometry_matches(last_geometry, nodes, cells, celltypes) -> bool:
    """
    Whether (nodes, cells, celltypes) are array-equal to the previously
    stored (nodes, cells, celltypes) -- checked explicitly (not assumed)
    so load_vtk_results only reuses the same array objects across
    timesteps when they truly are identical.
    """
    last_nodes, last_cells, last_celltypes = last_geometry

    if last_nodes.shape != nodes.shape or not np.array_equal(last_nodes, nodes):
        return False

    if (last_cells is None) != (cells is None):
        return False
    if cells is not None and (last_cells.shape != cells.shape or not np.array_equal(last_cells, cells)):
        return False

    if (last_celltypes is None) != (celltypes is None):
        return False
    if celltypes is not None and not np.array_equal(last_celltypes, celltypes):
        return False

    return True


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

    # NORMALIZE INPUT (SfepyOutputType OR DIRECT PATH)
    if isinstance(sim_input, str):
        output_dir = sim_input
        is_temp = False
    else:
        output_dir = sim_input.output_dir
        is_temp = sim_input.is_temp

    vtk_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".vtk"):
                vtk_files.append(os.path.join(root, f))

    vtk_files.sort()
    logger.info("Found %d VTK files", len(vtk_files))

    results = SimulationResults()

    # This pipeline solves one fixed FEM mesh throughout a run -- SfePy's
    # own per-step VTK files redundantly repeat the same geometry every
    # time (only field VALUES actually change). last_geometry tracks the
    # most recently stored (nodes, cells, celltypes) so an unchanged step
    # can reuse the same array objects instead of allocating a fresh
    # duplicate copy of the whole mesh per timestep -- for a real run with
    # many saved steps, that's an Nx reduction in this object's memory
    # footprint. Checked via _geometry_matches, not assumed, so a
    # hypothetical future remeshing/deforming solve still gets its own
    # distinct geometry stored correctly per step.
    last_geometry = None

    for file_path in vtk_files:
        mesh = pv.read(file_path)
        vtk_file = os.path.basename(file_path)

        time = extract_time(vtk_file)
        if time is None:
            continue

        time = float(time)

        # CELL HANDLING (raw, undecided yet whether this step needs its own copy)
        if hasattr(mesh, "cells") and mesh.cells is not None:
            raw_cells = mesh.cells
        elif hasattr(mesh, "faces") and mesh.faces is not None:
            raw_cells = mesh.faces
        else:
            raw_cells = None
        raw_celltypes = getattr(mesh, "celltypes", None)

        if last_geometry is not None and _geometry_matches(last_geometry, mesh.points, raw_cells, raw_celltypes):
            nodes, cells, celltypes = last_geometry
        else:
            nodes = mesh.points.copy()
            if raw_cells is not None:
                try:
                    cells = raw_cells.copy()
                except Exception:
                    cells = np.array(raw_cells)
            else:
                cells = None
            celltypes = raw_celltypes
            last_geometry = (nodes, cells, celltypes)

        results.nodes_by_time[time] = nodes
        results.cells_by_time[time] = cells
        results.celltypes_by_time[time] = celltypes

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

