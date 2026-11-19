import logging
import os
import shutil
import numpy as np
import pyvista as pv

from typing import Generator, Union
from core.object_components import SimulationResults
from py_api_wbgeo.nodesapi import wbgeo_component
from vtkmodules.vtkIOExodus import vtkExodusIIReader
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

logger = logging.getLogger(__name__)


##################
# LOAD EXODUS FILE
#################


def load_exodus_results(sim_input: Union[str, dict]) -> SimulationResults:
  # =====================================================
  # INPUT NORMALIZATION
  # Accept:
  #   - direct Exodus file path
  #   - directory containing Exodus file
  #   - GOLEM output dictionary
  # =====================================================

  is_temp = False

  if isinstance(sim_input, str):

    file_path = sim_input


  elif isinstance(sim_input, dict):

    is_temp = sim_input.get(
      "is_temp",
      False
    )

    if "output_file" in sim_input:

      file_path = sim_input["output_file"]

    elif "output_dir" in sim_input:

      file_path = sim_input["output_dir"]

    elif "mesh_file" in sim_input:

      file_path = sim_input["mesh_file"]

    else:

      raise ValueError(
        "Dictionary input must contain "
        "'output_file', 'output_dir', or 'mesh_file'"
      )


  else:

    raise TypeError(
      "sim_input must be either a string path or dictionary"
    )

  file_path = os.path.abspath(file_path)

  logger.debug("Loading GOLEM Exodus results from %s", file_path)

  # =====================================================
  # IF DIRECTORY IS GIVEN, FIND EXODUS FILE
  # =====================================================

  if os.path.isdir(file_path):

    logger.info( "Searching Exodus file inside directory: %s", file_path )

    exodus_files = [
      f for f in os.listdir(file_path)
      if f.lower().endswith(
        (".exo", ".e", ".ex2")
      )
    ]

    if len(exodus_files) == 0:
      raise FileNotFoundError(
        f"No Exodus file found in directory: {file_path}"
      )

    if len(exodus_files) > 1:
      logger.warning( "Multiple Exodus files found. " "Using first one: %s", exodus_files[0] )

    file_path = os.path.join(
      file_path,
      exodus_files[0]
    )

  # =====================================================
  # FINAL FILE CHECK
  # =====================================================

  if not os.path.isfile(file_path):
    raise FileNotFoundError(
      f"Exodus file not found: {file_path}"
    )

  logger.info( "Reading Exodus file: %s", file_path )

  # =====================================================
  # EXODUS READER SETUP
  # =====================================================

  reader = vtkExodusIIReader()

  reader.SetFileName( file_path )

  reader.UpdateInformation()

  # ENABLE RESULT ARRAYS

  for i in range(
      reader.GetNumberOfPointResultArrays()
  ):
    reader.SetPointResultArrayStatus( reader.GetPointResultArrayName(i), 1 )

  for i in range(
      reader.GetNumberOfElementResultArrays()
  ):
    reader.SetElementResultArrayStatus( reader.GetElementResultArrayName(i), 1 )

  reader.SetGenerateGlobalNodeIdArray(True)

  reader.SetGenerateGlobalElementIdArray(True)

  reader.SetGenerateObjectIdCellArray(True)

  # Element block names by block id (the id ObjectId holds per cell).
  # Unnamed blocks are reported as "Unnamed block ID: ..." and left out.
  elem_block = vtkExodusIIReader.ELEM_BLOCK
  block_names = {}
  for i in range( reader.GetNumberOfObjects(elem_block) ):
    name = reader.GetObjectName(elem_block, i) or ""
    if name and not name.startswith("Unnamed block"):
      block_names[int(reader.GetObjectId(elem_block, i))] = name

  # =====================================================
  # TIME STEPS
  # =====================================================

  info = reader.GetOutputInformation(0)

  time_steps = list(
    info.Get( vtkStreamingDemandDrivenPipeline.TIME_STEPS() )
  )

  logger.info( "Found %d time steps", len(time_steps) )

  results = SimulationResults(block_names=block_names)

  # =====================================================
  # LOOP OVER TIME STEPS
  # =====================================================

  for t in time_steps:

    logger.debug( "Processing time %s", t )

    info.Set( vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(), t )

    reader.Update()

    mb = pv.wrap(
      reader.GetOutput()
    )

    leaf_meshes = list(
      iter_leaf_blocks(mb)
    )

    leaf_meshes = [
      m for m in leaf_meshes
      if m is not None
         and getattr(m, "n_points", 0) > 0
    ]

    if len(leaf_meshes) == 0:
      logger.warning( "No valid leaf meshes at time %s", t )

      continue

    # The Exodus reader gives every element block its own copy of the
    # nodes it uses; merge them so nodes shared between blocks (lithology
    # interfaces, fault planes) appear once, as in the file.
    combined_mesh = pv.MultiBlock(
      leaf_meshes
    ).combine(merge_points=True)

    if combined_mesh is None or not hasattr(
        combined_mesh,
        "points"
    ):
      logger.warning( "Skipping empty mesh at time %s", t )

      continue

    time = float(t)

    # STORE RESULTS

    results.nodes_by_time[time] = (
      combined_mesh.points.copy()
    )

    results.cells_by_time[time] = (
      combined_mesh.cells.copy()
    )

    results.celltypes_by_time[time] = (
      combined_mesh.celltypes.copy()
    )

    results.node_data_by_time[time] = {
      k: np.array(v)
      for k, v in combined_mesh.point_data.items()
    }

    results.cell_data_by_time[time] = {
      k: np.array(v)
      for k, v in combined_mesh.cell_data.items()
    }

    logger.debug(
      "Time %s — node data keys: %s, cell data keys: %s",
      time,
      list(combined_mesh.point_data.keys()),
      list(combined_mesh.cell_data.keys())
    )

  logger.info( "Exodus file loaded successfully." )

  # =====================================================
  # CLEANUP
  # =====================================================

  if is_temp:

    try:

      shutil.rmtree( os.path.dirname(file_path) )

      logger.info( "Removed temp directory" )

    except Exception as e:

      logger.warning( "Cleanup failed: %s", e )

  else:

    logger.info( "Kept user file: %s", file_path )

  return results


#########
# HELPER
#########

def iter_leaf_blocks(
    block: pv.DataSet
) -> Generator[pv.DataSet, None, None]:
  if isinstance(block, pv.MultiBlock):

    for child in block:

      if child is not None:
        yield from iter_leaf_blocks(child)

  else:

    yield block
