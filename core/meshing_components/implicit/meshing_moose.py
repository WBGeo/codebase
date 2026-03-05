import pyvista as pv
import os

from core.object_components import StructuralModelResults


def create_mesh_moose(geomodel_results: StructuralModelResults, name: str) -> pv.DataSet:
    """
    Create a mesh for MOOSE based on the input input_data.

    Args:
        geomodel_results (StructuralModelResults): The input input_data for the geological model.

    Returns:
        comp_meshes (pyvista.core.composite.MultiBlock): The set of meshes precomputed by MOOSE
    """
    # Placeholder function: Right now we are just loading a precomputed set of meshes using pyvista
    # TODO: Load correct file automatically
    import os
    import pathlib

    # load exodus file relative to this .py file
    datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    path_to_file=os.path.join(datadir, f'examples/input_data/precomputed_meshes_temp/moose_mesh_input_{name}_in.e')
    # cwd = os.getcwd()
    # path_to_file = cwd + "/examples/input_data/precomputed_meshes_temp/moose_mesh_input_Model_7_UCK_in.e"
    # path_to_file = cwd + f"/examples/input_data/precomputed_meshes_temp/moose_mesh_input_{name}_in.e"
    comp_meshes = pv.read_exodus(path_to_file)

    return comp_meshes
