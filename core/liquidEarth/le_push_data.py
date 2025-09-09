import liquid_earth_sdk as le
import subsurface as ss
import numpy as np
import pandas as pd
from core.object_components import InputData, GeomodelResults
from py_api_wbgeo.nodesapi import wbgeo_component
from py_api_wbgeo import apitypes



def convert_to_subsurface_mesh(geosolution):
    # Note: surface_meshes_vertices and surface_meshes_edges are List[List[NpNDArrayFp64/Int64]]
    vertex_groups: List[List[np.ndarray]] = geosolution.surface_meshes_vertices
    simplex_groups: List[List[np.ndarray]] = geosolution.surface_meshes_edges

    # Flatten the nested lists to get individual arrays
    vertex = []
    simplex_list = []
    
    for vertex_group in vertex_groups:
        vertex.extend(vertex_group)
    
    for simplex_group in simplex_groups:
        simplex_list.extend(simplex_group)

    idx_max = 0
    for simplex_array in simplex_list:
        # Ensure we're working with numpy arrays
        if isinstance(simplex_array, np.ndarray) and simplex_array.size > 0:
            simplex_array += idx_max
            idx_max = simplex_array.max() + 1

    # Create ID arrays
    vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(vertex) if isinstance(v, np.ndarray) and v.size > 0]
    cell_id_array = [np.full(s.shape[0], i + 1) for i, s in enumerate(simplex_list) if isinstance(s, np.ndarray) and s.size > 0]

    # Filter out empty or invalid arrays
    valid_vertex = [v for v in vertex if isinstance(v, np.ndarray) and v.size > 0]
    valid_simplex = [s for s in simplex_list if isinstance(s, np.ndarray) and s.size > 0]
    
    if not valid_vertex or not valid_simplex:
        raise ValueError("No valid mesh data found in geosolution")

    concatenated_id_array = np.concatenate(vertex_id_array)
    concatenated_cell_id_array = np.concatenate(cell_id_array)

    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(valid_vertex),
        cells=np.concatenate(valid_simplex),
        vertex_attr=pd.DataFrame({'id': concatenated_id_array}),
        cells_attr=pd.DataFrame({'id': concatenated_cell_id_array})
    )

    return meshes

@wbgeo_component(identifier='geosolution_liquidearth_visualization',  # unique identifier
                 title='Push Geosolution to LiquidEarth',  # human readable (Default) title
                 description='push the geosolution to a new space in Liquid Earth',
                 color='#f4a259',
                 border_color='#000000',
                 group='visualisation',
                 return_name='space link',  # name of the returned port
                 )
def push_geosolution_to_le(geosolution: GeomodelResults , space_name:str, model_name:str, api_token:str) -> str:
    """
    Push a geosolution to Liquid Earth.

    Parameters:
    - geosolution: The geosolution object containing the results.
    - space_name: The name of the Liquid Earth space to create and upload to.
    - model_name: The name of the model in Liquid Earth.
    - api_token: The API token for authentication.

    Returns:
    - link: The link to the space  in Liquid Earth.
    """
    meshes = convert_to_subsurface_mesh(geosolution) # Convert geosolution to subsurface mesh format.requires a gempy solution
    link = le.upload_mesh_to_new_space(space_name, meshes, model_name, api_token).deep_link
    return link