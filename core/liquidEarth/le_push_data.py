import liquid_earth_sdk as le
import subsurface as ss
import numpy as np
import pandas as pd
from py_api_wbgeo.nodesapi import wbgeo_component

def convert_to_subsurface_mesh(geosolution):


    vertex: list[np.ndarray] = geosolution.surface_meshes_vertices
    simplex_list: list[np.ndarray] = geosolution.surface_meshes_edges

    idx_max = 0
    for simplex_array in simplex_list:
        simplex_array += idx_max
        idx_max = simplex_array.max() + 1

    vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(vertex)]
    cell_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(simplex_list)]

    concatenated_id_array = np.concatenate(vertex_id_array)
    concatenated_cell_id_array = np.concatenate(cell_id_array)

    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(vertex),
        cells=np.concatenate(simplex_list),
        vertex_attr=pd.DataFrame({'id': concatenated_id_array}),
        cells_attr=pd.DataFrame({'id': concatenated_cell_id_array})
    )

    return meshes

#%%
@wbgeo_component(identifier='geosolution_liquidearth_visualization',  # unique identifier
                 title='Push Geosolution to LiquidEarth',  # human readable (Default) title
                 description='push the geosolution to a new space in Liquid Earth',
                 color='#f4a259',
                 border_color='#000000',
                 group='Interpolation',
                 return_name='results',  # name of the returned port
                 )
def push_geosolution_to_le(geosolution, space_name, model_name, api_token):
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
    link = le.upload_mesh_to_new_space(space_name, meshes, model_name, api_token)
    return link