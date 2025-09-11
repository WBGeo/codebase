import os
import typing

import liquid_earth_sdk as le
import subsurface as ss
import numpy as np
import pandas as pd
from core.object_components import InputData, GeomodelResults
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_inspector
from py_api_wbgeo import apitypes



def convert_to_subsurface_mesh(geosolution):
    # Note: surface_meshes_vertices and surface_meshes_edges are List[List[NpNDArrayFp64/Int64]]
    vertex_groups: List[List[np.ndarray]] = geosolution.surface_meshes_vertices
    simplex_groups: List[List[np.ndarray]] = geosolution.surface_meshes_edges

    # Use only the first group (group 0)
    if not vertex_groups or not simplex_groups:
        raise ValueError("No mesh groups found in geosolution")

    if len(vertex_groups) == 0 or len(simplex_groups) == 0:
        raise ValueError("Empty mesh groups in geosolution")

    # Extract only the first group
    vertex = vertex_groups[0]
    simplex_list = simplex_groups[0]

    idx_max = 0
    for simplex_array in simplex_list:
        # Ensure we're working with numpy arrays
        if isinstance(simplex_array, np.ndarray) and simplex_array.size > 0:
            simplex_array += idx_max
            idx_max = simplex_array.max() + 1

    # Create ID arrays - filter out empty or invalid arrays
    valid_vertex = [v for v in vertex if isinstance(v, np.ndarray) and v.size > 0]
    valid_simplex = [s for s in simplex_list if isinstance(s, np.ndarray) and s.size > 0]

    if not valid_vertex or not valid_simplex:
        raise ValueError("No valid mesh data found in geosolution group 0")

    vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(valid_vertex)]
    cell_id_array = [np.full(s.shape[0], i + 1) for i, s in enumerate(valid_simplex)]

    concatenated_id_array = np.concatenate(vertex_id_array)
    concatenated_cell_id_array = np.concatenate(cell_id_array)

    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(valid_vertex),
        cells=np.concatenate(valid_simplex),
        vertex_attr=pd.DataFrame({'id': concatenated_id_array}),
        cells_attr=pd.DataFrame({'id': concatenated_cell_id_array})
    )

    return meshes

# define our own data type as a secret
SecretDataType = typing.Annotated[str, AnnotatedScriptType(name='secret', color='aqua', identifier='secret', controlled='password')]

#and annotate a string as being a liquid earth link
LETarget = typing.Annotated[str, AnnotatedScriptType(name='liquidearth_target', identifier='liquidearth_target')]


@wbgeo_component(identifier='geosolution_liquidearth_visualization',  # unique identifier
                 title='Push Geosolution to LiquidEarth',  # human readable (Default) title
                 description='push the geosolution to a new space in Liquid Earth',
                 color='#f4a259',
                 border_color='#000000',
                 group='visualisation',
                 return_name='space link',  # name of the returned port
                 )
def push_geosolution_to_le(geosolution: GeomodelResults, space_name: str = 'WBGeo: Demo',
                           model_name: str = None, api_token: SecretDataType = None) -> LETarget:
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
    if api_token is None:
      api_token = os.getenv("LIQUIDEARTH_TOKEN")
    if model_name is None:
      model_name = geosolution.name
    if api_token is None:
      raise Exception("No api_token or LIQUIDEARTH_TOKEN environment variable")

    meshes = convert_to_subsurface_mesh(geosolution) # Convert geosolution to subsurface mesh format.requires a gempy solution
    link = le.upload_mesh_to_new_space(space_name, meshes, model_name, api_token).deep_link
    return link

# the LE component outputs a `str`
# this component just provides a nice naming
@wbgeo_component(identifier='geosolution_liquidearth_visualization_open', title='Open in LiquidEarth',
                          description='')
@wbgeo_inspector()
def geosolution_liquidearth_visualization_open(space_link: LETarget):
  print(space_link) # the editor turns links in text into clickable links
