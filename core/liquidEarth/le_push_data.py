import os
import typing

import liquid_earth_sdk as le
import subsurface as ss
import numpy as np
import pandas as pd
from core.object_components import StructuralModelResults
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_inspector
from py_api_wbgeo import apitypes

def convert_to_subsurface_mesh(geosolution: StructuralModelResults, mesh_type: str = "masked") -> ss.UnstructuredData:
    """
    Convert a StructuralModelResults to a subsurface UnstructuredData mesh.
    Collects the requested mesh type from every structural element across all groups,
    offsets simplex indices so they do not overlap when concatenated, and assigns
    per-element integer IDs to vertices and cells.

    Args:
        geosolution: The structural model results containing the structural frame.
        mesh_type: Which mesh variant to use ("masked", "unmasked", or "combined").
                   Defaults to "masked". Elements that do not have this mesh type
                   are silently skipped.

    Returns:
        A subsurface UnstructuredData object ready for upload.

    Raises:
        ValueError: If no valid meshes of the requested type are found.
    """
    frame = geosolution.structural_frame

    valid_vertex = []
    valid_simplex = []
    # colors = []  # hex color per element, same order as valid_vertex/valid_simplex

    for group in frame.structural_groups:
        for elem in group.structural_elements:
            try:
                verts, edges = elem.get_mesh(mesh_type)
            except KeyError:
                continue  # mesh type not computed for this element
            if verts is None or edges is None or len(verts) == 0 or len(edges) == 0:
                continue
            valid_vertex.append(np.asarray(verts))
            valid_simplex.append(np.asarray(edges))
            # colors.append(elem.color)

    fault_frame = frame.fault_frame
    if fault_frame is not None:
        for fault in fault_frame.fault_elements:
            try:
                verts, edges = fault.get_mesh(mesh_type)
            except KeyError:
                continue  # mesh type not computed for this fault
            if verts is None or edges is None or len(verts) == 0 or len(edges) == 0:
                continue
            valid_vertex.append(np.asarray(verts))
            valid_simplex.append(np.asarray(edges))
            # colors.append(fault.color)

    if not valid_vertex or not valid_simplex:
        raise ValueError(
            f"No valid '{mesh_type}' meshes found in geosolution. "
            "Ensure surface mesh extraction has been run before uploading."
        )

    # Offset simplex indices so they refer to the correct rows after concatenation
    idx_max = 0
    for simplex_array in valid_simplex:
        simplex_array += idx_max
        idx_max = int(simplex_array.max()) + 1

    # Assign a 1-based integer ID per element for vertex/cell attribution
    vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(valid_vertex)]
    cell_id_array = [np.full(s.shape[0], i + 1) for i, s in enumerate(valid_simplex)]

    meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
        vertex=np.concatenate(valid_vertex),
        cells=np.concatenate(valid_simplex),
        vertex_attr=pd.DataFrame({'id': np.concatenate(vertex_id_array)}),
        cells_attr=pd.DataFrame({'id': np.concatenate(cell_id_array)}),
    )

    return meshes

# define our own data type as a secret
SecretDataType = typing.Annotated[str, AnnotatedScriptType(name='secret', color='aqua', identifier='secret', controlled='password')]

#and annotate a string as being a liquid earth link
LETarget = typing.Annotated[str, AnnotatedScriptType(name='liquidearth_target', identifier='liquidearth_target')]


@wbgeo_component(identifier='geosolution_liquidearth_visualization',  # unique identifier
                 title='Push Geosolution to LiquidEarth',  # human readable (Default) title
                 description='push the geosolution to a new space in Liquid Earth',
                 color='#9fc5e8',
                 border_color='#000000',
                 group='Visualisation',
                 return_name='space link',  # name of the returned port
                 )
def push_geosolution_to_le(geosolution: StructuralModelResults, space_name: str = 'WBGeo: Demo',
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
        model_name = "WBGeo Model"
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
