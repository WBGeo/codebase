from core.object_components import InputData_StructuralElements
import numpy as np
import pandas as pd


from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
import typing

# Add some file path types:
# The frontend will handle them specially (via their identifier), yet they are strings in the backend
# We use the typing.Annotated notation here, as we can't use the @wbgeo_type decorator on builtin types
CSVFileDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='CSVFileDataType', controlled='RemoteFile|endswith=.csv')]
JSONFileDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='JSONFileDataType', controlled='RemoteFile|endswith=.json')]


# Register this function as a component
@wbgeo_component(description='Provides a geo model',
                 title='Load Model',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='geo_input_data_fix',  # a unique identifier
                 return_name='input_data',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def geo_input_data_fix(name: str = 'Model 12',
                       extent_str: str = '0, 2000, 0, 1000, 0, 1000',
                       resolution_str: str = '40, 20, 20',
                       surface_points_file: CSVFileDataType = 'model5_surface_points_df.csv',
                       orientations_file: CSVFileDataType = 'model5_surface_points_df.csv',
                       mapping_file: JSONFileDataType = 'model_12_mapping.json',
                       with_faults : bool = False
                       ) -> InputData_StructuralElements:
    # TODO: Provide a proper input type which does not require strings
    import os
    import pathlib

    datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

    try:
      extent = np.array([int(i.strip()) for i in extent_str.split(",")])
    except ValueError as e:
      raise ValueError('Illegal format for extent_str: ', e)
    try:
      resolution = np.array([int(i.strip()) for i in resolution_str.split(",")])
    except ValueError as e:
      raise ValueError('Illegal format for resolution_str: ', e)

    surface_points = pd.read_csv(os.path.join(datadir, 'examples/input_data/', surface_points_file))
    orientations = None
    if orientations_file != 'None':
        orientations = pd.read_csv(os.path.join(datadir, 'examples/input_data/', orientations_file))

    mapping_object = {}
    if mapping_file != 'None':
        with open (os.path.join(datadir, 'examples/input_data/', mapping_file), 'r') as f:
            import json
            mapping_object = json.load(f)

    # turn list into tuple
    # todo: is this even necessary?
    real_mapping_object = {}
    if 'mapping' in mapping_object:
      real_mapping_object = {k: tuple(v) for k,v in mapping_object["mapping"].items()}

    return InputData_StructuralElements(
        name=name,
        extent=extent,
        resolution=resolution,
        surface_points=surface_points,
        orientations=orientations,
        mapping_object=real_mapping_object,
        faults=mapping_object['faults'] if with_faults and 'faults' in mapping_object else None,
        # fault_relations=np.array(
        #     [[0, 1, 1],
        #      [0, 0, 0],
        #      [0, 0, 0]]
        # ),
        fault_relations=None, # TODO: fault relations
    )


