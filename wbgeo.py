from typing import Optional

from py_api_wbgeo import apitypes
from core.object_components import InputData, GeomodelResults
from core.loading_components.geo_input_data import geo_input_data_fix
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator

####################################################################################################
# This file registers the various core components to be used with the visual DSL
####################################################################################################

# define the ScriptTypes (inputs & outputs are of this type)

IntDataType: apitypes.ScriptType = {"id": "IntDataType", "real_type": int, "name": 'int', "color": 'blue'}
BoolDataType: apitypes.ScriptType = {"id": "BoolDataType", "real_type": bool, "name": 'boolean', "color": 'blue'}
StringDataType: apitypes.ScriptType = {"id": "StringDataType", "real_type": str, "name": 'string', "color": 'blue'}
CSVFileDataType: apitypes.ScriptType = {"id": "CSVFileDataType", "real_type": str, "name": 'path', "color": 'aqua'}
JSONFileDataType: apitypes.ScriptType = {"id": "JSONFileDataType", "real_type": str, "name": 'path', "color": 'aqua'}

InterpolatedResultType: apitypes.ScriptType = {"id": "InterpolatedResultType",
                                               "real_type": BoolDataType,
                                               "name": "InterpolatedResult", "color": 'red'}

MeshInputType: apitypes.ScriptType = {"id": "MeshInputTypePlaceHolder",
                                      "real_type": BoolDataType,
                                      "name": "MeshInputType", "color": 'green'}
MeshOutputType: apitypes.ScriptType = {"id": "MeshOutputType",
                                       "real_type": BoolDataType,
                                       "name": "MeshOutputType", "color": '#2dd69e'}

PlaceholderType: apitypes.ScriptType = {"id": "PlaceholderType",
                                        "real_type": BoolDataType,
                                        "name": "?", "color": 'orange'}

PMType: apitypes.ScriptType = {"id": "PMTypePlaceHolder",
                               "real_type": BoolDataType,
                               "name": "ProcessSimType", "color": 'blue'}

InputDataType: apitypes.ScriptType = {"id": "InputData", "real_type": InputData,
                                      "name": 'Input data for a geological model', "color": 'orange'}
GeomodelResultsType: apitypes.ScriptType = {"id": "GeomodelResults", "real_type": GeomodelResults, "name": 'Geo Result',
                                            "color": '#76cf46'}


#
# Define some pre-conditions (guards)
# Return None if they match, otherwise return a human readable error text
def does_not_have_faults(input: InputData) -> Optional[str]:
    if input.faults is not None:
        # raising an error (or returning a string) gives the information about fails
        return "Faults are not supported"
    return None


def does_have_orientations(input: InputData) -> Optional[str]:
    if input.orientations is None:
        # raising an error (or returning a string) gives the information about fails
        return "Orientations are required"
    return None


# Register the various components

from py_api_wbgeo import nodesapi

nodesapi.register_script_block(identifier='geo_input_data_fix',  # unique identifier
                               title='Load Model',  # human readable (Default) title
                               is_object_type=True,  # this input represents an object itself
                               inputs=[  # the input ports
                                   {
                                       'param': 'name',  # the name of this port
                                       'type': StringDataType,  # the ports type (as in ScriptType)
                                       'default': 'Model 10',  # a default value
                                   }, {
                                       'param': 'extent_str',
                                       'type': StringDataType,
                                       'default': '0, 1000, 0, 1000, 0, 1000',
                                   }, {
                                       'param': 'resolution_str',
                                       'type': StringDataType,
                                       'default': '20, 20, 20',
                                   }, {
                                       'param': 'surface_points_file',
                                       'type': CSVFileDataType,  # the ports type (as in ScriptType)
                                       'default': 'model10_surface_points_df.csv',  # a default value
                                   }, {
                                       'param': 'orientations_file',
                                       'type': CSVFileDataType,  # the ports type (as in ScriptType)
                                       'default': 'model10_orientations_df.csv',  # a default value
                                   }, {
                                       'param': 'mapping_file',
                                       'type': JSONFileDataType,  # the ports type (as in ScriptType)
                                       'default': 'model_10_mapping.json',  # a default value
                                   }, {
                                       'param': 'with_faults',
                                       'type': BoolDataType,  # the ports type (as in ScriptType)
                                       'default': True,  # a default value
                                   },
                               ],
                               execute=nodesapi.create_geo_execute(geo_input_data_fix),
                               description='Provides a geo model ',
                               color='#4effef',
                               # the method which actually performs the calculation
                               outputs=[{  # the output ports
                                   'param': 'data', 'type': InputDataType,
                               }])

nodesapi.register_script_block(identifier='ordinary_kriging_interpolator',  # unique identifier
                               title='Ordinary Kriging interpolator',  # human readable (Default) title
                               inputs=[  # the (list of) input ports
                                   {
                                       'param': 'input_data',  # the name of this port
                                       'type': InputDataType,  # the ports type (as in ScriptType)
                                       'data_requirements': [does_not_have_faults],
                                   },
                                   {
                                       'param': 'var_range',
                                       'type': IntDataType,
                                       'default': 500,
                                       'data_requirements': [lambda
                                                                 var_range: None if var_range > 0 else "var_range must be greater than 0"]
                                   }
                               ],
                               execute=nodesapi.create_geo_execute(ordinary_kriging_interpolator),
                               description='Compute a model based on input data using kriging interpolation',
                               color='#74eb34',
                               # the method which actually performs the calculation
                               outputs=[{  # the output ports
                                   'param': 'result', 'type': GeomodelResultsType,
                               }])

nodesapi.register_script_block(identifier='universal_cokriging_interpolator',  # unique identifier
                               title='Cokriging interpolator',  # human readable (Default) title
                               inputs=[  # the input ports
                                   {
                                       'param': 'input_data',  # the name of this port
                                       'type': InputDataType,  # the ports type (as in ScriptType)
                                       'data_requirements': [does_have_orientations]
                                   }],
                               execute=nodesapi.create_geo_execute(universal_cokriging_interpolator),
                               description='Compute a model based on input data using universal co-kriging interpolation (gempy)',
                               color='#74eb34',
                               # the method which actually performs the calculation
                               outputs=[{  # the output ports
                                   'param': 'result', 'type': GeomodelResultsType,
                               }])


# Register 3 yet-to-be-implemented block types

# a placeholder, passthrough
def placeholder_m(**kwargs):
    return kwargs.get(next(iter(kwargs.keys())))


nodesapi.register_script_block(identifier='nyi_mesh',
                               title='Meshing',
                               inputs=[
                                   {
                                       'param': 'geomodel',
                                       'type': GeomodelResultsType,
                                       'data_requirements': [],
                                   }
                               ],
                               execute=nodesapi.create_geo_execute(placeholder_m),
                               color='#2cf6b3',
                               outputs=[{
                                   'param': 'mesh', 'type': MeshOutputType
                               }]
                               )

nodesapi.register_script_block(identifier='nyi_ps',
                               title='Process Simulation',
                               inputs=[
                                   {
                                       'param': 'i',
                                       'type': MeshOutputType,
                                       'data_requirements': [],
                                   }],
                               execute=nodesapi.create_geo_execute(placeholder_m),
                               color='#de6c83',
                               outputs=[
                                   {'param': 'o', 'type': PMType},
                               ],
                               )

nodesapi.register_script_block(identifier='nyi_ar',
                               title='Cloud AR Visualization',
                               inputs=[
                                   {
                                       'param': 'self',
                                       'type': PMType,
                                       'data_requirements': [],
                                   }],
                               execute=nodesapi.create_geo_execute(placeholder_m),
                               color='#e0ab4c',
                               outputs=[
                                   {'param': 'o', 'type': GeomodelResultsType},
                               ],
                               )
