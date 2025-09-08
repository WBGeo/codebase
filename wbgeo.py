from typing import Optional

from py_api_wbgeo import apitypes

from core.object_components import InputData, GeomodelResults
from core.loading_components.geo_input_data import geo_input_data_fix
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.rbf_interpolation import rbf_interpolator
# from core.interpolator_components.geo_inr import geo_inr_interpolator # removed until updated
# from core.interpolator_components.loopstructural_old import loop_structural_interpolator # removed until updated
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data

####################################################################################################
# This file registers the various core components to be used with the visual DSL
####################################################################################################

# define the ScriptTypes (inputs & outputs are of this type)

IntDataType: apitypes.ScriptType = {"id": "IntDataType", "real_type": int, "name": 'int', "color": 'blue'}
ListDataType: apitypes.ScriptType = {"id": "ListDataType", "real_type": list, "name": 'list', "color": 'blue'}
BoolDataType: apitypes.ScriptType = {"id": "BoolDataType", "real_type": bool, "name": 'boolean', "color": 'blue'}
StringDataType: apitypes.ScriptType = {"id": "StringDataType", "real_type": str, "name": 'string', "color": 'blue'}
# CSVFileDataType: apitypes.ScriptType = {"id": "CSVFileDataType", "real_type": str, "name": 'path', "color": 'aqua'}
# JSONFileDataType: apitypes.ScriptType = {"id": "JSONFileDataType", "real_type": str, "name": 'path', "color": 'aqua'}

InterpolatedResultType: apitypes.ScriptType = {"id": "InterpolatedResultType",
                                               "real_type": BoolDataType,
                                               "name": "InterpolatedResult", "color": 'red'}

MeshInputType: apitypes.ScriptType = {"id": "MeshInputTypePlaceHolder",
                                      "real_type": BoolDataType,
                                      "name": "MeshInputType", "color": 'green'}

import pyvista as pv

MeshOutputType: apitypes.ScriptType = {"id": "MeshOutputType",
                                       "real_type": pv.DataSet,
                                       "name": "MeshResult", "color": '#5b8e7d'}

PlaceholderType: apitypes.ScriptType = {"id": "PlaceholderType",
                                        "real_type": BoolDataType,
                                        "name": "?", "color": 'orange'}

PMType: apitypes.ScriptType = {"id": "PMTypePlaceHolder",
                               "real_type": BoolDataType,
                               "name": "ProcessSimResult", "color": 'blue'}

InputDataType: apitypes.ScriptType = {"id": "InputData", "real_type": InputData,
                                      "name": 'Input data for a geological model', "color": 'orange'}

GeomodelResultsType: apitypes.ScriptType = {"id": "GeomodelResults", "real_type": GeomodelResults, "name": 'Geo Result',
                                            "color": '#f4a259'}


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


# The following is to be replaced with the @wbgeo_type and @wbgeo_component annotations

# nodesapi.register_script_block(identifier='geo_input_data_fix',  # unique identifier
#                                title='Load Model',  # human readable (Default) title
#                                is_object_type=True,  # this input represents an object itself
#                                inputs=[  # the input ports
#                                    {
#                                        'param': 'name',  # the name of this port
#                                        'type': StringDataType,  # the ports type (as in ScriptType)
#                                        'default': 'Model 12',  # a default value
#                                    }, {
#                                        'param': 'extent_str',
#                                        'type': StringDataType,
#                                        'default': '0, 2000, 0, 1000, 0, 1000',
#                                    }, {
#                                        'param': 'resolution_str',
#                                        'type': StringDataType,
#                                        'default': '40, 20, 20',
#                                    }, {
#                                        'param': 'surface_points_file',
#                                        'type': CSVFileDataType,  # the ports type (as in ScriptType)
#                                        'default': 'model12_surface_points_df.csv',  # a default value
#                                    }, {
#                                        'param': 'orientations_file',
#                                        'type': CSVFileDataType,  # the ports type (as in ScriptType)
#                                        'default': 'model12_orientations_df.csv',  # a default value
#                                    }, {
#                                        'param': 'mapping_file',
#                                        'type': JSONFileDataType,  # the ports type (as in ScriptType)
#                                        'default': 'model_12_mapping.json',  # a default value
#                                    }, {
#                                        'param': 'with_faults',
#                                        'type': BoolDataType,  # the ports type (as in ScriptType)
#                                        'default': False,  # a default value
#                                    },
#                                ],
#                                execute=nodesapi.create_geo_execute(geo_input_data_fix),
#                                description='Provides a geo model ',
#                                color='#8cb369',
#                                border_color='#000000',
#                                group='Inputs',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'data', 'type': InputDataType,
#                                }])

# nodesapi.register_script_block(identifier='loop_structural_interpolator',  # unique identifier
#                                title='LoopStructural interpolator',  # human readable (Default) title
#                                inputs=[  # the (list of) input ports
#                                    {
#                                        'param': 'input_data',  # the name of this port
#                                        'type': InputDataType,  # the ports type (as in ScriptType)
#                                        'data_requirements': [does_not_have_faults],
#                                    },
#                                ],
#                                execute=nodesapi.create_geo_execute(loop_structural_interpolator),
#                                description='Compute a model based on input data using LoopStructural interpolation',
#                                color='#f4a259',
#                                border_color='#000000',
#                                group='Interpolation',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': GeomodelResultsType,
#                                }])

# nodesapi.register_script_block(identifier='geoinr_interpolator',  # unique identifier
#                                title='GeoINR interpolator',  # human readable (Default) title
#                                inputs=[  # the (list of) input ports
#                                    {
#                                        'param': 'input_data',  # the name of this port
#                                        'type': InputDataType,  # the ports type (as in ScriptType)
#                                        'data_requirements': [does_not_have_faults],
#                                    },
#                                    {
#                                        'param': 'beta',
#                                        'type': IntDataType,
#                                        'default': 5,
#                                    }
#                                ],
#                                execute=nodesapi.create_geo_execute(geo_inr_interpolator),
#                                description='Compute a model based on input data using GeoINR interpolation',
#                                color='#f4a259',
#                                border_color='#000000',
#                                group='Interpolation',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': GeomodelResultsType,
#                                }])

# nodesapi.register_script_block(identifier='rbf_interpolator',  # unique identifier
#                                title='Radial Basis Function interpolator',  # human readable (Default) title
#                                inputs=[  # the (list of) input ports
#                                    {
#                                        'param': 'input_data',  # the name of this port
#                                        'type': InputDataType,  # the ports type (as in ScriptType)
#                                        'data_requirements': [does_not_have_faults],
#                                    },
#                                    {
#                                        'param': 'kernel',
#                                        'type': StringDataType,
#                                        'default': "linear",
#                                    },
#                                    {
#                                        'param': 'smoothing',
#                                        'type': IntDataType,
#                                        'default': 0,
#                                    },
#                                    {
#                                        'param': 'neighbors',
#                                        'type': IntDataType,
#                                        'default': None,
#                                    },
#                                    {
#                                        'param': 'epsilon',
#                                        'type': IntDataType,
#                                        'default': 1,
#                                    }
#                                ],
#                                execute=nodesapi.create_geo_execute(rbf_interpolator),
#                                description='Compute a model based on input data using RBF interpolation',
#                                color='#f4a259',
#                                border_color='#000000',
#                                group='Interpolation',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': GeomodelResultsType,
#                                }])

# nodesapi.register_script_block(identifier='ordinary_kriging_interpolator',  # unique identifier
#                                title='Ordinary Kriging interpolator',  # human readable (Default) title
#                                inputs=[  # the (list of) input ports
#                                    {
#                                        'param': 'input_data',  # the name of this port
#                                        'type': InputDataType,  # the ports type (as in ScriptType)
#                                        'data_requirements': [does_not_have_faults],
#                                    },
# {
#                                        'param': 'var_model',
#                                        'type': StringDataType,
#                                        'default': "gaussian"
#                                    },
# {
#                                        'param': 'var_sill',
#                                        'type': IntDataType,
#                                        'default': 1,
#                                        'data_requirements': [lambda
#                                                                  var_sill: None if var_sill > 0 else "var_range must be greater than 0"]
#                                    },
#                                    {
#                                        'param': 'var_range',
#                                        'type': IntDataType,
#                                        'default': 500,
#                                        'data_requirements': [lambda
#                                                                  var_range: None if var_range > 0 else "var_range must be greater than 0"]
#                                    },
# {
#                                        'param': 'var_nugget',
#                                        'type': IntDataType,
#                                        'default': 0,
#                                        'data_requirements': [lambda
#                                                                  var_range: None if var_range >= 0 else "var_range must be 0 or greater"]
#                                    },
# {
#                                        'param': 'anisotropy_scaling_z=0.3',
#                                        'type': IntDataType,
#                                        'default': 0.3,
#                                    },
#
#                                ],
#                                execute=nodesapi.create_geo_execute(ordinary_kriging_interpolator),
#                                description='Compute a model based on input data using kriging interpolation',
#                                color='#f4a259',
#                                border_color='#000000',
#                                group='Interpolation',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': GeomodelResultsType,
#                                }])
#
# nodesapi.register_script_block(identifier='universal_cokriging_interpolator',  # unique identifier
#                                title='Cokriging interpolator',  # human readable (Default) title
#                                inputs=[  # the input ports
#                                    {
#                                        'param': 'input_data',  # the name of this port
#                                        'type': InputDataType,  # the ports type (as in ScriptType)
#                                        'data_requirements': [does_have_orientations]
#                                    }],
#                                execute=nodesapi.create_geo_execute(universal_cokriging_interpolator),
#                                description='Compute a model based on input data using universal co-kriging interpolation (gempy)',
#                                color='#f4a259',
#                                border_color='#000000',
#                                group='Interpolation',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': GeomodelResultsType,
#                                }])
#
# # meshing
# nodesapi.register_script_block(identifier='meshing',  # unique identifier
#                                title='structured meshing',  # human readable (Default) title
#                                inputs=[  # the input ports
#                                    {
#                                        'param': 'geomodel_result',  # the name of this port
#                                        'type': GeomodelResultsType,  # the ports type (as in ScriptType)
#                                    }, {
#                                        'param': 'refinement_data_str',
#                                        'type': StringDataType,
#                                        'default': '25, 21, 16, 5, 6',
#                                    }, {
#                                        'param': 'z_threshold',
#                                        'type': IntDataType,
#                                        'default': 0.1,
#                                    }, {
#                                        'param': 'tolerance',
#                                        'type': IntDataType,
#                                        'default': 1,
#                                    }],
#                                execute=nodesapi.create_geo_execute(create_structured_mesh_data_str),
#                                description='Create a structured mesh based on the Structural Geological Model',
#                                color='#5b8e7d',
#                                border_color='#000000',
#                                group='Meshing',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': MeshOutputType,
#                                }])
#
# nodesapi.register_script_block(identifier='meshing_unstructured',  # unique identifier
#                                title='unstructured meshing',  # human readable (Default) title
#                                inputs=[  # the input ports
#                                    {
#                                        'param': 'geomodel_result',  # the name of this port
#                                        'type': GeomodelResultsType,  # the ports type (as in ScriptType)
#                                    }, {
#                                        'param': 'mesh_size',
#                                        'type': IntDataType,
#                                        'default': 30,
#                                    }],
#                                execute=nodesapi.create_geo_execute(create_unstructured_mesh_data),
#                                description='Create a unstructured mesh based on the Structural Geological Model',
#                                color='#5b8e7d',
#                                border_color='#000000',
#                                group='Meshing',
#                                # the method which actually performs the calculation
#                                outputs=[{  # the output ports
#                                    'param': 'result', 'type': MeshOutputType,
#                                }])


# Register 3 yet-to-be-implemented block types

# a placeholder, passthrough
def placeholder_m(**kwargs):
    return kwargs.get(next(iter(kwargs.keys())))


# nodesapi.register_script_block(identifier='nyi_mesh',
#                                title='Meshing',
#                                inputs=[
#                                    {
#                                        'param': 'geomodel',
#                                        'type': GeomodelResultsType,
#                                        'data_requirements': [],
#                                    }
#                                ],
#                                execute=nodesapi.create_geo_execute(placeholder_m),
#                                color='#2cf6b3',
#                                outputs=[{
#                                    'param': 'mesh', 'type': MeshOutputType
#                                }]
#                                )
#
# nodesapi.register_script_block(identifier='nyi_ps',
#                                title='Process Simulation',
#                                inputs=[
#                                    {
#                                        'param': 'mesh',
#                                        'type': MeshOutputType,
#                                        'data_requirements': [],
#                                    }],
#                                execute=nodesapi.create_geo_execute(placeholder_m),
#                                color='#bc4b51',
#                                border_color='#000000',
#                                group='Process Simulation',
#                                outputs=[
#                                    {'param': 'result', 'type': PMType},
#                                ],
#                                )
#
# nodesapi.register_script_block(identifier='nyi_ar',
#                                title='Cloud AR Visualization',
#                                inputs=[
#                                    {
#                                        'param': 'self',
#                                        'type': PMType,
#                                        'data_requirements': [],
#                                    }],
#                                execute=nodesapi.create_geo_execute(placeholder_m),
#                                color='#f4e285',
#                                border_color='#000000',
#                                group='Οther', # greek Ο due to ordering
#                                outputs=[
#                                    {'param': 'o', 'type': GeomodelResultsType},
#                                ],
#                                )

