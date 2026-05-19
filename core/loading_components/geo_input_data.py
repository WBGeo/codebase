import json
import typing

import pandas as pd
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_inspector, \
  InspectorHelper

from core.object_components import InputData_StructuralElements, InputData_FaultElements, \
  FaultModelResults
from core.structural_modeling_components import general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import \
  plot_fault_model_2D, plot_fault_model_3D, \
  plot_input_data_3D, plot_fault_input_data_3D
from core.structural_modeling_components.structural_objects.grids import grid_classes

# Add some file path types:
# The frontend will handle them specially (via their identifier), yet they are strings in the backend
# We use the typing.Annotated notation here, as we can't use the @wbgeo_type decorator on builtin types
CSVFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='CSVFileDataType',
                           controlled='RemoteFile|endswith=.csv')]
SurfaceCSVFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='SurfaceCSVFileDataType',
                           controlled='RemoteFile|endswith=.csv|contains=surface_points')]
OrientationsCSVFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='OrientationsCSVFileDataType',
                           controlled='RemoteFile|endswith=.csv|contains=orientations_')]
JSONFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='JSONFileDataType',
                           controlled='RemoteFile|endswith=.json')]

# todo: ALu improve these by giving them a fixed tuple instead of Table?
# the following two annotations allow an editing of extends via the web-ui
AExtent6 = typing.Annotated[
  grid_classes.Extent6, AnnotatedScriptType(name='grid_classes.Extent6', color='aqua',
                                            identifier='grid_classes.Extent6',
                                            controlled='Tuple|xmin|xmax|ymin|ymax|zmin|zmax')]
AResolution3 = typing.Annotated[
  grid_classes.Resolution3, AnnotatedScriptType(name='grid_classes.Resolution3', color='aqua',
                                                identifier='grid_classes.Resolution3',
                                                controlled='Tuple|x|y|z')]

FaultNames = typing.Annotated[
  typing.List[str], AnnotatedScriptType(name='Fault Names', color='aqua',
                                                identifier='wbgeo::FaultNames',
                                                controlled='List|FaultName')]

@wbgeo_component(description='Regular (axis-aligned) grid defined by an extent and a 3D resolution',
                 title='Regular Grid',  # The title shown in the GUI
                 color='#b0dfa9',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::regular_grid',  # a unique identifier
                 return_name='grid',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def _regular_grid_constr(extent: AExtent6 = (0, 1000, 0, 1000, 0, 1000),
                         # todo: fix tuple reporting
                         resolution: AResolution3 = (50, 50, 50)) -> grid_classes.RegularGrid:
  return grid_classes.RegularGrid(extent, resolution)


RemoteMappingFileType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='RemoteMappingFileType',
                           controlled='RemoteFile|endswith=mapping.json')]


# the mapping is not provided as a interface-type? Should we do so?
# @wbgeo_component(description='load_mapping',
#                  title='load_mapping',  # The title shown in the GUI
#                  color='#8cb369',  # the color of the components
#                  border_color='#000000',  # and its border color
#                  group='Inputs',
#                  identifier='wbgeo::load_mapping',  # a unique identifier
#                  return_name='grid',  # the name for the returned-port
#                  is_object_type=True,
#                  )  # inputs are handled via the method signature
def load_mapping(path: RemoteMappingFileType):
  with open(path, "r") as fd:
    return {k: tuple(v) for k, v in json.load(fd).items()}


# Register this function as a component
@wbgeo_component(description='Provides a geo model',
                 title='Input data for Structural Elements',  # The title shown in the GUI
                 color='#b0dfa9',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::geo_input_data',  # a unique identifier
                 return_name='input_data',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def structural_input_data(
    name: str = 'Model 1',
    surface_points_file: SurfaceCSVFileDataType = 'examples/synthetic_examples/model1/input_data/geological_data/model1_surface_points_df.csv',
    orientations_file: typing.Optional[OrientationsCSVFileDataType] = 'examples/synthetic_examples/model1/input_data/geological_data/model1_orientations_df.csv',
    mapping_file: JSONFileDataType = 'examples/synthetic_examples/model1/input_data/geological_data/model1_mapping.json'
) -> InputData_StructuralElements:
  import os
  import pathlib

  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

  surface_points = pd.read_csv(os.path.join(datadir, surface_points_file))
  orientations = None
  if orientations_file is not None:
    orientations = pd.read_csv(os.path.join(datadir, orientations_file))

  mapping_object = {}
  if mapping_file is not None:
    with open(os.path.join(datadir, mapping_file), 'r') as fd:
      import json
      mapping_object = {k: tuple(v) for k, v in json.load(fd).items()}

  # turn list into tuple
  # todo: is this even necessary?
  # real_mapping_object = {}
  # if 'mapping' in mapping_object:
  #   real_mapping_object = {k: tuple(v) for k,v in mapping_object["mapping"].items()}

  data_elements = InputData_StructuralElements(
    name=name,
    surface_points=surface_points,
    orientations=orientations,
    mapping_object=mapping_object
  )

  return data_elements


@wbgeo_component(identifier='wbgeo::inspect_structural_input_data_plot_3D',
                 title='Plot Input Data 3D',
                 description='...')
@wbgeo_inspector()
def inspect_structural_input_data_plot_3D(input_data: InputData_StructuralElements,
                                          _inspector: InspectorHelper):
  plot_input_data_3D(input_data)



### faults
@wbgeo_component(description='Provides fault data',
                 title='Input data for Fault Elements',
                 color='#b0dfa9',
                 border_color='#000000',
                 group='Inputs',
                 identifier='wbgeo::faults_input_data',
                 return_name='faults_data',
                 is_object_type=True,
                 )
def faults_input_data(
    name: str = 'Faults Model 2',
    fault_surface_points_file: SurfaceCSVFileDataType = 'examples/synthetic_examples/model1/input_data/geological_data/model1_surface_points_df.csv',
    fault_orientations_file: typing.Optional[OrientationsCSVFileDataType] = 'examples/synthetic_examples/model1/input_data/geological_data/model1_orientations_df.csv',
    fault_names: FaultNames = ['fault',]) -> InputData_FaultElements:
  import os
  import pathlib

  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

  surface_points = pd.read_csv(os.path.join(datadir, fault_surface_points_file))
  orientations = None
  if fault_orientations_file is not None:
    orientations = pd.read_csv(os.path.join(datadir, fault_orientations_file))


  return InputData_FaultElements(
    name=name,
    fault_surface_points=surface_points,
    fault_orientations=orientations,
    fault_names=fault_names
  )


@wbgeo_component(identifier='wbgeo::inspect_fault_input_data_plot_3D',
                 title='Plot Fault Input Data 3D',
                 description='...')
@wbgeo_inspector()
def inspect_fault_input_data_plot_3D(faults_data: InputData_FaultElements,
                                     _inspector: InspectorHelper):
  plot_fault_input_data_3D(faults_data)


# Register this function as a component
@wbgeo_component(description='fault_modeling',
                 title='Compute Fault Model',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::fault_modeling',  # a unique identifier
                 return_name='fault_model',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def compute_fault_frame(
    data_faults: InputData_FaultElements,
    grid: grid_classes.RegularGrid,
) -> FaultModelResults:
  fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid
  )
  if grid is None:
    raise ValueError("Missing grid?")
  fault_model_result = general_faults.compute_fault_domains(fault_frame)

  return fault_model_result

@wbgeo_component(identifier='wbgeo::inspect_fault_model_result_detailed_report',
                 title='Detailed Fault Report',
                 description='...')
@wbgeo_inspector()
def inspect_fault_model_result_detailed_report(fault_model_result: FaultModelResults,
                                                    _inspector: InspectorHelper):
  fault_model_result.fault_frame.detailed_report()


@wbgeo_component(identifier='wbgeo::inspect_fault_model_result_plot_structural_model_2D',
                 title='Plot Fault Model Result 2D',
                 description='...')
@wbgeo_inspector()
def inspect_fault_model_result_plot_structural_model_2D(
    fault_model_result: FaultModelResults, _inspector: InspectorHelper):
  plot_fault_model_2D(fault_model_result.fault_frame)


@wbgeo_component(identifier='wbgeo::inspect_faull_model_result_plot_structural_model_3D_sf',
                 title='Plot Fault Model Result 3D',
                 description='...')
@wbgeo_inspector()
def inspect_fault_model_result_plot_structural_model_3D_sf(
    fault_model_result: FaultModelResults, _inspector: InspectorHelper):
  plot_fault_model_3D(fault_model_result.fault_frame, show_surface_meshes=True)
