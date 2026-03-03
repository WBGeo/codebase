import json

import pydantic

from core.object_components import InputData_StructuralElements, StructuralModelResults
import pandas as pd

from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_type
import typing

from core.structural_modeling_components import general
from core.structural_modeling_components.general import build_structural_frame
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import \
  InterpolationMethod, OKParams
from core.structural_modeling_components.structural_objects.grids import grid_classes
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import \
  StructuralFrame

# Add some file path types:
# The frontend will handle them specially (via their identifier), yet they are strings in the backend
# We use the typing.Annotated notation here, as we can't use the @wbgeo_type decorator on builtin types
CSVFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='CSVFileDataType',
                           controlled='RemoteFile|endswith=.csv')]
JSONFileDataType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='JSONFileDataType',
                           controlled='RemoteFile|endswith=.json')]

# todo: ALu improve these by giving them a fixed tuple instead of Table?
# the following two annotations allow an editing of extends via the web-ui
AExtent6 = typing.Annotated[
  grid_classes.Extent6, AnnotatedScriptType(name='grid_classes.Extent6', color='aqua',
                                            identifier='grid_classes.Extent6',
                                            controlled='Table|xmin|xmax|ymin|ymax|zmin|zmax')]
AResolution3 = typing.Annotated[
  grid_classes.Resolution3, AnnotatedScriptType(name='grid_classes.Resolution3', color='aqua',
                                                identifier='grid_classes.Resolution3',
                                                controlled='Table|x|y|z')]


@wbgeo_component(description='Regular (axis-aligned) grid defined by an extent and a 3D resolution',
                 title='Regular Grid',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::regular_grid',  # a unique identifier
                 return_name='grid',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def _regular_grid_constr(extent: AExtent6 = [(0, 1000, 0, 1000, 0, 1000)],
                         # todo: fix tuple reporting
                         resolution: AResolution3 = [(50, 50, 50)]) -> grid_classes.RegularGrid:
  return grid_classes.RegularGrid(extent, resolution)


RemoteMappingFileType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='JSONFileDataType',
                           controlled='RemoteFile|endswith=mapping.json')]


@wbgeo_component(description='load_mapping',
                 title='load_mapping',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::load_mapping',  # a unique identifier
                 return_name='grid',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def load_mapping(path: RemoteMappingFileType):
  with open(path, "r") as fd:
    return {k: tuple(v) for k, v in json.load(fd).items()}


# Register this function as a component
@wbgeo_component(description='Provides a geo model',
                 title='InputData',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::geo_input_data',  # a unique identifier
                 return_name='input_data',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def geo_input_data(
    name: str = 'Model 6',
    surface_points_file: CSVFileDataType = 'model6_surface_points_df.csv',
    orientations_file: typing.Optional[CSVFileDataType] = 'model6_orientations_df.csv',
    mapping_file: JSONFileDataType = 'model6_mapping.json'
) -> InputData_StructuralElements:
  import os
  import pathlib

  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

  surface_points = pd.read_csv(os.path.join(datadir, 'examples/input_data/', surface_points_file))
  orientations = None
  if orientations_file is not None:
    orientations = pd.read_csv(os.path.join(datadir, 'examples/input_data/', orientations_file))

  mapping_object = {}
  if mapping_file is not None:
    with open(os.path.join(datadir, 'examples/input_data/', mapping_file), 'r') as fd:
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


@wbgeo_type(name='TODO', color='orange', identifier='ComplexInputOptions')
class StructuralFrameInputOptions(pydantic.BaseModel):
  pass
  # todo: Add options here
  # data_per_number: typing.Dict[int, ComplexNumberOption]


from py_api_wbgeo.smartcontrols import CtrlLabel, CtrlSelect, CtrlInt, CtrlIf, CtrlText, \
  CtrlFromType, CtrlGroup, SmartInput, SmartInputFormData


def group_from_type(type: type[pydantic.BaseModel]) -> list[
  CtrlLabel | CtrlSelect | CtrlInt | CtrlIf | CtrlGroup | CtrlFromType]:
  ret = []

  for name, field in type.model_fields.items():
    field_type = field.annotation
    default_value = field.default
    description = field.description
    if field_type == int:
      ret.append(CtrlInt(id=name, label=description, defaultValue=default_value))
    elif field_type == float:
      ret.append(CtrlInt(id=name, label=description, defaultValue=default_value))
    elif field_type == str:
      ret.append(CtrlText(id=name, label=description, defaultValue=default_value))
    else:
      ret.append(CtrlLabel(label="Unhandled " + name + " " + str(field_type)))

  return ret


@wbgeo_component(identifier='wbgeo:__internal__structral_modeling_smart_options',
                 description='structral_modeling_smart_options',
                 title='structral_modeling_smart_options')
def structral_modeling_smart_options(data_elements: InputData_StructuralElements) -> CtrlGroup:
  group_names = data_elements.mapping_object.keys()
  return CtrlGroup(id='root', inner=[
    CtrlGroup(id='faults', inner=[
                                   CtrlLabel(label='Group ')] + [
                                   CtrlGroup(id=key, inner=[
                                     CtrlLabel(label=key),
                                     CtrlSelect(label='method', id='method',
                                                options=[e.value for e in InterpolationMethod]),

                                     CtrlGroup(id='ok', inner=group_from_type(OKParams))

                                   ]) for key in group_names
                                 ]),
    # CtrlLabel(label='Group ' + key) for key in group_names
  ])


@wbgeo_component(identifier='wbgeo:__internal__structral_modeling_smart_options_to_data',
                 description='structral_modeling_smart_options_to_data',
                 title='structral_modeling_smart_options_to_data')
def structral_modeling_smart_options_to_data(
    _input: SmartInputFormData) -> StructuralFrameInputOptions:
  return None # todo


SmartStructuralFrameInputOptions = typing.Annotated[
  StructuralFrameInputOptions, SmartInput(inputs=['data_elements'],
                                          to_form=structral_modeling_smart_options,
                                          to_data=structral_modeling_smart_options_to_data)]


# Register this function as a component
@wbgeo_component(description='structral_modeling',
                 title='structral_modeling',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::structral_modeling',  # a unique identifier
                 return_name='structural_model',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def structural_modeling(
    data_elements: InputData_StructuralElements,
    options: SmartStructuralFrameInputOptions,
    grid: grid_classes.RegularGrid,
) -> StructuralModelResults:
  frame = general.build_structural_frame(input_data_elements=data_elements,
                                         grid=grid,
                                         # fault_model_results=fault_model_result
                                         )

  # todo: also faults with smart inputs

  structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
  )

  return structural_model_result
