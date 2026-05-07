import json

import pydantic
from pydantic import BaseModel

from core.object_components import InputData_StructuralElements, StructuralModelResults, \
  InputData_FaultElements, FaultModelResults
import pandas as pd

from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_type, wbgeo_inspector, \
  InspectorHelper
import typing

from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.general import build_structural_frame
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import \
  InterpolationMethod, OKParams, RBFParams, UCKParams, GeoINRParams, FDIParams, UKParams, \
  InterpolationParameterSet
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import \
  plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D
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


class StructuralFrameInputOptions_Frame(pydantic.BaseModel):
  method: InterpolationMethod = InterpolationMethod.RADIAL_BASIS_FUNCTION
  ok: typing.Optional[OKParams] = None
  rbf: typing.Optional[RBFParams] = None
  uck: typing.Optional[UCKParams] = None
  geoinr: typing.Optional[GeoINRParams] = None
  fdi: typing.Optional[FDIParams] = None
  uk: typing.Optional[UKParams] = None

  def get_params(self) -> InterpolationParameterSet | None:
    if self.method == InterpolationMethod.ORDINARY_KRIGING:
      return self.ok
    elif self.method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
      return self.rbf
    elif self.method == InterpolationMethod.UNIVERSAL_COKRIGING:
      return self.uck
    elif self.method == InterpolationMethod.GEOINR:
      return self.geoinr
    elif self.method == InterpolationMethod.FINITE_DIFFERENCES:
      return self.fdi
    elif self.method == InterpolationMethod.UNIVERSAL_KRIGING:
      return self.uk
    else:
      raise ValueError("Unhandled interpolation method", self.method)


class StructuralFrameInputOptions_Root(pydantic.BaseModel):
  frame: typing.Dict[str, StructuralFrameInputOptions_Frame]


@wbgeo_type(name='StructuralFrameInputOptions', color='orange',
            identifier='wbgeo::StructuralFrameInputOptions')
class StructuralFrameInputOptions(pydantic.BaseModel):
  root: StructuralFrameInputOptions_Root


from py_api_wbgeo.smartcontrols import CtrlLabel, CtrlSelect, CtrlInt, CtrlFloat, CtrlIf, CtrlText, \
  CtrlFromType, CtrlGroup, SmartInput, SmartInputFormData, ASmartControl


def group_from_type(type: type[pydantic.BaseModel]) -> list[
  ASmartControl]:
  ret = []

  for name, field in type.model_fields.items():
    field_type = field.annotation
    default_value = field.default
    description = field.description
    if field_type == int :
      ret.append(CtrlInt(id=name, label=name, description=description, defaultValue=default_value))
    elif field_type == typing.Optional[int]:
      ret.append(CtrlInt(id=name, label=f'({name})', description=description, defaultValue=default_value))
    elif field_type == float:
      ret.append(CtrlFloat(id=name, label=name, description=description, defaultValue=default_value))
    elif field_type == typing.Optional[float]:
      ret.append(CtrlFloat(id=name, label=f'({name})', description=description, defaultValue=default_value))
    elif field_type == str:
      ret.append(CtrlText(id=name, label=name, description=description, defaultValue=default_value))
    elif field_type == typing.Optional[str]:
      ret.append(CtrlText(id=name, label=f'({name})', description=description, defaultValue=default_value))
    else:
      ret.append(CtrlLabel(label=description + f": Unhandled `{str(name)}: {str(field_type)}`"))

  return ret


@wbgeo_component(identifier='wbgeo:__internal__structural_modeling_smart_options',
                 description='structural_modeling_smart_options',
                 title='structural_modeling_smart_options')
def structural_modeling_smart_options(data_elements: InputData_StructuralElements) -> CtrlGroup:
  group_names = data_elements.mapping_object.keys()

  def construct_group_from(stack: typing.List[typing.Tuple[InterpolationMethod, str, typing.Type[BaseModel]]]) -> typing.List[ASmartControl]:
    """A helper function to construct the if/else block for smart inputs"""
    if len(stack) == 0:
      return [CtrlLabel(label=' Unhandled method ')]
    enum_val, key, param_type  = stack[0]
    return [CtrlIf(condition=f"'$.method' == \"{enum_val.value}\"",
                   when_true=[CtrlGroup(id=key, inner=group_from_type(param_type))],
                   when_false=construct_group_from(stack[1:])
                   )]

  return CtrlGroup(id='root', inner=[
    CtrlGroup(id='frame', inner=[
                                   CtrlLabel(label='Frame Options: ')] + [
                                   CtrlGroup(id=key, inner=[
                                     CtrlLabel(label="Options for layer: " + key),
                                     CtrlSelect(label='method', id='method',
                                                options=[e.value for e in InterpolationMethod]),
                                     ]+

                                     construct_group_from([
                                       (InterpolationMethod.ORDINARY_KRIGING, 'ok', OKParams),
                                       (InterpolationMethod.RADIAL_BASIS_FUNCTION, 'rbf', RBFParams),
                                       (InterpolationMethod.UNIVERSAL_COKRIGING, 'uck', UCKParams),
                                       (InterpolationMethod.GEOINR, 'geoinr', GeoINRParams),
                                       (InterpolationMethod.FINITE_DIFFERENCES, 'fdi', FDIParams),
                                       (InterpolationMethod.UNIVERSAL_KRIGING, 'uk', UKParams),
                                                           ]),

                                   ) for key in group_names
                                 ]),
    #CtrlGroup(id='faults', inner=[CtrlLabel(label='Faults: (WIP) ')]) # show nothing for faults
  ])

def unflatten_dict(d):
  ret = dict()
  for key, vlaue in d.items():
    parts = key.split(".")
    d = ret
    for part in parts[:-1]:
      if part not in d:
        d[part] = dict()
      d = d[part]
    d[parts[-1]] = vlaue
  return ret

@wbgeo_component(identifier='wbgeo:__internal__structural_modeling_smart_options_to_data',
                 description='structural_modeling_smart_options_to_data',
                 title='structural_modeling_smart_options_to_data')
def structural_modeling_smart_options_to_data(
    _input: SmartInputFormData) -> StructuralFrameInputOptions:
  data = json.loads(_input) # _input is a json dict
  data = unflatten_dict({k: v for k, v in data.items() if v is not None}) # un-flatten it and remove nulls
  # and convert it to our StructuralFrameInputOptions type
  return StructuralFrameInputOptions(**data)



SmartStructuralFrameInputOptions = typing.Annotated[
  StructuralFrameInputOptions, SmartInput(inputs=['data_elements'],
                                          to_form=structural_modeling_smart_options,
                                          to_data=structural_modeling_smart_options_to_data)]


# Register this function as a component
@wbgeo_component(description='structural_modeling',
                 title='Compute Structural Model',  # The title shown in the GUI
                 color='#8cb369',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Inputs',
                 identifier='wbgeo::structural_modeling',  # a unique identifier
                 return_name='structural_model',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def structural_modeling(
    elements: InputData_StructuralElements,
    grid: grid_classes.RegularGrid,
    fault_model: typing.Optional[FaultModelResults] = None,
    options: SmartStructuralFrameInputOptions = None,
) -> StructuralModelResults:
  frame = general.build_structural_frame(input_data_elements=elements,
                                         grid=grid,
                                         fault_model_results=fault_model
                                         )

  # Apply Options (if present)
  if options and options.root and options.root.frame:
    for frame_name, frame_options in options.root.frame.items():
      frame[frame_name].set_interpolation_method(frame_options.method)
      params = frame_options.get_params()
      if params is not None:
        # frame[frame_name].set_interpolation_params(params) # todo: Does not work due to very weird defaults
        frame[frame_name].configure_interpolation_params(**params.model_dump(exclude_none=True))

  # todo: also faults with smart inputs?

  structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
  )

  return structural_model_result


@wbgeo_component(identifier='wbgeo::structural_model_result_detailed_report',
                 title='Detailed Report',
                 description='...')
@wbgeo_inspector()
def inspect_structural_model_result_detailed_report(structural_model_result: StructuralModelResults,
                                                    _inspector: InspectorHelper):
  structural_model_result.structural_frame.detailed_report()


@wbgeo_component(identifier='wbgeo::inspect_structural_model_result_plot_structural_model_2D',
                 title='Plot Model Result 2D',
                 description='...')
@wbgeo_inspector()
def inspect_structural_model_result_plot_structural_model_2D(
    structural_model_result: StructuralModelResults, _inspector: InspectorHelper):
  plot_structural_model_2D(structural_model_result.structural_frame)


@wbgeo_component(identifier='wbgeo::inspect_structural_model_result_plot_structural_model_3D_sf',
                 title='Plot Model Result 3D',
                 description='...')
@wbgeo_inspector()
def inspect_structural_model_result_plot_structural_model_3D(
    structural_model_result: StructuralModelResults, _inspector: InspectorHelper):
  plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

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
def inspect_faull_model_result_plot_structural_model_3D_sf(
    fault_model_result: FaultModelResults, _inspector: InspectorHelper):
  plot_fault_model_3D(fault_model_result.fault_frame, show_surface_meshes=True)
