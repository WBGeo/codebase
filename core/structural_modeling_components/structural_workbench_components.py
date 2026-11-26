import json
import pathlib
import typing

import pydantic
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type, wbgeo_inspector, \
  InspectorHelper, AnnotatedScriptType, BasicallyABufferedFile
from pydantic import BaseModel

from core.object_components import InputData_StructuralElements, InputData_FaultElements, \
  StructuralModelResults, FaultModelResults
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.interpolator_functions import interpolator_parameters
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import \
  InterpolationMethod, OKParams, RBFParams, UCKParams, GeoINRParams, FDIParams, PLIParams, UKParams, \
  InterpolationParameterSet
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import \
  plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D
from core.structural_modeling_components.structural_objects.grids import grid_classes
from core.structural_modeling_components.structural_model_export import (StructuralModelExportFormat,
                                                                         export_structural_model as _export)
from core.structural_modeling_components.structural_model_file import FILE_EXTENSION, load_structural_model
from core import workbench_palette as palette


class StructuralFrameInputOptions_Frame(pydantic.BaseModel):
  method: InterpolationMethod = InterpolationMethod.RADIAL_BASIS_FUNCTION
  ok: typing.Optional[OKParams] = None
  rbf: typing.Optional[RBFParams] = None
  uck: typing.Optional[UCKParams] = None
  geoinr: typing.Optional[GeoINRParams] = None
  fdi: typing.Optional[FDIParams] = None
  pli: typing.Optional[PLIParams] = None
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
    elif self.method == InterpolationMethod.PIECEWISE_LINEAR:
      return self.pli
    elif self.method == InterpolationMethod.UNIVERSAL_KRIGING:
      return self.uk
    else:
      raise ValueError("Unhandled interpolation method", self.method)


class StructuralFrameInputOptions_Root(pydantic.BaseModel):
  frame: typing.Dict[str, StructuralFrameInputOptions_Frame]
  faults: typing.Dict[str, str] = {} # fault -> group name


@wbgeo_type(name='StructuralFrameInputOptions', color=palette.PARAMETER,
            identifier='wbgeo::StructuralFrameInputOptions')
class StructuralFrameInputOptions(pydantic.BaseModel):
  root: StructuralFrameInputOptions_Root


from py_api_wbgeo import smartcontrols
from py_api_wbgeo.smartcontrols import CtrlLabel, CtrlSelect, CtrlInt, CtrlFloat, CtrlIf, CtrlText, \
  CtrlGroup, SmartInput, SmartInputFormData, ASmartControl



@wbgeo_component(identifier='wbgeo:__internal__structural_modeling_smart_options',
                 title='structural_modeling_smart_options')
def structural_modeling_smart_options(data_elements: InputData_StructuralElements,
                                      grid: grid_classes.RegularGrid,
                                      fault_model: typing.Optional[FaultModelResults] = None
                                      ) -> CtrlGroup:
  """(Internal) component that builds the smart input form for interpolation method selection."""
  group_names = data_elements.mapping_object.keys()

  def construct_group_from(stack: typing.List[
    typing.Tuple[InterpolationMethod, str, typing.Type[BaseModel], typing.Any]]) -> typing.List[
    ASmartControl]:
    """A helper function to construct the if/else block for smart inputs"""
    if len(stack) == 0:
      return [CtrlLabel(label=' Unhandled method ')]
    enum_val, key, param_type, param_instance = stack[0]
    return [CtrlIf(condition=f"'$.method' == \"{enum_val.value}\"",
                   when_true=[CtrlGroup(id=key, inner=smartcontrols.smart_group_from_type(param_type, param_instance))],
                   when_false=construct_group_from(stack[1:])
                   )]

  frame = None
  try:
    frame = general.build_structural_frame(input_data_elements=data_elements,
                                           grid=grid,
                                           fault_model_results=fault_model
                                           )
  except Exception:
    pass
  # interpolator_parameters.default_ok_params(frame[key].context) if frame is not None else None

  ret = [CtrlGroup(id='frame', inner=[
                                  CtrlLabel(label='Frame Options: ')] + [
                                  CtrlGroup(id=key, inner=[
                                                            CtrlLabel(
                                                              label="Options for layer: " + key),
                                                            CtrlSelect(label='method', id='method',
                                                                       options=[e.value for e in
                                                                                InterpolationMethod],
                                                                       defaultValue=InterpolationMethod.RADIAL_BASIS_FUNCTION),
                                                          ]
                                                          +

                                                          construct_group_from([
                                                            (InterpolationMethod.ORDINARY_KRIGING,
                                                             'ok', OKParams,
                                                             interpolator_parameters.default_ok_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.RADIAL_BASIS_FUNCTION,
                                                             'rbf', RBFParams,
                                                             interpolator_parameters.default_rbf_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.UNIVERSAL_COKRIGING,
                                                             'uck', UCKParams,
                                                             interpolator_parameters.default_uck_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.GEOINR, 'geoinr',
                                                             GeoINRParams,
                                                             interpolator_parameters.default_geo_inr_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.FINITE_DIFFERENCES,
                                                             'fdi', FDIParams,
                                                             interpolator_parameters.default_fdi_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.PIECEWISE_LINEAR,
                                                             'pli', PLIParams,
                                                             interpolator_parameters.default_pli_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                            (InterpolationMethod.UNIVERSAL_KRIGING,
                                                             'uk', UKParams,
                                                             interpolator_parameters.default_uk_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                          ]),

                                            ) for key in group_names
                                ])]
  if fault_model is not None:
    fault_options = ["<All>"] + [str(k) for k in data_elements.mapping_object.keys()]
    ret.append(CtrlGroup(id='faults', inner=
    [CtrlLabel(label='Fault Activity By Group: ')]
    +
    [CtrlSelect(label=fault_elem.name, id=fault_elem.name, options=fault_options) for
     fault_elem in fault_model.fault_frame.fault_elements]
                         ))
  return CtrlGroup(id='root', inner=ret)
    # CtrlGroup(id='faults', inner=[CtrlLabel(label='Faults: (WIP) ')]) # show nothing for faults


@wbgeo_component(identifier='wbgeo:__internal__structural_modeling_smart_options_to_data',
                 title='structural_modeling_smart_options_to_data')
def structural_modeling_smart_options_to_data(
    _input: SmartInputFormData) -> StructuralFrameInputOptions:
  """(Internal) component that converts the smart input form data dict to StructuralFrameInputOptions."""
  return smartcontrols.form_data_to_wbgeo_type_data(_input, StructuralFrameInputOptions)


SmartStructuralFrameInputOptions = typing.Annotated[
  StructuralFrameInputOptions, SmartInput(inputs=['data_elements'],
                                          to_form=structural_modeling_smart_options,
                                          to_data=structural_modeling_smart_options_to_data)]

# Register this function as a component
@wbgeo_component(description='Computes a Structural Geological model from input data, grid, and (optionally) fault model. With smart options for interpolation methods and parameters.',
                 title='Compute Structural Model',  # The title shown in the GUI
                 color=palette.STRUCTURAL_MODEL,  # the color of the components
                 border_color=palette.BORDER,  # and its border color
                 group='Inputs',
                 identifier='wbgeo::structural_modeling',  # a unique identifier
                 return_name='structural_model',  # the name for the returned-port
                 input_checks=[general_faults.check_fault_model_not_crosscutting],
                 positionInGroup=880,
                 )  # inputs are handled via the method signature
def structural_modeling(
    elements: InputData_StructuralElements,
    grid: grid_classes.RegularGrid,
    fault_model: typing.Optional[FaultModelResults] = None,
    options: SmartStructuralFrameInputOptions = None,
) -> StructuralModelResults:
  """
  Computes a 3D structural geological model from input data and a grid.

  Interpolates scalar fields for each stratigraphic group to reconstruct
  geological interfaces across the model domain. If a fault model is
  connected, interpolation is performed per fault domain so that formations
  are correctly offset across faults. The optional smart options input allows
  selecting the interpolation method (e.g. RBF, Kriging) and tuning its
  parameters per stratigraphic group.

  The output contains the lithology block, scalar fields, and surface meshes
  for all geological formations and — if present — fault surfaces.

  :param elements: Structural input data including surface points, orientations, and group mapping.
  :param grid: The computation grid defining the model domain.
  :param fault_model: Optional fault model output from the Compute Fault Model component.
      Must have cross_cutting=False (raises ValueError otherwise) -- a
      cross-cutting fault model has no domain map to interpolate against.
  :param options: Optional smart options for interpolation method and parameter selection.
  :return: StructuralModelResults containing scalar fields, lithology block, and surface meshes.

  Raises:
      ValueError: fault_model is connected and has cross_cutting=True.
  """
  # Pre-checks can be ignored/skipped by a caller (see docs/developers/components.md),
  # so also enforced here rather than relying solely on the Workbench's
  # input_checks catching the connection.
  general_faults.check_fault_model_not_crosscutting(fault_model)

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

  # Apply fault set_fault_activity_by_group
  if options and options.root and options.root.faults:
    for fault_name, group_name in options.root.faults.items():
      if group_name != '<All>':
        frame.set_fault_activity_by_group(fault_name=fault_name, group_name=group_name)


  structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
  )

  return structural_model_result


@wbgeo_component(identifier='wbgeo::structural_model_result_detailed_report',
                 title='Detailed Report')
@wbgeo_inspector()
def inspect_structural_model_result_detailed_report(structural_model_result: StructuralModelResults,
                                                    _inspector: InspectorHelper):
  structural_model_result.structural_frame.detailed_report()


@wbgeo_component(identifier='wbgeo::inspect_structural_model_result_plot_structural_model_2D',
                 title='Plot Model Result 2D')
@wbgeo_inspector()
def inspect_structural_model_result_plot_structural_model_2D(
    structural_model_result: StructuralModelResults, _inspector: InspectorHelper):
  plot_structural_model_2D(structural_model_result.structural_frame)


@wbgeo_component(identifier='wbgeo::inspect_structural_model_result_plot_structural_model_3D_sf',
                 title='Plot Model Result 3D')
@wbgeo_inspector()
def inspect_structural_model_result_plot_structural_model_3D(
    structural_model_result: StructuralModelResults, _inspector: InspectorHelper):
  plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)


# -----------------------------------------------------------------------------
# Fault modeling
# -----------------------------------------------------------------------------

@wbgeo_component(description='Computes fault surfaces and fault domains from input data and grid.',
                 title='Compute Fault Model',
                 color=palette.FAULT_MODEL,
                 border_color=palette.BORDER,
                 group='Inputs',
                 identifier='wbgeo::fault_modeling',
                 return_name='fault_model',
                 positionInGroup=890,
                 )
def compute_fault_frame(
    data_faults: InputData_FaultElements,
    grid: grid_classes.RegularGrid,
) -> FaultModelResults:
  """
  Computes the fault model from fault input data and a grid.

  Builds a fault frame from the provided surface points and orientations,
  then runs the fault domain computation. The resulting fault model partitions
  the grid into domains separated by fault surfaces, and provides the fault
  geometry as surface meshes. Connect the output to the Compute Structural
  Model component to incorporate faults into the geological model.

  If any two faults cross-cut (their isovalue bands overlap), domaining is
  skipped and the returned result has cross_cutting=True -- fault surfaces
  are still available to inspect (e.g. with Plot Fault Model 3D) to see where
  the faults cross, but this result cannot be wired into Compute Structural
  Model until the fault geometry is fixed and recomputed without crossing.

  :param data_faults: Fault input data including surface points, orientations, and fault names.
  :param grid: The computation grid defining the model domain.
  :return: FaultModelResults containing fault surfaces and, if not cross-cutting, the domain map.
  """
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


@wbgeo_component(identifier='wbgeo::inspect_fault_model_result_plot_structural_model_3D_sf',
                 title='Plot Fault Model Result 3D',
                 description='...')
@wbgeo_inspector()
def inspect_fault_model_result_plot_structural_model_3D_sf(
    fault_model_result: FaultModelResults, _inspector: InspectorHelper):
  plot_fault_model_3D(fault_model_result.fault_frame, show_surface_meshes=True)


StructuralModelExportFormatType = typing.Annotated[
  StructuralModelExportFormat,
  AnnotatedScriptType(name='StructuralModelExportFormat', identifier='StructuralModelExportFormatType',
                      controlled='Select|' + '|'.join(f.value for f in StructuralModelExportFormat),
                      color=palette.PARAMETER),
]

StructuralModelFileType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color=palette.PARAMETER, identifier='wbgeo::structural_model_file',
                           controlled='RemoteFile|endswith=' + FILE_EXTENSION)]


@wbgeo_component(
  title='Export Structural Model',
  description='Exports a structural model as a file. "wbgeo_sm" is the WBGeo file format: it keeps everything and '
              'can be loaded again with Load Structural Model. "gocad_ts", "obj" and "vtk" export the surfaces '
              'for other software as a ZIP (GOCAD TSurf, Wavefront OBJ, VTK; VTK also contains the lithology '
              'block and scalar fields as a grid), with surfaces.csv listing the surfaces in stratigraphic order.',
  group='Inputs',
  positionInGroup=0,
  identifier='wbgeo::export_structural_model',
  return_name='file',
  color=palette.STRUCTURAL_EXPORT,
  border_color=palette.BORDER,
)
def export_structural_model(
    structural_model: StructuralModelResults,
    format: StructuralModelExportFormatType = StructuralModelExportFormat.WBGEO.value,
) -> BasicallyABufferedFile:
  """
  :param structural_model: The structural model to export.
  :param format: "wbgeo_sm" (WBGeo's own file), "gocad_ts", "obj" or "vtk".
  :return: The file to download: structural_model.wbgeo_sm, or a ZIP for the other formats.
  """
  return _export(structural_model, format)


@wbgeo_component(
  title='Load Structural Model',
  description='Loads a structural model saved with Export Structural Model (format "wbgeo_sm"). The loaded '
              'model can be used like a computed one: meshing, simulation and inspection.',
  color=palette.STRUCTURAL_MODEL,
  border_color=palette.BORDER,
  group='Inputs',
  positionInGroup=870,
  identifier='wbgeo::load_structural_model',
  return_name='structural_model',
)
def load_structural_model_from_file(model_file: StructuralModelFileType) -> StructuralModelResults:
  """
  :param model_file: Path (relative to the codebase root) of a '.wbgeo_sm' file, picked via the file browser.
  :return: The structural model stored in the file.
  """
  root = pathlib.Path(__file__).parent.parent.parent.resolve()
  path = root / model_file
  if not path.is_file():
    raise ValueError(f"Structural model file not found: {path}")
  return load_structural_model(path)
