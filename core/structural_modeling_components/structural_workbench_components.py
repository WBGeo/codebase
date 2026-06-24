import json
import typing

import pydantic
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type, wbgeo_inspector, \
  InspectorHelper
from pydantic import BaseModel

from core.object_components import InputData_StructuralElements, InputData_FaultElements, \
  StructuralModelResults, FaultModelResults
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.interpolator_functions import interpolator_parameters
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import \
  InterpolationMethod, OKParams, RBFParams, UCKParams, GeoINRParams, FDIParams, UKParams, \
  InterpolationParameterSet
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import \
  plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D
from core.structural_modeling_components.structural_objects.grids import grid_classes


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
  except:
    print("Failed to build frame")
    pass
  # interpolator_parameters.default_ok_params(frame[key].context) if frame is not None else None

  return CtrlGroup(id='root', inner=[
    CtrlGroup(id='frame', inner=[
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
                                                            (InterpolationMethod.UNIVERSAL_KRIGING,
                                                             'uk', UKParams,
                                                             interpolator_parameters.default_uk_params(
                                                               frame[
                                                                 key].context) if frame is not None else None),
                                                          ]),

                                            ) for key in group_names
                                ]),
    # CtrlGroup(id='faults', inner=[CtrlLabel(label='Faults: (WIP) ')]) # show nothing for faults
  ])


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
  :param options: Optional smart options for interpolation method and parameter selection.
  :return: StructuralModelResults containing scalar fields, lithology block, and surface meshes.
  """
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
                 color='#8cb369',
                 border_color='#000000',
                 group='Inputs',
                 identifier='wbgeo::fault_modeling',
                 return_name='fault_model',
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

  :param data_faults: Fault input data including surface points, orientations, and fault names.
  :param grid: The computation grid defining the model domain.
  :return: FaultModelResults containing fault surfaces and the domain map.
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


@wbgeo_component(identifier='wbgeo::inspect_faull_model_result_plot_structural_model_3D_sf',
                 title='Plot Fault Model Result 3D',
                 description='...')
@wbgeo_inspector()
def inspect_fault_model_result_plot_structural_model_3D_sf(
    fault_model_result: FaultModelResults, _inspector: InspectorHelper):
  plot_fault_model_3D(fault_model_result.fault_frame, show_surface_meshes=True)
