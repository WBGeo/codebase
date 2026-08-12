"""
WBGeo workflow components for loading geological input data from files.

Provides components for creating RegularGrid instances and loading structural
and fault input data from CSV and JSON files into the pipeline.
"""
import collections
import json
import os
import pathlib
import typing

import pandas as pd
import pydantic
from py_api_wbgeo import smartcontrols
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_inspector, \
  InspectorHelper, wbgeo_type
from py_api_wbgeo.smartcontrols import CtrlGroup, CtrlOrderedGrouping, SmartInputFormData, \
  SmartInput

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import \
  plot_input_data_3D, plot_fault_input_data_3D
from core.structural_modeling_components.structural_objects.grids import grid_classes

# Add some file path types:
# The frontend handles these specially (via their identifier); in the backend they are strings.
# typing.Annotated is used because @wbgeo_type cannot be applied to builtin types.
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
# The following two annotations allow editing of extent/resolution via the web-UI.
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
                 title='Regular Grid',
                 color='#b0dfa9',
                 border_color='#000000',
                 group='Inputs',
                 identifier='wbgeo::regular_grid',
                 return_name='grid',
                 is_object_type=True,
                 )
def _regular_grid_constr(extent: AExtent6 = (0, 1000, 0, 1000, 0, 1000),
                         resolution: AResolution3 = (50, 50, 50)) -> grid_classes.RegularGrid:
  """
  Defines the 3D computation domain for the geological model.

  The extent sets the bounding box of the model in world coordinates.
  The resolution controls how many voxels the domain is divided into along
  each axis — higher resolution gives more detail but increases computation time.

  Parameters
  ----------
  extent : tuple[float, float, float, float, float, float]
      Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
  resolution : tuple[int, int, int]
      Number of voxels along each axis (nx, ny, nz).

  Returns
  -------
  RegularGrid
      Grid defining the computation domain.
  """
  return grid_classes.RegularGrid(extent, resolution)


GroupNames = typing.Annotated[
  typing.List[str], AnnotatedScriptType(name='Group Names', color='aqua',
                                        identifier='wbgeo::GroupNames',
                                        controlled='List|GroupName|header=ORDER MATTERS: From top (younger) to bottom (older). Younger groups will erode older groups forming unconformities.')]


@wbgeo_type(name='StructuralInputSmartInputOptions', color='orange',
            identifier='wbgeo::StructuralInputSmartInputOptions')
class StructuralInputSmartInputOptions(pydantic.BaseModel):
  root: typing.Dict[str, str]  # mapping of formation name -> grouo


@wbgeo_component(identifier='wbgeo:__internal__structural_input_smart_options',
                 title='structural_input_smart_options')
def structural_input_smart_options(surface_points_file: typing.Optional[SurfaceCSVFileDataType],
                                   group_names: GroupNames
                                   ) -> CtrlGroup:
  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
  # surface_points_file defaults to None (an empty canvas has no file picked
  # yet) -- the mapping form just comes back empty until one is connected,
  # rather than crashing on os.path.join(datadir, None).
  try:
    surface_points = pd.read_csv(os.path.join(datadir, surface_points_file))
    formations = sorted(surface_points["formation"].unique())
  except Exception:
    formations = []

  group_options = ['<Exclude>'] + group_names
  dv = {
    '<Exclude>': formations,
  }
  for g in group_names:
    dv[g] = []

  return CtrlGroup(id='root', inner=[
    CtrlOrderedGrouping(id='groups',
                        label='Ordering for groups by age  - from top (youngest) to bottom (oldest). Ordering of elements within groups follows the same rule - make sure to order from top (youngest) to bottom (oldest).',
                        groups=group_options,
                        items=formations,
                        defaultValue=dv,
                        )

  ]
                   )

StructuralInputMappingOptions = typing.Annotated[typing.Dict[str, typing.Tuple[str, ...]], AnnotatedScriptType(name='wbgeo::StructuralInputMappingOptions', identifier='wbgeo::StructuralInputMappingOptions', color='orange')]

@wbgeo_component(identifier='wbgeo:__internal__structural_input_smart_options_to_data',
                 title='structural_input_smart_options_to_data')
def structural_input_smart_options_to_data(
    _input: SmartInputFormData,
    group_names: GroupNames,
) -> StructuralInputMappingOptions:
  input_as_json = smartcontrols.form_data_as_dict(_input)
  grouped = collections.defaultdict(list)
  if "root" in input_as_json and "groups" in input_as_json["root"]:
    for gn in group_names:
      if gn in input_as_json["root"]["groups"]:
        grouped[gn].extend(input_as_json["root"]["groups"][gn])

  return {k: tuple(v) for k, v in grouped.items()}

SmartStructuralInputSmartInputOptions = typing.Annotated[
  StructuralInputMappingOptions, SmartInput(inputs=['surface_points_file', 'group_names'],
                                          to_form=structural_input_smart_options,
                                          to_data=structural_input_smart_options_to_data)]



@wbgeo_component(description='Input data for a Structural Geological Model. Requires surface points, optionally orientations and a mapping file.',
                 title='Input data for Structural Elements',
                 color='#b0dfa9',
                 border_color='#000000',
                 group='Inputs',
                 identifier='wbgeo::geo_input_data',
                 return_name='input_data',
                 is_object_type=True,
                 )
def structural_input_data(
    name: str = 'Model 1',
    surface_points_file: typing.Optional[SurfaceCSVFileDataType] = None,
    orientations_file: typing.Optional[OrientationsCSVFileDataType] = None,
    group_names: GroupNames = [],
    mapping_object: SmartStructuralInputSmartInputOptions = {}
) -> InputData_StructuralElements:
  """
  Loads input data for a structural geological model from CSV and JSON files.

  Surface points define the contact locations of geological interfaces.
  Orientations (optional) provide dip and azimuth constraints that improve
  interpolation quality. The mapping file assigns each formation to a
  stratigraphic group, which controls the layer ordering in the model.

  Parameters
  ----------
  name : str
      Name of the model, used for identification.
  surface_points_file : str
      CSV file with surface contact points. Required columns: X, Y, Z, formation.
  orientations_file : str, optional
      CSV file with orientation measurements. Required columns: X, Y, Z, G_x, G_y, G_z, formation.
  group_names: list[str]
      Sorted list of all stratigraphic groups names
  mapping: dict[str, tuple[str]]
      mapping formation names to stratigraphic groups.

  Returns
  -------
  InputData_StructuralElements
      Ready to connect to the Compute Structural Model component.
  """
  if surface_points_file is None:
    raise ValueError("A surface points CSV file is required.")

  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

  surface_points = pd.read_csv(os.path.join(datadir, surface_points_file))

  required_sp_cols = {"X", "Y", "Z", "formation"}
  missing = required_sp_cols - set(surface_points.columns)
  if missing:
    raise ValueError(f"Surface points CSV missing required columns: {missing}")

  orientations = None
  if orientations_file is not None:
    orientations = pd.read_csv(os.path.join(datadir, orientations_file))
    required_ori_cols = {"X", "Y", "Z", "G_x", "G_y", "G_z", "formation"}
    missing = required_ori_cols - set(orientations.columns)
    if missing:
      raise ValueError(f"Orientations CSV missing required columns: {missing}")

  return InputData_StructuralElements(
    name=name,
    surface_points=surface_points,
    orientations=orientations,
    mapping_object=mapping_object
  )


@wbgeo_component(identifier='wbgeo::inspect_structural_input_data_plot_3D',
                 title='Plot Input Data 3D',
                 description='...')
@wbgeo_inspector()
def inspect_structural_input_data_plot_3D(input_data: InputData_StructuralElements,
                                          _inspector: InspectorHelper):
  plot_input_data_3D(input_data)


# --- Fault input data ---


@wbgeo_component(identifier='wbgeo:__internal__faults_input_data_smart_options',
                 title='faults_input_data_smart_options')
def faults_input_data_smart_options(fault_surface_points_file: typing.Optional[SurfaceCSVFileDataType]) -> CtrlGroup:
  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
  # fault_surface_points_file defaults to None (an empty canvas has no file
  # picked yet) -- the selection form just comes back empty until one is
  # connected, rather than crashing on os.path.join(datadir, None).
  try:
    surface_points = pd.read_csv(os.path.join(datadir, fault_surface_points_file))
    formations = sorted(surface_points["formation"].unique())
  except Exception:
    formations = []
  group_options = ['<Exclude>', '<Faults>']
  dv = {'<Exclude>': formations, '<Faults>': []}
  return CtrlGroup(id='root', inner=[
    CtrlOrderedGrouping(id='fault_selection',
                        label='Drag fault formations into the <Faults> list',
                        groups=group_options,
                        items=formations,
                        defaultValue=dv,
                        )]
                   )


@wbgeo_component(identifier='wbgeo:__internal__faults_input_data_smart_options_to_data',
                 title='faults_input_data_smart_options_to_data')
def faults_input_data_smart_options_to_data(_input: SmartInputFormData) -> FaultNames:
  input_as_json = smartcontrols.form_data_as_dict(_input)
  fault_names = []
  if "root" in input_as_json and "fault_selection" in input_as_json["root"]:
    if "<Faults>" in input_as_json["root"]["fault_selection"]:
      fault_names.extend(input_as_json["root"]["fault_selection"]["<Faults>"])
  return fault_names


SmartFaultInputSmartInputOptions = typing.Annotated[
  FaultNames, SmartInput(inputs=['fault_surface_points_file'],
                         to_form=faults_input_data_smart_options,
                         to_data=faults_input_data_smart_options_to_data)]




@wbgeo_component(description='Input data for Fault Elements. Requires surface points and orientations.',
                 title='Fault Input Data',
                 color='#b0dfa9',
                 border_color='#000000',
                 group='Inputs',
                 identifier='wbgeo::faults_input_data',
                 return_name='faults_data',
                 is_object_type=True,
                 )
def faults_input_data(
    name: str = 'Faults Model 2',
    fault_surface_points_file: typing.Optional[SurfaceCSVFileDataType] = None,
    fault_orientations_file: typing.Optional[OrientationsCSVFileDataType] = None,
    fault_names: SmartFaultInputSmartInputOptions = []) -> InputData_FaultElements:
  """
  Loads input data for fault elements from CSV files.

  Surface points define the locations where faults are observed (e.g. from
  boreholes or outcrop mapping). Orientations constrain the dip and strike of
  each fault plane. The fault names list must match the formation names used
  in the surface points file and determines which faults are modelled.

  Parameters
  ----------
  name : str
      Name of the fault dataset, used for identification.
  fault_surface_points_file : str
      CSV file with fault surface points. Required columns: X, Y, Z, formation.
  fault_orientations_file : str, optional
      CSV file with fault orientation measurements. Required columns: X, Y, Z, G_x, G_y, G_z, formation.
      Orientations are required for fault interpolation (Universal Co-Kriging).
  fault_names : list[str]
      Fault names to model, matching formation names in the surface points file.

  Returns
  -------
  InputData_FaultElements
      Ready to connect to the Compute Fault Model component.
  """
  datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()

  if not fault_names:
    raise ValueError("At least one fault name must be provided.")

  if fault_surface_points_file is None:
    raise ValueError("A fault surface points CSV file is required.")

  surface_points = pd.read_csv(os.path.join(datadir, fault_surface_points_file))

  required_sp_cols = {"X", "Y", "Z", "formation"}
  missing = required_sp_cols - set(surface_points.columns)
  if missing:
    raise ValueError(f"Fault surface points CSV missing required columns: {missing}")

  if fault_orientations_file is None:
    raise ValueError(
      "Fault orientations are required for fault interpolation (Universal Co-Kriging) "
      "but no orientations file was provided."
    )

  orientations = pd.read_csv(os.path.join(datadir, fault_orientations_file))

  required_ori_cols = {"X", "Y", "Z", "G_x", "G_y", "G_z", "formation"}
  missing = required_ori_cols - set(orientations.columns)
  if missing:
    raise ValueError(f"Fault orientations CSV missing required columns: {missing}")

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
