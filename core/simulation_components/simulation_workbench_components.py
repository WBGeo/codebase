"""
WBGeo Workbench components for the SfePy pipeline and GOLEM hydrothermal/simple-reservoir pipelines.

Thin-wrapper file only: per the project's "one workbench file per step" rule
(already applied to structural_modeling_components/meshing_components),
@wbgeo_component must appear here and nowhere else in core/simulation_components/.
The actual logic lives in the SfePy/GOLEM builder and runner modules
and is called here unmodified.
"""
import os
import pathlib
import shutil
import typing
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pydantic
from py_api_wbgeo import smartcontrols
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type, wbgeo_inspector, InspectorHelper, \
    AnnotatedScriptType, BasicallyABufferedFile
from py_api_wbgeo.smartcontrols import CtrlGroup, CtrlLabel, SmartInput, SmartInputFormData

from core.object_components import MeshResults, StructuralModelResults, SimulationResults, SliceAxis, SliceAxisType
from core.meshing_components.explicit.unstructured.mesh_data import LithoMappingMode, LithoMappingModeType, WellData
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    SfepyHydrothermalProblemBuilder, SfepyCustomBuilder, SfepyProblem, RockUnitProperties, FluidProperties,
    enumerate_rock_units, check_mesh_has_known_type, check_mesh_has_lithology_mapping,
    check_mesh_has_no_engineering_objects,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_run import (
    run_simulation_sfepy as _run_simulation_sfepy,
)
from core.simulation_components.simulation_packages.simulation_export import (
    SimulationExportFormat,
    export_simulation_results as _export_simulation_results,
)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemCustomBuilder, GolemFaultProperties, GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_run import (
    run_simulation_golem as _run_simulation_golem,
)
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials,
    plot_variable_along_line, plot_variable_time_series,
)
from core import workbench_palette as palette

AnnotatedPoint = typing.Annotated[Tuple[float, float, float], AnnotatedScriptType(identifier='wbgeo::mesh:point', name='Point', controlled='Tuple|X|Y|Z', color=palette.PARAMETER)]

# RemoteFile-controlled path (frontend handles it specially via the
# identifier; in the backend it's a plain string) -- same pattern as
# core.loading_components.geo_input_data's CSVFileDataType/etc., and the
# proven-working way to let a user pick a file in this Workbench. Used
# instead of BasicallyABufferedFile (a direct upload widget) for
# build_custom_sfepy_problem's input_file: confirmed live that
# BasicallyABufferedFile's upload widget isn't supported as a component
# *input* in this frontend build ("Unsupported input type
# file_import_export") -- it's the only place in this codebase
# BasicallyABufferedFile was ever used as an input (every other usage is
# an export/download, where it does work), unlike RemoteFile, which
# every existing file-input component in this codebase already relies on.
CustomSfepyFileDataType = typing.Annotated[
    str, AnnotatedScriptType(name='path', color=palette.PARAMETER, identifier='CustomSfepyFileDataType',
                             controlled='RemoteFile|endswith=.py')]

SimulationExportFormatType = typing.Annotated[
    SimulationExportFormat,
    AnnotatedScriptType(name='SimulationExportFormat', identifier='SimulationExportFormatType',
                        controlled='Select|' + '|'.join(f.value for f in SimulationExportFormat), color=palette.PARAMETER),
]

# GOLEM custom inputs use the same RemoteFile pattern, but select MOOSE .i files.
CustomGolemFileDataType = typing.Annotated[
    str, AnnotatedScriptType(name='path', color=palette.PARAMETER, identifier='CustomGolemFileDataType',
                             controlled='RemoteFile|endswith=.i')]


@wbgeo_type(name='GOLEM Problem', color=palette.SIMULATION_BUILDER, identifier='wbgeo::GolemProblem')
class GolemProblem(pydantic.BaseModel):
    """Workbench-visible envelope around one GOLEM problem definition.

    This mirrors SfepyProblem: the Workbench DSL needs one concrete registered
    port type. GOLEM has two generated model families (hydrothermal and simple
    reservoir) plus a custom-input path, and all three connect to the same
    Run GOLEM Simulation component through this envelope.
    """
    hydrothermal: Optional[GolemHydrothermalProblemBuilder] = None
    simple_reservoir: Optional[GolemSimpleReservoirBuilder] = None
    custom: Optional[GolemCustomBuilder] = None

    def model_post_init(self, __context) -> None:
        configured = sum(
            item is not None
            for item in (self.hydrothermal, self.simple_reservoir, self.custom)
        )
        if configured != 1:
            raise ValueError(
                "GolemProblem needs exactly one of hydrothermal, simple_reservoir "
                "or custom set."
            )

    def inner(self):
        if self.hydrothermal is not None:
            return self.hydrothermal
        if self.simple_reservoir is not None:
            return self.simple_reservoir
        return self.custom

# -----------------------------------------------------------------------------
# Smart options: per-rock-unit / fluid / fault-zone property editing
# -----------------------------------------------------------------------------
# Mirrors structural_workbench_components.py's structural_modeling_smart_options
# (per-group interpolation method/params) -- same shape of problem: the set of
# rock unit names is only known once geomodel_result is connected, so the form
# has to be built dynamically from it rather than being a fixed set of fields.


#: Shared default so build_hydrothermal_problem's fault zone still works with
#: enable_fault_zone=True even when no `options` is connected at all (mirrors
#: include_flow needing no wiring) -- also used to pre-fill the smart form.
_DEFAULT_FAULT_ZONE_PROPERTIES = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
)


class HydrothermalOptions_Root(pydantic.BaseModel):
    fluid: FluidProperties = FluidProperties()
    rock_properties: Dict[str, RockUnitProperties] = {}
    # Always present/editable in the form (so its fields have somewhere to
    # live), but only actually used by SfepyHydrothermalProblemBuilder if
    # enable_fault_zone=True on build_hydrothermal_problem below -- keeping
    # that as a plain checkbox parameter (rather than an in-form toggle)
    # avoids relying on CtrlIf's condition-path scoping, which isn't
    # exercised anywhere else in this codebase against a dict-of-groups
    # layout.
    fault_zone_properties: RockUnitProperties = _DEFAULT_FAULT_ZONE_PROPERTIES


@wbgeo_type(name='HydrothermalOptions', color=palette.PARAMETER,
            identifier='wbgeo::HydrothermalOptions')
class HydrothermalOptions(pydantic.BaseModel):
    root: HydrothermalOptions_Root


@wbgeo_component(identifier='wbgeo:__internal__hydrothermal_problem_smart_options',
                 title='hydrothermal_problem_smart_options')
def hydrothermal_problem_smart_options(geomodel_result: StructuralModelResults) -> CtrlGroup:
    """(Internal) component that builds the smart input form for hydrothermal
    fluid/rock-unit/fault-zone properties."""
    try:
        rock_units = enumerate_rock_units(geomodel_result)
    except Exception:
        rock_units = []

    return CtrlGroup(id='root', inner=[
        CtrlLabel(label='Fluid Properties:'),
        CtrlGroup(id='fluid', inner=smartcontrols.smart_group_from_type(
            FluidProperties, FluidProperties())),
        CtrlLabel(label='Rock Unit Properties:'),
        CtrlGroup(id='rock_properties', inner=[
            CtrlGroup(id=name, inner=[CtrlLabel(label=name)] + smartcontrols.smart_group_from_type(
                RockUnitProperties, RockUnitProperties(name=name)))
            for _, name in rock_units
        ]),
        CtrlLabel(label='Fault Zone Properties (only used if "Enable Fault Zone" is checked below):'),
        CtrlGroup(id='fault_zone_properties', inner=smartcontrols.smart_group_from_type(
            RockUnitProperties, _DEFAULT_FAULT_ZONE_PROPERTIES)),
    ])


@wbgeo_component(identifier='wbgeo:__internal__hydrothermal_problem_smart_options_to_data',
                 title='hydrothermal_problem_smart_options_to_data')
def hydrothermal_problem_smart_options_to_data(_input: SmartInputFormData) -> HydrothermalOptions:
    """(Internal) component that converts the smart input form data dict to HydrothermalOptions."""
    return smartcontrols.form_data_to_wbgeo_type_data(_input, HydrothermalOptions)


SmartHydrothermalOptions = typing.Annotated[
    HydrothermalOptions, SmartInput(inputs=['geomodel_result'],
                                    to_form=hydrothermal_problem_smart_options,
                                    to_data=hydrothermal_problem_smart_options_to_data)]


# -----------------------------------------------------------------------------
# GOLEM smart options: same Workbench pattern as SfePy
# -----------------------------------------------------------------------------

_DEFAULT_GOLEM_FAULT_PROPERTIES = GolemFaultProperties(name="fault")


class GolemHydrothermalOptions_Root(pydantic.BaseModel):
    fluid: FluidProperties = FluidProperties()
    rock_properties: Dict[str, RockUnitProperties] = {}
    # One template is enough: GolemHydrothermalProblemBuilder duplicates it
    # across multiple exported 2D fault blocks when necessary.
    fault_properties: GolemFaultProperties = _DEFAULT_GOLEM_FAULT_PROPERTIES


@wbgeo_type(name='GolemHydrothermalOptions', color=palette.PARAMETER,
            identifier='wbgeo::GolemHydrothermalOptions')
class GolemHydrothermalOptions(pydantic.BaseModel):
    root: GolemHydrothermalOptions_Root


@wbgeo_component(identifier='wbgeo:__internal__golem_hydrothermal_problem_smart_options',
                 title='golem_hydrothermal_problem_smart_options')
def golem_hydrothermal_problem_smart_options(geomodel_result: StructuralModelResults) -> CtrlGroup:
    """(Internal) smart form for GOLEM fluid/rock/fault properties."""
    try:
        rock_units = enumerate_rock_units(geomodel_result)
    except Exception:
        rock_units = []

    return CtrlGroup(id='root', inner=[
        CtrlLabel(label='Fluid Properties:'),
        CtrlGroup(id='fluid', inner=smartcontrols.smart_group_from_type(
            FluidProperties, FluidProperties())),
        CtrlLabel(label='Rock Unit Properties:'),
        CtrlGroup(id='rock_properties', inner=[
            CtrlGroup(id=name, inner=[CtrlLabel(label=name)] + smartcontrols.smart_group_from_type(
                RockUnitProperties, RockUnitProperties(name=name)))
            for _, name in rock_units
        ]),
        CtrlLabel(label='Fault Properties (used when the exported mesh contains 2D fault blocks):'),
        CtrlGroup(id='fault_properties', inner=smartcontrols.smart_group_from_type(
            GolemFaultProperties, _DEFAULT_GOLEM_FAULT_PROPERTIES)),
    ])


@wbgeo_component(identifier='wbgeo:__internal__golem_hydrothermal_problem_smart_options_to_data',
                 title='golem_hydrothermal_problem_smart_options_to_data')
def golem_hydrothermal_problem_smart_options_to_data(
    _input: SmartInputFormData,
) -> GolemHydrothermalOptions:
    return smartcontrols.form_data_to_wbgeo_type_data(_input, GolemHydrothermalOptions)


SmartGolemHydrothermalOptions = typing.Annotated[
    GolemHydrothermalOptions,
    SmartInput(inputs=['geomodel_result'],
               to_form=golem_hydrothermal_problem_smart_options,
               to_data=golem_hydrothermal_problem_smart_options_to_data),
]


@wbgeo_component(
    description="Builds the SfePy hydrothermal problem definition (rock/fluid "
                "properties, boundary conditions, and solver settings) from a "
                "mesh and its structural model. Use the smart options to set "
                "per-rock-unit and fluid properties; check 'Enable Fault Zone' "
                "to give cells near any fault their own separate material "
                "(from the options' Fault Zone Properties group) instead of "
                "treating them as ordinary lithology.",
    title='Build SfePy Hydrothermal Problem',
    color=palette.SIMULATION_BUILDER,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=990,
    identifier='wbgeo::simulation_build_hydrothermal_problem',
    return_name='hydrothermal_problem',
    input_checks=[check_mesh_has_known_type, check_mesh_has_lithology_mapping],
)
def build_hydrothermal_problem(
    mesh_results: MeshResults,
    geomodel_result: StructuralModelResults,
    options: SmartHydrothermalOptions = None,
    include_flow: bool = True,
    enable_fault_zone: bool = False,
    fault_zone_n_voxels: int = 1,
    # Saved Workbench pipelines store input values by position: keep the
    # existing order (start_time/end_time took t0/t1's places) and only
    # ever append new inputs at the end.
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    num_steps: int = 3,
    p_top: float = 0.0,
    p_bottom: float = 1.0e6,
    t_top: float = 10.0,
    t_bottom: float = 60.0,
    linear_solver: str = "iterative",
    linear_solver_i_max: int = 500,
    linear_solver_eps_r: float = 1e-8,
    initial_temperature: Optional[float] = None,
    print_linear_residuals: bool = False,
) -> SfepyProblem:
    """
    :param mesh_results: The mesh to solve on (implicit, structured, or unstructured).
    :param geomodel_result: The structural model mesh_results was generated from.
    :param options: Smart options for per-rock-unit, fluid, and fault-zone properties.
    :param include_flow: If False, skips the pressure/Darcy stage and solves pure heat conduction.
    :param enable_fault_zone: If True, gives cells near any fault their own material
        (options' Fault Zone Properties), using fault_zone_n_voxels as the damage-zone width.
        Ignored (no fault zone) if the structural model has no faults.
    :param fault_zone_n_voxels: Fault damage-zone width in structural-grid voxels.
    :param p_top: Pressure Dirichlet boundary value at the model top (Pa).
    :param p_bottom: Pressure Dirichlet boundary value at the model bottom (Pa).
    :param t_top: Temperature Dirichlet boundary value at the model top (°C).
    :param t_bottom: Temperature Dirichlet boundary value at the model bottom (°C).
    :param initial_temperature: Constant initial temperature (°C). Leave empty for a
        linear profile between t_bottom and t_top.
    :param start_time: Simulation start time (s).
    :param end_time: Simulation end time (s). Leave empty for the model's thermal
        diffusion timescale (roughly the time to reach thermal equilibrium).
    :param num_steps: Number of uniform time steps between start_time and end_time.
    :param print_linear_residuals: Include the linear solver residuals in the solver output.
    :param linear_solver: "direct" or "iterative" (see SfepyHydrothermalProblemBuilder.LINEAR_SOLVERS).
    :param linear_solver_i_max: Max iterations for the iterative linear solver.
    :param linear_solver_eps_r: Relative residual tolerance for the iterative linear solver.
    :return: The problem definition, wrapped in the shared SfepyProblem envelope so it can
        connect into the same Run SfePy Simulation component as Build SfePy Custom Problem's output.

    Raises:
        ValueError: mesh_results has no known mesh_type, or no lithology mapping
            (cell_data['block_id']) -- see check_mesh_has_known_type/
            check_mesh_has_lithology_mapping for details.
    """
    # Pre-checks can be ignored/skipped by a caller (see docs/developers/components.md),
    # so also enforced here rather than relying solely on the Workbench's
    # input_checks catching the connection.
    check_mesh_has_known_type(mesh_results)
    check_mesh_has_lithology_mapping(mesh_results)

    rock_properties = options.root.rock_properties if options and options.root.rock_properties else None
    fluid = options.root.fluid if options else None
    if enable_fault_zone:
        fault_zone_properties = options.root.fault_zone_properties if options else _DEFAULT_FAULT_ZONE_PROPERTIES
    else:
        fault_zone_properties = None

    return SfepyProblem(hydrothermal=SfepyHydrothermalProblemBuilder(
        mesh_results=mesh_results,
        geomodel_result=geomodel_result,
        rock_properties=rock_properties,
        fluid=fluid,
        include_flow=include_flow,
        fault_zone_properties=fault_zone_properties,
        fault_zone_n_voxels=fault_zone_n_voxels,
        p_top=p_top,
        p_bottom=p_bottom,
        t_top=t_top,
        t_bottom=t_bottom,
        initial_temperature=initial_temperature,
        start_time=start_time,
        end_time=end_time,
        num_steps=num_steps,
        print_linear_residuals=print_linear_residuals,
        linear_solver=linear_solver,
        linear_solver_i_max=linear_solver_i_max,
        linear_solver_eps_r=linear_solver_eps_r,
    ))


@wbgeo_component(
    description="Wraps a user-supplied, already-complete SfePy input file "
                "(with its own internal staging/sequencing logic) instead of "
                "auto-generating one -- for bringing an existing hand-written "
                "SfePy problem into the Workbench. Validated as well as "
                "possible against the mesh at build time (known mesh type, "
                "known lithology mapping, and every 'cells of group N' the "
                "file references actually exists on the mesh), but this "
                "cannot catch wrong material properties or malformed "
                "equations -- sfepy-run's own region resolution remains the "
                "final authority at run time.",
    title='Build SfePy Custom Problem',
    color=palette.SIMULATION_BUILDER,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=980,
    identifier='wbgeo::simulation_build_custom_sfepy_problem',
    return_name='sfepy_problem',
    input_checks=[check_mesh_has_known_type, check_mesh_has_lithology_mapping, check_mesh_has_no_engineering_objects],
    # Explicit override, kept defensively even after switching input_file
    # away from BasicallyABufferedFile (see below): the framework appeared
    # to auto-infer is_object_type=True (a "loading component", presented
    # collapsed into a single node -- see docs/developers/components.md's
    # is_object_type row) purely from the presence of a
    # BasicallyABufferedFile parameter, confirmed live via
    # objectType=true/typeMagic=2 vs Build SfePy Hydrothermal Problem's
    # objectType=false/typeMagic=0. Forcing False matches Build
    # Hydrothermal Problem's normal (non-collapsed) presentation.
    is_object_type=False,
)
def build_custom_sfepy_problem(
    input_file: CustomSfepyFileDataType,
    mesh_results: MeshResults,
    geomodel_result: Optional[StructuralModelResults] = None,
    fault_zone_n_voxels: Optional[int] = None,
) -> SfepyProblem:
    """
    :param input_file: Path (relative to the codebase root, e.g.
        'examples/own_data/my_problem.py') to a complete SfePy input file (.py) with its own
        staging/sequencing logic. Picked via the RemoteFile browser widget -- place the file
        under own_data/ (bind-mounted for local development; see docker-compose.yml) so it's
        visible to pick from.
    :param mesh_results: The mesh to solve on (implicit, structured, or unstructured). Meshes
        with wells/sources are rejected -- not supported yet.
    :param geomodel_result: Optional structural model, used for an additional soft sanity
        check (referenced-group count vs. lithology count), and required if
        fault_zone_n_voxels is set.
    :param fault_zone_n_voxels: If set, gives cells within this many structural-grid voxels
        of any active fault's trace their own group id (see the returned SfepyCustomBuilder's
        fault_group_id) -- the same "damage zone" cell selection Build SfePy Hydrothermal Problem
        uses, referenceable from the custom file as 'cells of group N'. No material/equation
        is auto-generated for it; that's entirely up to the custom file. None (default)
        leaves only the mesh's real lithology groups available, exactly like today.
    :return: The custom problem definition, wrapped in the shared SfepyProblem envelope so it
        can connect into the same Run SfePy Simulation component as Build SfePy Hydrothermal Problem's
        output. Not restricted to hydrothermal physics -- the file can define any SfePy
        physics; only the mesh/region bookkeeping is validated, not the equations themselves.

    Raises:
        ValueError: input_file doesn't exist or isn't valid UTF-8 text; mesh_results has no
            known mesh_type, no lithology mapping, or contains a well/source block; the file
            references a 'cells of group N' that doesn't exist on mesh_results; or
            fault_zone_n_voxels is set with no geomodel_result.
    """
    # Pre-checks can be ignored/skipped by a caller (see docs/developers/components.md),
    # so also enforced here rather than relying solely on the Workbench's
    # input_checks catching the connection.
    check_mesh_has_known_type(mesh_results)
    check_mesh_has_lithology_mapping(mesh_results)
    check_mesh_has_no_engineering_objects(mesh_results)

    # input_file is a RemoteFile-controlled path, not an uploaded byte
    # stream (see CustomSfepyFileDataType) -- resolved the same way
    # core.loading_components.geo_input_data resolves its own
    # RemoteFile-controlled paths, relative to the codebase root.
    datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    full_path = os.path.join(datadir, input_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError as e:
        raise ValueError(f"Custom SfePy input file not found: {full_path}") from e
    except UnicodeDecodeError as e:
        raise ValueError("Custom SfePy input file is not valid UTF-8 text.") from e

    return SfepyProblem(custom=SfepyCustomBuilder(
        input_file_contents=content, mesh_results=mesh_results, geomodel_result=geomodel_result,
        fault_zone_n_voxels=fault_zone_n_voxels,
    ))


@wbgeo_component(
    description='Runs the given problem definition (from either Build SfePy '
                'Hydrothermal Problem or Build SfePy Custom Problem) -- the '
                'two-stage SfePy hydrothermal solve (steady-state pressure, '
                'then transient advective-conductive heat transport) for a '
                'Hydrothermal problem, or a single sfepy-run invocation for a '
                'Custom SfePy problem.',
    title='Run SfePy Simulation',
    color=palette.SIMULATION_RUN,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=890,
    identifier='wbgeo::simulation_run',
    return_name='simulation_result',
)
def run_simulation(problem: SfepyProblem) -> SimulationResults:
    """
    :param problem: Problem definition from Build SfePy Hydrothermal Problem or Build SfePy Custom Problem.
    :return: Simulation results for every solved/saved time step.
    """
    return _run_simulation_sfepy(problem.inner())


# -----------------------------------------------------------------------------
# GOLEM problem builders / runner -- same Workbench pattern as SfePy
# -----------------------------------------------------------------------------

@wbgeo_component(
    description="Builds the GOLEM hydrothermal problem definition from a mesh "
                "and structural model. Use the smart options to set the same "
                "per-rock-unit and fluid properties, plus a fault-property "
                "template for exported 2D fault blocks. This only prepares "
                "the problem; Docker/GOLEM is started by Run GOLEM Simulation.",
    title='Build GOLEM Hydrothermal Problem',
    color=palette.SIMULATION_BUILDER,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=970,
    identifier='wbgeo::simulation_build_golem_hydrothermal_problem',
    return_name='golem_problem',
    input_checks=[check_mesh_has_known_type, check_mesh_has_lithology_mapping],
)
def build_golem_hydrothermal_problem(
    mesh_results: MeshResults,
    geomodel_result: StructuralModelResults,
    options: SmartGolemHydrothermalOptions = None,
    p_top: float = 0.0,
    p_bottom: float = 1.0e6,
    t_top: float = 10.0,
    t_bottom: float = 60.0,
    initial_pressure: Optional[float] = None,
    initial_temperature: Optional[float] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    num_steps: int = 3,
    output_interval: int = 1,
    print_linear_residuals: bool = False,
) -> GolemProblem:
    """
    :param mesh_results: The WBGeo mesh to solve on (implicit, structured, or unstructured).
    :param geomodel_result: Structural model associated with the mesh.
    :param options: Smart GOLEM fluid/rock/fault properties.
    :param p_top: Pressure Dirichlet boundary value at the model top (Pa).
    :param p_bottom: Pressure Dirichlet boundary value at the model bottom (Pa).
    :param t_top: Temperature Dirichlet boundary value at the model top (°C).
    :param t_bottom: Temperature Dirichlet boundary value at the model bottom (°C).
    :param initial_pressure: Constant initial pressure (Pa). Leave empty for a linear
        profile between p_bottom and p_top.
    :param initial_temperature: Constant initial temperature (°C). Leave empty for a
        linear profile between t_bottom and t_top (as in the SfePy hydrothermal builder).
    :param start_time: Simulation start time (s).
    :param end_time: Simulation end time (s). Leave empty for the model's thermal
        diffusion timescale (roughly the time to reach thermal equilibrium).
    :param num_steps: Number of uniform time steps between start_time and end_time.
    :param output_interval: Write results every output_interval time steps.
    :param print_linear_residuals: Include GOLEM's linear residuals in the solver output.
    :return: GOLEM problem envelope for Run GOLEM Simulation.
    """
    check_mesh_has_known_type(mesh_results)
    check_mesh_has_lithology_mapping(mesh_results)

    rock_properties = options.root.rock_properties if options and options.root.rock_properties else None
    fluid = options.root.fluid if options else None
    fault_properties = options.root.fault_properties if options else _DEFAULT_GOLEM_FAULT_PROPERTIES

    return GolemProblem(hydrothermal=GolemHydrothermalProblemBuilder(
        mesh_results=mesh_results,
        geomodel_result=geomodel_result,
        rock_properties=rock_properties,
        fault_properties=fault_properties,
        fluid=fluid,
        p_top=p_top,
        p_bottom=p_bottom,
        t_top=t_top,
        t_bottom=t_bottom,
        initial_pressure=initial_pressure,
        initial_temperature=initial_temperature,
        start_time=start_time,
        end_time=end_time,
        num_steps=num_steps,
        output_interval=output_interval,
        print_linear_residuals=print_linear_residuals,
    ))


@wbgeo_component(
    description="Builds the GOLEM simple-reservoir model for any structural model large "
                "enough for the layout. Unlike the GOLEM hydrothermal model, this builder "
                "also creates its own unstructured mesh: a doublet of two square fracture "
                "planes (default 100 m, 200 m apart) and a well, centred in the model "
                "(or at layout_center), with refinement in metres around them and a "
                "background mesh size that follows the model size. This only prepares the "
                "problem; Docker/GOLEM is started by Run GOLEM Simulation.",
    title='Build GOLEM Simple Reservoir Problem',
    color=palette.SIMULATION_BUILDER,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=950,
    identifier='wbgeo::simulation_build_golem_simple_reservoir_problem',
    return_name='golem_problem',
)
def build_golem_simple_reservoir_problem(
    geomodel_result: StructuralModelResults,
    mesh_size: Optional[float] = None,
    curve_mesh_size: float = 5.0,
    mapping_litho: LithoMappingModeType = LithoMappingMode.AUTOMATIC_CENTERS.value,
    well_refine_size_min: float = 5.0,
    well_refine_size_max: float = 30.0,
    well_refine_dist_min: float = 10.0,
    well_refine_dist_max: float = 60.0,
    fracture_refine_size_min: float = 5.0,
    fracture_refine_size_max: float = 30.0,
    fracture_refine_dist_min: float = 10.0,
    fracture_refine_dist_max: float = 60.0,
    rock_porosity: float = 0.1,
    rock_permeability: float = 1.0e-15,
    rock_density: float = 2.6e3,
    rock_thermal_conductivity: float = 3.0,
    rock_heat_capacity: float = 950.0,
    fracture_porosity: float = 1.0,
    fracture_aperture: float = 1.0e-4,
    geological_fault_aperture: float = 1.0,
    well_porosity: float = 1.0,
    well_permeability: float = 3.12e-4,
    well_scaling_factor: float = 0.05,
    fluid_density: float = 1.0e3,
    fluid_viscosity: float = 1.0e-3,
    fluid_heat_capacity: float = 4.2e3,
    fluid_thermal_conductivity: float = 0.65,
    fluid_modulus: float = 2.5e9,
    initial_pressure: float = 1.0e7,
    initial_temperature: float = 150.0,
    boundary_pressure: float = 1.0e7,
    boundary_temperature: float = 150.0,
    injection_temperature: float = 70.0,
    injection_rate: float = 3.0,
    extraction_rate: float = 3.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    # Appended inputs: saved pipelines store values by position.
    fracture_size: float = 100.0,
    fracture_spacing: float = 200.0,
    layout_center: Optional[AnnotatedPoint] = None,
    injection_point: Optional[AnnotatedPoint] = None,
    extraction_point: Optional[AnnotatedPoint] = None,
    wells: Optional[WellData] = None,
) -> GolemProblem:
    """
    The reservoir mesh fills the structural model's extent. A doublet of physical size is
    placed in it: two square fracture planes (fracture_size, default 100 m) fracture_spacing
    apart (default 200 m) around layout_center (default: the model centre), and a well at
    the second plane; injection and extraction act at the plane centres. The layout plus
    its refinement zone must fit into the model on every axis. All rock units get the same
    rock properties.

    :param geomodel_result: Structural model whose extent the reservoir mesh fills.
    :param mesh_size: Target unstructured volume-mesh size (m). Leave empty for 3 % of the
        model's smallest axis (30 m for a 1 km model).
    :param curve_mesh_size: Target mesh size on the well curve (m).
    :param mapping_litho: Lithology mapping mode passed to the unstructured mesher.
    :param well_refine_size_min: Smallest element size next to the well (m).
    :param well_refine_size_max: Element size beyond well_refine_dist_max (m).
    :param well_refine_dist_min: Distance from the well up to which well_refine_size_min applies (m).
    :param well_refine_dist_max: Distance from the well at which well_refine_size_max is reached (m).
    :param fracture_refine_size_min: Smallest element size next to the fractures (m).
    :param fracture_refine_size_max: Element size beyond fracture_refine_dist_max (m).
    :param fracture_refine_dist_min: Distance from the fractures up to which fracture_refine_size_min applies (m).
    :param fracture_refine_dist_max: Distance from the fractures at which fracture_refine_size_max is reached (m).
        The eight refinement values are metres around the fractures and the well,
        independent of the model size.
    :param rock_porosity: Porosity of the rock (-).
    :param rock_permeability: Permeability of the rock (m²).
    :param rock_density: Density of the rock (kg/m³).
    :param rock_thermal_conductivity: Thermal conductivity of the rock (W/(m·K)).
    :param rock_heat_capacity: Specific heat capacity of the rock (J/(kg·K)).
    :param fracture_porosity: Porosity of the fracture elements (-).
    :param fracture_aperture: Aperture of the fracture elements (m).
    :param geological_fault_aperture: Aperture given to faults of the structural model (m);
        they otherwise use the rock properties.
    :param well_porosity: Porosity of the well elements (-).
    :param well_permeability: Permeability of the well elements (m²).
    :param well_scaling_factor: Initial scaling factor of the 1D well elements in GOLEM.
    :param fluid_density: Fluid density (kg/m³).
    :param fluid_viscosity: Fluid viscosity (Pa·s).
    :param fluid_heat_capacity: Specific heat capacity of the fluid (J/(kg·K)).
    :param fluid_thermal_conductivity: Thermal conductivity of the fluid (W/(m·K)).
    :param fluid_modulus: Fluid bulk modulus (Pa).
    :param initial_pressure: Initial reservoir pressure (Pa).
    :param initial_temperature: Initial reservoir temperature (°C).
    :param boundary_pressure: Pressure held at the model boundary (Pa).
    :param boundary_temperature: Temperature held at the model boundary (°C).
    :param injection_temperature: Temperature of the injected fluid (°C).
    :param injection_rate: Injection rate, passed to GOLEM's point source (constant_rate).
    :param extraction_rate: Extraction rate, passed to GOLEM's point source (constant_rate).
    :param start_time: Simulation start time (s).
    :param end_time: Simulation end time (s). Leave empty to end at the last solve time of
        the builder's default time sequence.
    :param fracture_size: Edge length of the two square fracture planes (m).
    :param fracture_spacing: Distance between the two fracture planes along x (m).
    :param layout_center: Centre of the fracture pair. Leave empty for the model centre.
    :param injection_point: Injection point. Leave empty for the centre of the first fracture.
    :param extraction_point: Extraction point. Leave empty for the centre of the second fracture.
    :param wells: Well path(s), e.g. from Load Wells. Leave empty for the default well at the
        second fracture.
    :return: GOLEM problem envelope for Run GOLEM Simulation.
    """
    if isinstance(wells, dict):
        raise ValueError("The simple reservoir takes a plain list of wells; hierarchical (keyed) "
                         "well data from Load Wells is not supported here.")
    return GolemProblem(simple_reservoir=GolemSimpleReservoirBuilder(
        geomodel_result=geomodel_result,
        mesh_size=mesh_size,
        curve_mesh_size=curve_mesh_size,
        mapping_litho=mapping_litho,
        well_refine_size_min=well_refine_size_min,
        well_refine_size_max=well_refine_size_max,
        well_refine_dist_min=well_refine_dist_min,
        well_refine_dist_max=well_refine_dist_max,
        fracture_refine_size_min=fracture_refine_size_min,
        fracture_refine_size_max=fracture_refine_size_max,
        fracture_refine_dist_min=fracture_refine_dist_min,
        fracture_refine_dist_max=fracture_refine_dist_max,
        rock_porosity=rock_porosity,
        rock_permeability=rock_permeability,
        rock_density=rock_density,
        rock_thermal_conductivity=rock_thermal_conductivity,
        rock_heat_capacity=rock_heat_capacity,
        fracture_porosity=fracture_porosity,
        fracture_aperture=fracture_aperture,
        geological_fault_aperture=geological_fault_aperture,
        well_porosity=well_porosity,
        well_permeability=well_permeability,
        well_scaling_factor=well_scaling_factor,
        fluid_density=fluid_density,
        fluid_viscosity=fluid_viscosity,
        fluid_heat_capacity=fluid_heat_capacity,
        fluid_thermal_conductivity=fluid_thermal_conductivity,
        fluid_modulus=fluid_modulus,
        initial_pressure=initial_pressure,
        initial_temperature=initial_temperature,
        boundary_pressure=boundary_pressure,
        boundary_temperature=boundary_temperature,
        injection_temperature=injection_temperature,
        injection_rate=injection_rate,
        extraction_rate=extraction_rate,
        start_time=start_time,
        end_time=end_time,
        fracture_size=fracture_size,
        fracture_spacing=fracture_spacing,
        layout_center=layout_center,
        injection_point=injection_point,
        extraction_point=extraction_point,
        wells_override=wells,
    ))


@wbgeo_component(
    description="Wraps a user-supplied complete GOLEM/MOOSE input file (.i) "
                "instead of auto-generating one. The input must use "
                "${GOLEM_MESH_FILE} in its FileMeshGenerator block.",
    title='Build GOLEM Custom Problem',
    color=palette.SIMULATION_BUILDER,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=960,
    identifier='wbgeo::simulation_build_custom_golem_problem',
    return_name='golem_problem',
    input_checks=[check_mesh_has_known_type],
    is_object_type=False,
)
def build_custom_golem_problem(
    input_file: CustomGolemFileDataType,
    mesh_results: MeshResults,
    geomodel_result: Optional[StructuralModelResults] = None,
    print_linear_residuals: bool = True,
) -> GolemProblem:
    """
    :param input_file: Path (relative to the codebase root) to a complete GOLEM/MOOSE input
        file (.i), picked via the file browser. Its FileMeshGenerator must use
        ${GOLEM_MESH_FILE}, which is replaced by the exported mesh_results.
    :param mesh_results: The WBGeo mesh to solve on (implicit, structured, or unstructured).
    :param geomodel_result: Optional structural model associated with the mesh.
    :param print_linear_residuals: Include GOLEM's linear residuals in the solver output.
    :return: GOLEM problem envelope for Run GOLEM Simulation.
    """
    check_mesh_has_known_type(mesh_results)

    datadir = pathlib.Path(__file__).parent.parent.parent.resolve().as_posix()
    full_path = os.path.join(datadir, input_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError as e:
        raise ValueError(f"Custom GOLEM input file not found: {full_path}") from e
    except UnicodeDecodeError as e:
        raise ValueError("Custom GOLEM input file is not valid UTF-8 text.") from e

    return GolemProblem(custom=GolemCustomBuilder(
        input_file_contents=content,
        mesh_results=mesh_results,
        geomodel_result=geomodel_result,
        print_linear_residuals=print_linear_residuals,
    ))


@wbgeo_component(
    description='Runs the given GOLEM problem definition in the configured Docker image. '
                'The image/project paths and MPI process count are explicit because GOLEM '
                'is external to the WBGeo Python environment.',
    title='Run GOLEM Simulation',
    color=palette.SIMULATION_RUN,
    border_color=palette.BORDER,
    group='Simulation',
    positionInGroup=880,
    identifier='wbgeo::simulation_run_golem',
    return_name='simulation_result',
)
def run_golem_simulation(
    problem: GolemProblem,
    nproc: int = 4,
) -> SimulationResults:
    """Run GOLEM with an isolated output directory for this execution."""
    run_output_dir = pathlib.Path.cwd() / "golem_output" / uuid.uuid4().hex
    workspace_dir = os.environ.get('GOLEM_WORKSPACE', 'golem_workspace') + "/" + str(uuid.uuid4())
    try:
        # the `registry.git.rwth-aachen.de/wbgeo/container-registry/pyrunner-base` image comes with golem already installed
        # it is based upon `moose-dev:fixed`
        golem_image = os.environ.get('GOLEM_IMAGE', None)

        golem_binary = os.environ.get('GOLEM_BINARY', '/golem/golem-opt')

        return _run_simulation_golem(
            problem=problem.inner(),
            image_name=golem_image,
            golem_binary=golem_binary,
            workspace_dir=workspace_dir,
            output_dir=str(run_output_dir),
            nproc=nproc,
            temporary=True,
            stream_output=True,
        )
    finally:
        # The result holds everything later nodes need (fields, CSV tables,
        # input, solver output), so GOLEM's output folder can go right away,
        # whether the run succeeded or not.
        shutil.rmtree(run_output_dir, ignore_errors=True)
        shutil.rmtree(workspace_dir, ignore_errors=True)


@wbgeo_component(
    description='Exports a SfePy or GOLEM result as a ZIP: the fields as VTK files with a ParaView '
                'collection or as one Exodus file (auto: Exodus for GOLEM, VTK otherwise), plus the '
                "solver's CSV tables, input file(s) and console output.",
    title='Export Simulation Results',
    group='Simulation',
    positionInGroup=0,
    identifier='wbgeo::export_simulation_results',
    return_name='file',
    color=palette.SIMULATION_EXPORT,
    border_color=palette.BORDER,
)
def export_simulation_results(
    simulation_result: SimulationResults,
    format: SimulationExportFormatType = SimulationExportFormat.AUTO.value,
) -> BasicallyABufferedFile:
    """
    :param simulation_result: Simulation results from Run SfePy Simulation or Run GOLEM Simulation.
    :param format: "auto" (Exodus for GOLEM, VTK otherwise), "vtk" or "exodus".
    :return: A ZIP with the results plus the solver's tables (CSV), input file(s) and console output.
    """
    return _export_simulation_results(simulation_result, format)


# -----------------------------------------------------------------------------
# Inspectors -- mirrors StructuralModelResults'/MeshResults' plot inspectors
# (structural_workbench_components.py / meshing_visualization.py)
# -----------------------------------------------------------------------------

@wbgeo_component(identifier='wbgeo::inspect_sfepy_problem_plot_materials',
                 title='Plot Materials',
                 description='Plots the mesh colored one solid color per material '
                             '(lithologies in their structural-model colors, plus the '
                             'fault zone if active), with a legend grouped like the '
                             'structural model plot, or colored by a numeric material '
                             'property -- a pre-flight check to catch a wrong '
                             'material/fault-zone assignment before running the '
                             '(possibly slow) solve. Same preview as Plot GOLEM Materials.')
@wbgeo_inspector()
def inspect_sfepy_problem_plot_materials(
    problem: SfepyProblem,
    var_name: Optional[str] = None,
):
    """Leave var_name empty for material groups; enter a numeric material property to color by value."""
    plot_builder_materials(problem.inner(), property_name=var_name, cmap= "viridis")


@wbgeo_component(identifier='wbgeo::inspect_golem_problem_plot_materials',
                 title='Plot GOLEM Materials',
                 description='Preview GOLEM rock, fault, fracture and well material blocks '
                             'before running the solver, with the same colors and legend '
                             'as Plot Materials for SfePy. Custom input files must assign '
                             'every material to explicit blocks; otherwise the preview '
                             'reports what it cannot resolve.')
@wbgeo_inspector()
def inspect_golem_problem_plot_materials(
    problem: GolemProblem,
    var_name: Optional[str] = None,
):
    """Leave var_name empty for material blocks; enter a numeric material property to color by value."""
    plot_builder_materials(problem.inner(), property_name=var_name, cmap= "viridis")


def _saved_time_for_step(simulation_result: SimulationResults, time_step: int) -> float:
    """Translate a zero-based saved-result index into its original simulation-time key."""
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("Simulation has no saved time steps.")
    if not isinstance(time_step, int) or isinstance(time_step, bool) or not 0 <= time_step < len(times):
        available = ", ".join(f"step {i}: {t:g}" for i, t in enumerate(times))
        raise ValueError(
            f"time_step must be an integer from 0 to {len(times) - 1} "
            f"({len(times)} saved results); got {time_step!r}. "
            f"Saved time steps (index: simulation time): {available}."
        )
    return times[time_step]


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_plot_variable_at_a_time',
                 title='Plot Variable (Time Step)',
                 description='Plot a simulation variable at the selected zero-based saved time step. '
                             'Defaults to the final saved step. The time_step input is an index, '
                             'not physical simulation time.')
@wbgeo_inspector()
def inspect_simulation_result_plot_variable_at_a_time(
    simulation_result: SimulationResults,
    var_name: Optional[str] = None,
    time_step: Optional[int] = None,
):
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("Simulation has no saved time steps.")
    physical_time = times[-1] if time_step is None else _saved_time_for_step(simulation_result, time_step)
    plot_variable_at_a_time(
        simulation_result,
        var_name=var_name,
        time=physical_time,
        cmap="coolwarm", show_edges=False,
        off_screen=True, window_size=(800, 600), anti_aliasing=False,
    )


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_plot_cross_section_2D',
                 title='Plot Cross Section 2D',
                 description='Compare a variable between two zero-based saved time-step indices '
                             '(defaults: the first and last saved steps) on a section normal to the chosen '
                             'axis (x, y or z). position is the coordinate along that axis in model units; '
                             'empty = the middle of the mesh.')
@wbgeo_inspector()
def inspect_simulation_result_plot_cross_section_2D(
    simulation_result: SimulationResults,
    _inspector: InspectorHelper,
    var_name: Optional[str] = None,
    time_step1: Optional[int] = None,
    time_step2: Optional[int] = None,
    axis: SliceAxisType = SliceAxis.Y.value,
    position: Optional[float] = None,
):
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("Simulation has no saved time steps.")
    # Axis-aligned section like the structural 2D plots; plot_cross_section_2D itself also
    # takes tilted planes (any origin/normal) for use in Python.
    dim = "xyz".index(SliceAxis(axis).value)
    normal = tuple(1.0 if i == dim else 0.0 for i in range(3))
    nodes = simulation_result.nodes_by_time[times[0]]
    # The mesh centroid keeps the plane inside the domain for any model extent.
    origin = [float(v) for v in nodes.mean(axis=0)]
    if position is not None:
        low, high = float(nodes[:, dim].min()), float(nodes[:, dim].max())
        if not low <= position <= high:
            raise ValueError(f"position={position:g} is outside the mesh: along {SliceAxis(axis).value} "
                             f"it must be between {low:g} and {high:g}.")
        origin[dim] = float(position)
    origin = tuple(origin)
    physical_time_a = times[0] if time_step1 is None else _saved_time_for_step(simulation_result, time_step1)
    physical_time_b = times[-1] if time_step2 is None else _saved_time_for_step(simulation_result, time_step2)
    plot_cross_section_2D(
        simulation_result,
        time_a=physical_time_a,
        time_b=physical_time_b,
        var_name=var_name,
        origin=origin,
        normal=normal,
        cmap="coolwarm",
    )

@wbgeo_component(
    identifier='wbgeo::inspect_simulation_result_plot_variable_along_line',
    title='Plot Variable Along Line',
    description='Sample a result variable along a 3D line at a zero-based saved time step. '
                'Line endpoints are model coordinates. Defaults: the last saved step and a '
                'vertical line through the mesh centre, from the bottom to the top of the mesh.',
)
@wbgeo_inspector()
def inspect_simulation_result_plot_variable_along_line(
    simulation_result: SimulationResults,
    _inspector: InspectorHelper,
    var_name: Optional[str] = None,
    time_step: Optional[int] = None,
    p0: Optional[AnnotatedPoint] = None,
    p1: Optional[AnnotatedPoint] = None,
):
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("The simulation result has no saved time steps.")
    physical_time = (times[-1] if time_step is None
                     else _saved_time_for_step(simulation_result, time_step))
    # Default line: vertical through the mesh centre, like a virtual borehole.
    nodes = simulation_result.nodes_by_time[physical_time]
    centre = nodes.mean(axis=0)
    if p0 is None:
        p0 = (float(centre[0]), float(centre[1]), float(nodes[:, 2].min()))
    if p1 is None:
        p1 = (float(centre[0]), float(centre[1]), float(nodes[:, 2].max()))
    plot_variable_along_line(
        simulation_result,
        var_name=var_name,
        time=physical_time,
        p0=p0,
        p1=p1,
    )


@wbgeo_component(
    identifier='wbgeo::inspect_simulation_result_plot_variable_time_series',
    title='Plot Variable Time Series',
    description='Plot a variable at a 3D point over saved time steps, optionally limiting '
                'the range with zero-based inclusive start/end step indices.',
)
@wbgeo_inspector()
def inspect_simulation_result_plot_variable_time_series(
    simulation_result: SimulationResults,
    _inspector: InspectorHelper,
    var_name: Optional[str] = None,
    point: Optional[AnnotatedPoint] = None,
    time_step_start: Optional[int] = None,
    time_step_end: Optional[int] = None,
):
    # Default to the mesh centroid (as the cross-section inspector does), so
    # leaving the Point input empty samples inside the domain instead of
    # failing on a missing point.
    if point is None:
        times = sorted(simulation_result.nodes_by_time)
        if times:
            point = tuple(float(v) for v in simulation_result.nodes_by_time[times[0]].mean(axis=0))
    plot_variable_time_series(
        simulation_result,
        var_name=var_name,
        point=point,
        time_step_start=time_step_start,
        time_step_end=time_step_end,
    )


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_stdout',
                 title='View Solver Output',
                 description='Prints the raw solver output for this result, per stage '
                             '("pressure"/"heat" for a SfePy hydrothermal problem, "custom" for a '
                             'custom SfePy problem, "golem" for GOLEM) -- lets solver '
                             'convergence/quality be checked on a normal, successful run too, not '
                             'only via the error raised when a solve fails to converge.')
@wbgeo_inspector()
def inspect_simulation_result_stdout(simulation_result: SimulationResults, _inspector: InspectorHelper):
    if not simulation_result.solver_output:
        print("No solver output recorded for this result.")
        return
    for stage, output in simulation_result.solver_output.items():
        print(f"=== {stage} ===")
        print(output)
