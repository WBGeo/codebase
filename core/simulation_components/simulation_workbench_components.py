"""
WBGeo Workbench components for the SfePy pipeline and GOLEM hydrothermal/simple-reservoir pipelines.

Thin-wrapper file only: per the project's "one workbench file per step" rule
(already applied to structural_modeling_components/meshing_components),
@wbgeo_component must appear here and nowhere else in core/simulation_components/.
The actual logic lives in the SfePy/GOLEM builder and runner modules
and is called here unmodified.
"""
import atexit
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

from core.object_components import MeshResults, StructuralModelResults, SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, SfepyProblem, RockUnitProperties, FluidProperties,
    enumerate_rock_units, check_mesh_has_known_type, check_mesh_has_lithology_mapping,
    check_mesh_has_no_engineering_objects,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_run import (
    run_simulation_sfepy as _run_simulation_sfepy,
)
from core.simulation_components.simulation_packages.simulation_export import (
    export_simulation_results as _export_simulation_results,
)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder, GolemFaultProperties, GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_run import (
    run_simulation_golem as _run_simulation_golem,
)
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials,
    plot_variable_along_line, plot_variable_time_series,
)

AnnotatedPoint = typing.Annotated[Tuple[float, float, float], AnnotatedScriptType(identifier='wbgeo::mesh:point', name='Point', controlled='Tuple|X|Y|Z')]

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
    str, AnnotatedScriptType(name='path', color='aqua', identifier='CustomSfepyFileDataType',
                             controlled='RemoteFile|endswith=.py')]

# GOLEM custom inputs use the same RemoteFile pattern, but select MOOSE .i files.
CustomGolemFileDataType = typing.Annotated[
    str, AnnotatedScriptType(name='path', color='aqua', identifier='CustomGolemFileDataType',
                             controlled='RemoteFile|endswith=.i')]


@wbgeo_type(name='GOLEM Problem', color='#e07a5f', identifier='wbgeo::GolemProblem')
class GolemProblem(pydantic.BaseModel):
    """Workbench-visible envelope around one GOLEM problem definition.

    This mirrors SfepyProblem: the Workbench DSL needs one concrete registered
    port type. GOLEM has two generated model families (hydrothermal and simple
    reservoir) plus a custom-input path, and all three connect to the same
    Run GOLEM Simulation component through this envelope.
    """
    hydrothermal: Optional[GolemHydrothermalProblemBuilder] = None
    simple_reservoir: Optional[GolemSimpleReservoirProblemBuilder] = None
    custom: Optional[CustomGolemBuilder] = None

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
    # live), but only actually used by HydrothermalProblemBuilder if
    # enable_fault_zone=True on build_hydrothermal_problem below -- keeping
    # that as a plain checkbox parameter (rather than an in-form toggle)
    # avoids relying on CtrlIf's condition-path scoping, which isn't
    # exercised anywhere else in this codebase against a dict-of-groups
    # layout.
    fault_zone_properties: RockUnitProperties = _DEFAULT_FAULT_ZONE_PROPERTIES


@wbgeo_type(name='HydrothermalOptions', color='#e07a5f',
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


@wbgeo_type(name='GolemHydrothermalOptions', color='#e07a5f',
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
    title='Build Sfepy Hydrothermal Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    positionInGroup=89,
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
    t0: float = 0.0,
    t1: Optional[float] = None,
    num_steps: int = 1,
    p_top: float = 0.0,
    p_bottom: float = 100e4,
    t_top: float = 10.0,
    t_bottom: float = 60.0,
    linear_solver: str = "iterative",
    linear_solver_i_max: int = 500,
    linear_solver_eps_r: float = 1e-8,
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
    :param t0: Heat stage start time (s).
    :param t1: Heat stage end time (s); defaults to a data-driven thermal diffusion timescale if not given.
    :param num_steps: Number of heat-stage time steps.
    :param p_top: Pressure Dirichlet boundary value at the model top (Pa).
    :param p_bottom: Pressure Dirichlet boundary value at the model bottom (Pa).
    :param t_top: Temperature Dirichlet boundary value at the model top (°C).
    :param t_bottom: Temperature Dirichlet boundary value at the model bottom (°C).
    :param linear_solver: "direct" or "iterative" (see HydrothermalProblemBuilder.LINEAR_SOLVERS).
    :param linear_solver_i_max: Max iterations for the iterative linear solver.
    :param linear_solver_eps_r: Relative residual tolerance for the iterative linear solver.
    :return: The problem definition, wrapped in the shared SfepyProblem envelope so it can
        connect into the same Run Simulation component as Build Sfepy Custom Problem's output.

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

    return SfepyProblem(hydrothermal=HydrothermalProblemBuilder(
        mesh_results=mesh_results,
        geomodel_result=geomodel_result,
        rock_properties=rock_properties,
        fluid=fluid,
        include_flow=include_flow,
        fault_zone_properties=fault_zone_properties,
        fault_zone_n_voxels=fault_zone_n_voxels,
        t0=t0,
        t1=t1,
        num_steps=num_steps,
        p_top=p_top,
        p_bottom=p_bottom,
        t_top=t_top,
        t_bottom=t_bottom,
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
    title='Build Sfepy Custom Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    positionInGroup=88,
    identifier='wbgeo::simulation_build_custom_sfepy_problem',
    return_name='sfepy_problem',
    input_checks=[check_mesh_has_known_type, check_mesh_has_lithology_mapping, check_mesh_has_no_engineering_objects],
    # Explicit override, kept defensively even after switching input_file
    # away from BasicallyABufferedFile (see below): the framework appeared
    # to auto-infer is_object_type=True (a "loading component", presented
    # collapsed into a single node -- see docs/developers/components.md's
    # is_object_type row) purely from the presence of a
    # BasicallyABufferedFile parameter, confirmed live via
    # objectType=true/typeMagic=2 vs Build SfepyHydrothermal Problem's
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
        of any active fault's trace their own group id (see the returned CustomSfepyBuilder's
        fault_group_id) -- the same "damage zone" cell selection Build SfepyHydrothermal Problem
        uses, referenceable from the custom file as 'cells of group N'. No material/equation
        is auto-generated for it; that's entirely up to the custom file. None (default)
        leaves only the mesh's real lithology groups available, exactly like today.
    :return: The custom problem definition, wrapped in the shared SfepyProblem envelope so it
        can connect into the same Run Simulation component as Build Sfepy Hydrothermal Problem's
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

    return SfepyProblem(custom=CustomSfepyBuilder(
        input_file_contents=content, mesh_results=mesh_results, geomodel_result=geomodel_result,
        fault_zone_n_voxels=fault_zone_n_voxels,
    ))


@wbgeo_component(
    description='Runs the given problem definition (from either Build Sfepy '
                'Hydrothermal Problem or Build Sfepy Custom Problem) -- the '
                'two-stage SfePy hydrothermal solve (steady-state pressure, '
                'then transient advective-conductive heat transport) for a '
                'Hydrothermal problem, or a single sfepy-run invocation for a '
                'Custom SfePy problem.',
    title='Run Sfepy Simulation',
    color='pink',
    border_color='#000000',
    group='Simulation',
    positionInGroup=80,
    identifier='wbgeo::simulation_run',
    return_name='simulation_result',
)
def run_simulation(problem: SfepyProblem) -> SimulationResults:
    """
    :param problem: Problem definition from Build Sfepy Hydrothermal Problem or Build Sfepy Custom Problem.
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
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    positionInGroup=100,
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
    initial_pressure: Optional[float] = 0,
    initial_temperature: Optional[float] = 0,
    start_time: float = 0.0,
    end_time: Optional[float] = 5e11,
    dt: float = 1.667e11,
    num_steps: int = 3,
    output_interval: int = 1,
    print_linear_residuals: bool = True,
) -> GolemProblem:
    """
    :param mesh_results: The WBGeo mesh to solve on.
    :param geomodel_result: Structural model associated with the mesh.
    :param options: Smart GOLEM fluid/rock/fault properties.
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
        dt=dt,
        num_steps=num_steps,
        output_interval=output_interval,
        print_linear_residuals=print_linear_residuals,
    ))


@wbgeo_component(
    description="Builds the GOLEM simple-reservoir model. Unlike the GOLEM "
                "hydrothermal model, this builder also creates its dedicated "
                "unstructured reservoir mesh with synthetic fractures and a "
                "well before generating the GOLEM/MOOSE input. This only "
                "prepares the problem; Docker/GOLEM is started by Run GOLEM "
                "Simulation.",
    title='Build GOLEM Simple Reservoir Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    positionInGroup=99,
    identifier='wbgeo::simulation_build_golem_simple_reservoir_problem',
    return_name='golem_problem',
)
def build_golem_simple_reservoir_problem(
    geomodel_result: StructuralModelResults,
    mesh_size: float = 30.0,
    curve_mesh_size: float = 5.0,
    mapping_litho: str = "automatic_centers",
    z_threshold: Optional[float] = None,
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
) -> GolemProblem:
    """
    :param geomodel_result: Structural model used to place/scale the reservoir geometry.
    :param mesh_size: Target unstructured volume-mesh size.
    :param curve_mesh_size: Target mesh size on well curves.
    :param mapping_litho: Lithology mapping mode passed to the unstructured mesher.
    :param initial_pressure: Initial reservoir pressure.
    :param initial_temperature: Initial reservoir temperature.
    :param boundary_pressure: Reservoir boundary pressure.
    :param boundary_temperature: Reservoir boundary temperature.
    :param injection_temperature: Injection-fluid temperature.
    :param injection_rate: Injection rate.
    :param extraction_rate: Extraction rate.
    :param start_time: Simulation start time.
    :param end_time: Simulation end time; defaults inside the builder when None.
    :param print_linear_residuals: Include GOLEM linear residual output.
    :return: GOLEM problem envelope for Run GOLEM Simulation.
    """
    return GolemProblem(simple_reservoir=GolemSimpleReservoirProblemBuilder(
        geomodel_result=geomodel_result,
        mesh_size=mesh_size,
        curve_mesh_size=curve_mesh_size,
        mapping_litho=mapping_litho,
        z_threshold=z_threshold,
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
    ))


@wbgeo_component(
    description="Wraps a user-supplied complete GOLEM/MOOSE input file (.i) "
                "instead of auto-generating one. The input must use "
                "${GOLEM_MESH_FILE} in its FileMeshGenerator block.",
    title='Build GOLEM Custom Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    positionInGroup=98,
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
    """Build a CustomGolemBuilder from a RemoteFile-controlled .i input."""
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

    return GolemProblem(custom=CustomGolemBuilder(
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
    color='pink',
    border_color='#000000',
    group='Simulation',
    positionInGroup=90,
    identifier='wbgeo::simulation_run_golem',
    return_name='simulation_result',
)
def run_golem_simulation(
    problem: GolemProblem,
    nproc: int = 4,
) -> SimulationResults:
    """Run GOLEM with an isolated output directory for this execution."""
    run_output_dir = pathlib.Path.cwd() / "golem_output" / uuid.uuid4().hex
    try:
        # the `registry.git.rwth-aachen.de/wbgeo/container-registry/pyrunner-base` image comes with golem already installed
        # it is based upon `moose-dev:fixed`
        golem_image = os.environ.get('GOLEM_IMAGE', None)

        golem_binary = os.environ.get('GOLEM_BINARY', '/golem/golem-opt')
        workspace_dir = os.environ.get('GOLEM_WORKSPACE', 'golem_workspace') + "/" + str(uuid.uuid4())
        result = _run_simulation_golem(
            problem=problem.inner(),
            image_name=golem_image,
            golem_binary=golem_binary,
            workspace_dir=workspace_dir,
            output_dir=str(run_output_dir),
            nproc=nproc,
            temporary=True,
        )
    except BaseException:
        # The GOLEM runner cleans its staging area, not the copied output.
        # Never retain partial output from a failed Workbench execution.
        shutil.rmtree(run_output_dir, ignore_errors=True)
        raise

    # Results, Exodus export and the solver-output inspector may read files
    # after this function returns. Keep them until the Workbench process exits.
    atexit.register(shutil.rmtree, run_output_dir, ignore_errors=True)
    return result


@wbgeo_component(
    description='Exports SfePy as VTK/PVD ZIP and GOLEM as native output ZIP (Exodus, CSV and other output files) by default. Explicit formats: auto, vtk, exodus, golem.',
    title='Export Simulation Results',
    group='Simulation',
    positionInGroup=0,
    identifier='wbgeo::export_simulation_results',
    return_name='file',
)
def export_simulation_results(simulation_result: SimulationResults, format: str = "auto", golem_output_dir: str = "") -> BasicallyABufferedFile:
    """
    :param simulation_result: Simulation results from Run Simulation or Run GOLEM Simulation.
    :param format: "auto" (SfePy VTK ZIP / GOLEM native output ZIP), "vtk", "exodus", or "golem".
    :param golem_output_dir: Optional exact GOLEM output directory, needed when multiple GOLEM runs exist and solver metadata was lost.
    :return: A downloadable file containing the exported results.
    """
    return _export_simulation_results(simulation_result, format, golem_output_dir)


# -----------------------------------------------------------------------------
# Inspectors -- mirrors StructuralModelResults'/MeshResults' plot inspectors
# (structural_workbench_components.py / meshing_visualization.py)
# -----------------------------------------------------------------------------

@wbgeo_component(identifier='wbgeo::inspect_sfepy_problem_plot_materials',
                 title='Plot Materials',
                 description='Plots the mesh colored one solid color per material '
                             '(lithologies, plus the fault zone if active), with a '
                             'legend of the properties each material actually uses '
                             '(bare material names for a Custom SfePy problem, which '
                             'has no structured properties object to draw from) -- a '
                             'pre-flight check to catch a wrong material/fault-zone '
                             'assignment before running the (possibly slow) solve.')
@wbgeo_inspector()
def inspect_sfepy_problem_plot_materials(
    problem: SfepyProblem,
    var_name: Optional[str] = None,
):
    """Leave property_name empty for material groups; enter a numeric material property to color by value."""
    plot_builder_materials(problem.inner(), property_name=var_name, cmap= "viridis")


@wbgeo_component(identifier='wbgeo::inspect_golem_problem_plot_materials',
                 title='Plot GOLEM Materials',
                 description='Preview GOLEM rock, fault, fracture and well material blocks '
                             'before running the solver. Custom input assignments cannot '
                             'be inferred safely and produce an informative error.')
@wbgeo_inspector()
def inspect_golem_problem_plot_materials(
    problem: GolemProblem,
    var_name: Optional[str] = None,
):
    """Leave property_name empty for material blocks; enter a numeric material property to color by value."""
    plot_builder_materials(problem.inner(), property_name=var_name, cmap= "viridis")


def _saved_time_for_step(simulation_result: SimulationResults, time_step: int) -> float:
    """Translate a zero-based saved-result index into its original simulation-time key."""
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("Simulation has no saved time steps.")
    if not isinstance(time_step, int) or isinstance(time_step, bool) or not 0 <= time_step < len(times):
        raise ValueError(
            f"time_step must be an integer from 0 to {len(times) - 1} "
            f"({len(times)} saved results); got {time_step!r}."
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
                 description='Compare a variable between two zero-based saved time-step indices. '
                             'Defaults to the first and last saved steps.')
@wbgeo_inspector()
def inspect_simulation_result_plot_cross_section_2D(
    simulation_result: SimulationResults,
    _inspector: InspectorHelper,
    var_name: Optional[str] = None,
    time_step1: Optional[int] = None,
    time_step2: Optional[int] = None,
    origin: Optional[AnnotatedPoint] = (500, 500, 500),
    normal: AnnotatedPoint = (0, 1, 0),
):
    times = sorted(simulation_result.nodes_by_time)
    if not times:
        raise ValueError("Simulation has no saved time steps.")
    # Use the mesh centroid so the slicing plane intersects the domain.
    if not origin:
      origin = tuple(simulation_result.nodes_by_time[times[0]].mean(axis=0))
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
                'Line endpoints are model coordinates; defaults use the mesh bounding-box diagonal.',
)
@wbgeo_inspector()
def inspect_simulation_result_plot_variable_along_line(
    simulation_result: SimulationResults,
    _inspector: InspectorHelper,
    var_name: Optional[str] = None,
    time_step: Optional[int] = None,
    p0: Optional[AnnotatedPoint] = None,
    p1: Optional[AnnotatedPoint] = None,
    n_samples: int = 200,
):
    physical_time = (None if time_step is None
                     else _saved_time_for_step(simulation_result, time_step))
    plot_variable_along_line(
        simulation_result,
        var_name=var_name,
        time=physical_time,
        p0=p0,
        p1=p1,
        n_samples=n_samples,
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
    decimals: int = 3,
):
    plot_variable_time_series(
        simulation_result,
        var_name=var_name,
        point=point,
        decimals=decimals,
        time_step_start=time_step_start,
        time_step_end=time_step_end,
    )


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_stdout',
                 title='View Solver Output',
                 description='Prints the raw sfepy-run solver output for this result, per stage '
                             '("pressure"/"heat" for a Hydrothermal problem, "custom" for a Custom '
                             'SfePy problem) -- lets solver convergence/quality be checked on a '
                             'normal, successful run too, not only via the error raised when a '
                             'solve fails to converge.')
@wbgeo_inspector()
def inspect_simulation_result_stdout(simulation_result: SimulationResults, _inspector: InspectorHelper):
    # GOLEM stores its transcript on disk; leave the SfePy handling below unchanged.
    if getattr(simulation_result, "source_solver", None) == "golem":
        output_dir = getattr(simulation_result, "source_output_dir", None)
        if output_dir:
            transcript_path = pathlib.Path(output_dir) / "golem_stdout.txt"
            if transcript_path.is_file():
                print("=== GOLEM ===")
                print(transcript_path.read_text(encoding="utf-8", errors="replace"))
                return
        print("No GOLEM solver output recorded for this result.")
        return
    if simulation_result.sfepy_stdout:
      for stage, stdout in simulation_result.sfepy_stdout.items():
        print(f"=== {stage} ===")
        print(stdout)
    elif simulation_result.golem_solver_output:
      print("=== GOLEM ===")
      print(simulation_result.golem_solver_output)
    else:
      print("No solver output recorded for this result.")
