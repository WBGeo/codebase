"""
WBGeo Workbench components for the SfePy hydrothermal simulation pipeline.

Thin-wrapper file only: per the project's "one workbench file per step" rule
(already applied to structural_modeling_components/meshing_components),
@wbgeo_component must appear here and nowhere else in core/simulation_components/.
The actual logic lives in sfepy_hydrothermal_builder.py / sfepy_hydrothermal_run.py
and is called here unmodified.
"""
import typing
from typing import Dict, Optional

import pydantic
from py_api_wbgeo import smartcontrols
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type, wbgeo_inspector, InspectorHelper, \
    BasicallyABufferedFile
from py_api_wbgeo.smartcontrols import CtrlGroup, CtrlLabel, SmartInput, SmartInputFormData

from core.object_components import MeshResults, StructuralModelResults, SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, SfepyProblem, RockUnitProperties, FluidProperties,
    enumerate_rock_units, check_mesh_has_known_type, check_mesh_has_lithology_mapping,
    check_mesh_has_no_engineering_objects,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_run import (
    run_simulation_sfepy as _run_simulation_sfepy,
    export_simulation_results as _export_simulation_results,
)
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials,
)

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


@wbgeo_component(
    description="Builds the SfePy hydrothermal problem definition (rock/fluid "
                "properties, boundary conditions, and solver settings) from a "
                "mesh and its structural model. Use the smart options to set "
                "per-rock-unit and fluid properties; check 'Enable Fault Zone' "
                "to give cells near any fault their own separate material "
                "(from the options' Fault Zone Properties group) instead of "
                "treating them as ordinary lithology.",
    title='Build Hydrothermal Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
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
        connect into the same Run Simulation component as Build Custom SfePy Problem's output.

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
    title='Build Custom SfePy Problem',
    color='#e07a5f',
    border_color='#000000',
    group='Simulation',
    identifier='wbgeo::simulation_build_custom_sfepy_problem',
    return_name='sfepy_problem',
    input_checks=[check_mesh_has_known_type, check_mesh_has_lithology_mapping, check_mesh_has_no_engineering_objects],
)
def build_custom_sfepy_problem(
    input_file: BasicallyABufferedFile,
    mesh_results: MeshResults,
    geomodel_result: Optional[StructuralModelResults] = None,
    fault_zone_n_voxels: Optional[int] = None,
) -> SfepyProblem:
    """
    :param input_file: A complete SfePy input file (.py) with its own staging/sequencing logic.
    :param mesh_results: The mesh to solve on (implicit, structured, or unstructured). Meshes
        with wells/sources are rejected -- not supported yet.
    :param geomodel_result: Optional structural model, used for an additional soft sanity
        check (referenced-group count vs. lithology count), and required if
        fault_zone_n_voxels is set.
    :param fault_zone_n_voxels: If set, gives cells within this many structural-grid voxels
        of any active fault's trace their own group id (see the returned CustomSfepyBuilder's
        fault_group_id) -- the same "damage zone" cell selection Build Hydrothermal Problem
        uses, referenceable from the custom file as 'cells of group N'. No material/equation
        is auto-generated for it; that's entirely up to the custom file. None (default)
        leaves only the mesh's real lithology groups available, exactly like today.
    :return: The custom problem definition, wrapped in the shared SfepyProblem envelope so it
        can connect into the same Run Simulation component as Build Hydrothermal Problem's
        output. Not restricted to hydrothermal problems -- the file can define any SfePy
        physics; only the mesh/region bookkeeping is validated, not the equations themselves.

    Raises:
        ValueError: mesh_results has no known mesh_type, no lithology mapping, or contains
            a well/source block; the uploaded file isn't valid UTF-8 text; the file
            references a 'cells of group N' that doesn't exist on mesh_results; or
            fault_zone_n_voxels is set with no geomodel_result.
    """
    # Pre-checks can be ignored/skipped by a caller (see docs/developers/components.md),
    # so also enforced here rather than relying solely on the Workbench's
    # input_checks catching the connection.
    check_mesh_has_known_type(mesh_results)
    check_mesh_has_lithology_mapping(mesh_results)
    check_mesh_has_no_engineering_objects(mesh_results)

    # Read exactly once here, not as a separate input_checks pre-check:
    # BasicallyABufferedFile is stream-like (Union[io.IOBase, GeoTempFile]),
    # and there's no guarantee a pre-check and this function body would be
    # handed the same already-consumed stream -- CustomSfepyBuilder does
    # all content-based validation against the decoded string instead.
    raw = input_file.read()
    try:
        content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as e:
        raise ValueError("Uploaded custom SfePy input file is not valid UTF-8 text.") from e

    return SfepyProblem(custom=CustomSfepyBuilder(
        input_file_contents=content, mesh_results=mesh_results, geomodel_result=geomodel_result,
        fault_zone_n_voxels=fault_zone_n_voxels,
    ))


@wbgeo_component(
    description='Runs the given problem definition (from either Build '
                'Hydrothermal Problem or Build Custom SfePy Problem) -- the '
                'two-stage SfePy hydrothermal solve (steady-state pressure, '
                'then transient advective-conductive heat transport) for a '
                'Hydrothermal problem, or a single sfepy-run invocation for a '
                'Custom SfePy problem.',
    title='Run Simulation',
    color='pink',
    border_color='#000000',
    group='Simulation',
    identifier='wbgeo::simulation_run',
    return_name='simulation_result',
)
def run_simulation(problem: SfepyProblem) -> SimulationResults:
    """
    :param problem: Problem definition from Build Hydrothermal Problem or Build Custom SfePy Problem.
    :return: Simulation results for every solved/saved time step.
    """
    return _run_simulation_sfepy(problem.inner())


@wbgeo_component(
    description='Exports hydrothermal simulation results (all time steps) to '
                'a downloadable VTK time series (.pvd + one .vtk per step, zipped).',
    title='Export Simulation Results',
    group='Export',
    identifier='wbgeo::export_simulation_results',
    return_name='file',
)
def export_simulation_results(simulation_result: SimulationResults, format: str = "vtk") -> BasicallyABufferedFile:
    """
    :param simulation_result: Simulation results from Run Simulation.
    :param format: Export format; only "vtk" is implemented so far.
    :return: A downloadable file containing the exported results.
    """
    return _export_simulation_results(simulation_result, format)


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
def inspect_sfepy_problem_plot_materials(problem: SfepyProblem, _inspector: InspectorHelper):
    plot_builder_materials(problem.inner())


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_plot_variable_at_a_time',
                 title='Plot Variable (Final Time)',
                 description='Plots "T" at the final solved time step if present, else '
                             'whatever variable the result actually has -- a custom SfePy '
                             'problem is not restricted to hydrothermal physics.')
@wbgeo_inspector()
def inspect_simulation_result_plot_variable_at_a_time(
    simulation_result: SimulationResults, _inspector: InspectorHelper):
    # var_name/time left at their defaults ("T" if present else whatever
    # field exists, at the final time step) -- see
    # simulation_visualization.plot_variable_at_a_time/_default_var_name.
    plot_variable_at_a_time(simulation_result, cmap="coolwarm", show_edges=True)


@wbgeo_component(identifier='wbgeo::inspect_simulation_result_plot_cross_section_2D',
                 title='Plot Cross Section 2D',
                 description='Plots a cross-section of "T" if present, else whatever '
                             'variable the result actually has, comparing the initial and '
                             'final time steps.')
@wbgeo_inspector()
def inspect_simulation_result_plot_cross_section_2D(
    simulation_result: SimulationResults, _inspector: InspectorHelper):
    # plot_cross_section_2D's own default origin=(0, 0, 0) sits right on the
    # domain boundary -- mesh nodes are cell-centered (inset from the
    # nominal domain edges), so that default slice can miss every node and
    # raise. The mesh's own centroid is always a valid slice origin
    # regardless of the model's actual extent.
    first_time = min(simulation_result.nodes_by_time.keys())
    origin = tuple(simulation_result.nodes_by_time[first_time].mean(axis=0))
    # var_name left at its default ("T" if present else whatever field
    # exists) -- see simulation_visualization._default_var_name.
    plot_cross_section_2D(simulation_result, origin=origin, cmap="coolwarm")
