# Simulation

The simulation component connects the Workbench to numerical solvers.
Since the meshing component exports the mesh to different formats, WBGeo
can interface with various simulation packages. The current simulation
architecture provides two solver pipelines:

-   **SfePy** (Simple Finite Elements in Python) for Python-native
    finite-element simulations, supporting the built-in hydrothermal
    workflow as well as user-supplied SfePy problems.
-   **GOLEM/MOOSE** for hydrothermal and simple-reservoir simulations,
    as well as user-supplied GOLEM/MOOSE problems, executed through an
    external Docker environment.

Both pipelines follow the same general Workbench pattern:

`Build problem -> Run simulation -> SimulationResults -> Visualise / Export`

The solver-specific builders and runners are different, but both
pipelines return the common `SimulationResults` type. This allows the
same downstream result visualisation and export components.

------------------------------------------------------------------------

## Concepts

Before a model can be solved, it needs a **problem definition**: the
mesh together with material properties, boundary and initial conditions,
solver settings, and the solver-specific description of the governing
equations.

The Workbench currently provides five problem-building routes.

### SfePy problem definitions

**Hydrothermal Problem** --- auto-generated from the mesh and structural
model. Material regions and boundary regions are inferred from the
lithology mapping and mesh information. The built-in formulation solves
simplified single-phase Darcy flow followed by advective-conductive heat
transport.

**Custom SfePy Problem** --- wraps a complete user-supplied SfePy `.py`
input file. The custom file specifies a user defined probem with its
variables, materials, equations and solvers. WBGeo validates the input
file against the mesh (known mesh type, known lithology mapping, and
that every region the file references actually exists).

Both SfePy routes produce an `SfepyProblem` and connect to the same
`Run Simulation` component.

### GOLEM problem definitions

**GOLEM Hydrothermal Problem** --- auto-generates a GOLEM/MOOSE
hydrothermal problem from a WBGeo mesh, structural model, rock
properties, fluid properties, boundary conditions, and optional fault
properties.

**GOLEM Simple Reservoir Problem** --- builds a dedicated reservoir
model for GOLEM, including the reservoir mesh and support for synthetic
fractures, a well, and injection/extraction source behaviour.

**Custom GOLEM Problem** --- wraps a complete user-supplied GOLEM/MOOSE
`.i` input file. The file is validated against the WBGeo mesh before
execution and must use `${GOLEM_MESH_FILE}` as the mesh placeholder.

All three GOLEM routes produce a `GolemProblem` and connect to the same
`Run GOLEM Simulation` component.

### Materials

For the built-in hydrothermal workflows, each geological unit can be
assigned its own rock properties. These include quantities such as
porosity, permeability, thermal conductivity, density, or volumetric
heat capacity, depending on the solver-specific builder. Fluid
properties include viscosity, thermal_conductivity, density, and
heat_capacity.

### Fault representation

Fault handling is an important difference between the two solver
pipelines.

In the built-in **SfePy** workflow, faults are represented as a
**volumetric damage zone**. Mesh cells within a specified distance of
the fault trace are reassigned to a fault-zone material. This is a
simplified representation of a sealing or conductive fault zone rather
than an explicit lower-dimensional discontinuity.

In **GOLEM**, an explicit unstructured mesh can contain exported **2D
fault element blocks**. `GolemFaultProperties` can then be assigned to
those fault blocks, including an aperture used for the lower-dimensional
fault formulation. This makes the GOLEM route suitable when the
explicitly meshed fault surface should participate in the numerical
model.

### Solver execution

SfePy and GOLEM differ mainly at execution time.

**SfePy** runs inside the WBGeo Python environment. For the built-in
hydrothermal problem, pressure and temperature are solved sequentially:
first the Darcy pressure problem, then transient heat transport using
the resulting Darcy velocity. A custom SfePy problem instead follows the
execution defined by the supplied Python input.

**GOLEM** is external to the WBGeo Python environment. The GOLEM runner
stages the mesh and input file, validates their compatibility, executes
GOLEM/MOOSE through Docker and MPI, collects the Exodus output, and
converts it back to WBGeo's common result representation.

------------------------------------------------------------------------

## Workflow

### 1. Build the problem definition

Choose the builder corresponding to the solver and model type.

### SfePy: Auto-generated Hydrothermal Problem

> **Workbench component:** `Build Hydrothermal Problem`

Requires a mesh together with the structural model used to create it.

-   **`mesh_results`**: The mesh to solve on (implicit, structured, or
    unstructured).
-   **`geomodel_result`**: The structural model.
-   **`options`** *(optional)*: Per-rock-unit properties (porosity,
    permeability, thermal conductivity, volumetric heat capacity),
    global fluid properties, and fault-zone properties.
-   **`include_flow`** *(optional)*: If `False`, skips the pressure
    stage and solves pure heat conduction only. Default `True`.
-   **`enable_fault_zone`** *(optional)*: If `True`, gives cells near
    any fault their own material (the fault-zone properties from
    `options`) instead of treating them as ordinary lithology. Has no
    effect if the structural model has no faults. Default `False`.
-   **`fault_zone_n_voxels`** *(optional)*: Width of the fault damage
    zone, in structural-grid voxels. Default `1`.
-   **`t0`**, **`t1`**, **`num_steps`** *(optional)*: Start time, end
    time, and number of steps for the transient heat stage. Default
    `t0=0.0`, `num_steps=1`; `t1` defaults to a data-driven estimate of
    the model's thermal diffusion timescale if not given.
-   **`p_top`**, **`p_bottom`** *(optional)*: Pressure boundary values
    at the model top/bottom, in Pa. Default `p_top=0.0`, `p_bottom=1e6`.
-   **`t_top`**, **`t_bottom`** *(optional)*: Temperature boundary
    values at the model top/bottom, in °C. Default `t_top=10.0`,
    `t_bottom=60.0`.
-   **`linear_solver`** *(optional)*: `"iterative"` (default) scales to
    large meshes; `"direct"` is exact but does not scale well --- use it
    for small or debug meshes.
-   **`linear_solver_i_max`**, **`linear_solver_eps_r`** *(optional)*:
    Maximum iterations and relative residual tolerance for the iterative
    linear solver. Default `500` and `1e-8`.
-   **`linear_solver_eps_r`** *(optional)*: Linear-solver controls.

Example call in the codebase:

``` python
rock_properties = {
    "basement": RockUnitProperties(name="basement", porosity=0.10, permeability=1e-13, k_solid=50, rho_c_solid=3900000),
    "rock1": RockUnitProperties(name="rock1", porosity=0.30, permeability=1e-11, k_solid=10, rho_c_solid=800000),
    "rock2": RockUnitProperties(name="rock2", porosity=0.20, permeability=1e-12, k_solid=20, rho_c_solid=3400000),
}
fluid = FluidProperties(mu=1.0e-3, k_fluid=15, rho_c_fluid=2400000)

builder = HydrothermalProblemBuilder(
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    include_flow=True,
    t1=5e11,
    num_steps=3,
)
```

Here, `rock_properties` gives each lithology its own porosity,
permeability, thermal conductivity, and heat capacity; `fluid` describes
the pore fluid. `t1=5e11` overrides the default diffusion-timescale
estimate for the heat stage's end time, solved over `num_steps=3` steps.

A faulted model additionally enables the damage zone:

``` python
fault_zone_properties = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2300000
)

builder = HydrothermalProblemBuilder(
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    fault_zone_properties=fault_zone_properties,
    fault_zone_n_voxels=1,
)
```

This gives the fault a low porosity/permeability --- a sealing fault.
Raising `permeability` above the surrounding rock instead represents a
fracture-enhanced conduit.

**SfePy hydrothermal problem limitations:**

-   The built-in model uses a single fluid phase.
-   Pressure-to-temperature coupling is one-way; temperature does not
    feed back into the flow field.
-   Faults are represented as a combined volumetric damage zone rather
    than explicit interface elements.
-   Engineering objects such as wells and point sources are not
    physically solved by this built-in workflow.

### SfePy: User-supplied Problem

> **Workbench component:** `Build Custom SfePy Problem`

This component brings a complete SfePy `.py` input into the Workbench.

-   **`input_file`**: Path to a complete SfePy input file (`.py`).
-   **`mesh_results`**: The mesh to solve on. Meshes containing wells or
    sources are rejected --- not supported by this component.
-   **`geomodel_result`** *(optional)*: The structural model.
-   **`fault_zone_n_voxels`** *(optional)*: If set, gives cells within
    this many structural-grid voxels of any active fault's trace their
    own region id --- the same damage-zone cell selection
    `Build Hydrothermal Problem` uses, referenceable from the custom
    file as `cells of group N`.

The builder validates mesh type, lithology mapping, and referenced cell
groups. Arbitrary equations, material definitions, and solver settings
remain the responsibility of the custom SfePy input and are ultimately
checked by SfePy.

Example call in the codebase:

``` python
with open("examples/synthetic_examples/model1/input_data/simulation_files/ custom_hydrothermal.py", encoding="utf-8") as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
)
```

**SfePy custom problem limitations:**

-   Faults can only be represented as a combined volumetric damage zone
    rather than explicit interface elements.
-   Engineering objects such as wells and point sources cannot be
    physically solved.

### GOLEM: Auto-generated Hydrothermal Problem

> **Workbench component:** `Build GOLEM Hydrothermal Problem`

This component constructs a `GolemHydrothermalProblemBuilder` from a
WBGeo mesh and structural model. It uses rock and fluid properties
analogous to the SfePy hydrothermal workflow, while supporting explicit
GOLEM fault properties when suitable 2D fault blocks are present in an
unstructured mesh.

Important inputs and controls include:

-   **`mesh_results`**: Mesh used by GOLEM (implicit, structured, or
    unstructured).
-   **`geomodel_result`**: Structural model.
-   **rock properties**: Geological-unit hydrothermal properties.
-   **fluid properties**: Properties of the fluid phase.
-   **fault properties**: `GolemFaultProperties` for explicit 2D fault
    blocks.
-   **`p_top`**, **`p_bottom`**: Pressure boundary values.
-   **`t_top`**, **`t_bottom`**: Temperature boundary values.
-   **initial pressure / temperature**: Initial-state controls.
-   **`start_time`**, **`end_time`**, **`dt`**, **`num_steps`**: Time
    controls.
-   **`output_interval`**: Result-output frequency.
-   **`print_linear_residuals`**: Enables linear residual reporting.

Example call in the codebase:

``` python
rock_properties = {
    "basement": RockUnitProperties(name="basement", porosity=0.05, permeability=1e-15, k_solid=3.0, rho_c_solid=2.2e6),
    "rock1": RockUnitProperties(name="rock1", porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6),
    "rock2": RockUnitProperties(name="rock2", porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6),
    "rock3": RockUnitProperties(name="rock3", porosity=0.15, permeability=1e-14, k_solid=2.0, rho_c_solid=2.1e6),
    "rock4": RockUnitProperties(name="rock4", porosity=0.15, permeability=1e-14, k_solid=2.0, rho_c_solid=2.1e6),
}

fluid = FluidProperties(
    viscosity=1.0e-3,
    thermal_conductivity=0.6,
    density=1000,
    fluid_modulus = 2.2e9
)

fault_zone_properties = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
)


golem_fault_properties = GolemFaultProperties(
    name="fault",
    porosity=fault_zone_properties.porosity,
    permeability=fault_zone_properties.permeability,
    thermal_conductivity=fault_zone_properties.thermal_conductivity,
    density=fault_zone_properties.density,
    heat_capacity=fault_zone_properties.heat_capacity,
    fluid_modulus=fault_zone_properties.fluid_modulus,
    aperture=40,
    permeability_model="constant",
)

problem_golem = GolemHydrothermalProblemBuilder(
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    fault_properties=golem_fault_properties,
    p_top=0.0,
    p_bottom=1.0e6,
    t_top=10.0,
    t_bottom=60.0,
    num_steps=3,
)
```

When explicit fault behaviour is required, use an unstructured mesh
containing the exported 2D fault block(s). This differs from SfePy's
built-in damage-zone representation.

### GOLEM: Simple Reservoir Problem

> **Workbench component:** `Build GOLEM Simple Reservoir Problem`

The simple-reservoir builder creates both the dedicated unstructured
reservoir mesh and the corresponding GOLEM/MOOSE problem definition. It
supports two synthetic fractures, a well, an injection source,
geological faults already present in the structural model, local mesh
refinement, material/fluid properties, transient controls, and low-level
PETSc solver settings.

The builder accepts the following inputs.

#### Structural-model input

-   **`geomodel_result`**: Required structural model used to determine
    the model extent and construct the reservoir geometry.

#### Unstructured-mesh inputs

-   **`mesh_size`** *(optional)*: Target volume-mesh size. Default
    `30.0`.
-   **`curve_mesh_size`** *(optional)*: Target mesh size on well curves.
    Default `5.0`.
-   **`mapping_litho`** *(optional)*: Lithology mapping method. Default
    `"automatic_centers"`.
-   **`gmsh_flag`** *(optional)*: Option forwarded to the unstructured
    mesh generator. Default `False`.
-   **`merge_file`** *(optional)*: Optional merge file passed to the
    unstructured meshing workflow. Default `None`.
-   **`z_threshold`** *(optional)*: Optional z-direction threshold used
    by the unstructured meshing workflow. Default `None`.
-   **`unstructured_mesh_options`** *(optional)*: Dictionary of
    additional keyword arguments accepted by
    `create_unstructured_mesh_data()`. Default `None`.

For example, additional meshing controls can be supplied as:

``` python
unstructured_mesh_options={
    "tolerance": 50,
    "DISTANCE_THRESHOLD": 40,
    "PROJECTION_THRESHOLD": 60,
    "EXTRUSION_FACTOR": 80,
}
```

#### Optional geometry overrides

Normally, the builder uses the complete structural-model extent and
places its reference reservoir geometry automatically. The generated
fracture pair + a well are centred in the model while preserving the
reference geometry relationships.

-   **`reservoir_extent_override`** *(optional)*: Explicit
    `(xmin, xmax, ymin, ymax, zmin, zmax)` reservoir extent. Default
    `None`.
-   **`fracture_planes_override`** *(optional)*: Explicit
    synthetic-fracture plane coordinates. Default `None`.
-   **`wells_override`** *(optional)*: Explicit well coordinates.
    Default `None`.

#### Well refinement inputs

-   **`well_refine_size_min`** *(optional)*: Minimum target mesh size
    close to the well. Default `5.0`.
-   **`well_refine_size_max`** *(optional)*: Maximum target mesh size
    away from the well. Default `30.0`.
-   **`well_refine_dist_min`** *(optional)*: Inner distance for well
    refinement. Default `10.0`.
-   **`well_refine_dist_max`** *(optional)*: Outer distance for well
    refinement. Default `60.0`.

#### Synthetic-fracture refinement inputs

-   **`fracture_refine_size_min`** *(optional)*: Minimum target mesh
    size close to the synthetic fractures. Default `5.0`.
-   **`fracture_refine_size_max`** *(optional)*: Maximum target mesh
    size away from the fractures. Default `30.0`.
-   **`fracture_refine_dist_min`** *(optional)*: Inner distance for
    fracture refinement. Default `10.0`.
-   **`fracture_refine_dist_max`** *(optional)*: Outer distance for
    fracture refinement. Default `60.0`.
-   **`scale_refinement_with_model`** *(optional)*: If `True`,
    refinement lengths are scaled relative to the reference model using
    the smallest model-axis scale. `mesh_size` and `curve_mesh_size`
    remain direct meshing inputs. Default `True`.

#### Reservoir rock properties

One common intrinsic rock material is applied to all 3D reservoir
blocks. Existing geological-fault blocks use the same intrinsic rock
properties.

-   **`rock_porosity`** *(optional)*: Rock porosity. Default `0.1`.
-   **`rock_permeability`** *(optional)*: Rock permeability. Default
    `1.0e-15`.
-   **`rock_density`** *(optional)*: Rock density. Default `2.6e3`.
-   **`rock_thermal_conductivity`** *(optional)*: Rock thermal
    conductivity. Default `3.0`.
-   **`rock_heat_capacity`** *(optional)*: Rock heat capacity. Default
    `950.0`.

#### Synthetic-fracture properties

-   **`fracture_porosity`** *(optional)*: Porosity assigned to the two
    internally generated synthetic fractures. Default `1.0`.
-   **`fracture_aperture`** *(optional)*: Aperture of the synthetic
    fractures. Default `1.0e-4`.

#### Geological-fault property

-   **`geological_fault_aperture`** *(optional)*: Aperture/scaling
    required by GOLEM for exported lower-dimensional geological-fault
    blocks. Geological faults otherwise use the common rock properties
    above. Default `1.0`.

#### Well properties

-   **`well_porosity`** *(optional)*: Well porosity. Default `1.0`.
-   **`well_permeability`** *(optional)*: Well permeability. Default
    `3.12e-4`.
-   **`well_scaling_factor`** *(optional)*: GOLEM scaling factor used
    for the well block. Default `0.05`.

#### Fluid properties

-   **`fluid_density`** *(optional)*: Fluid density. Default `1.0e3`.
-   **`fluid_viscosity`** *(optional)*: Fluid dynamic viscosity. Default
    `1.0e-3`.
-   **`fluid_heat_capacity`** *(optional)*: Fluid heat capacity. Default
    `4.2e3`.
-   **`fluid_thermal_conductivity`** *(optional)*: Fluid thermal
    conductivity. Default `0.65`.
-   **`fluid_modulus`** *(optional)*: Fluid modulus. Default `2.5e9`.

#### Initial and boundary conditions

-   **`initial_pressure`** *(optional)*: Initial reservoir pressure.
    Default `1.0e7`.
-   **`initial_temperature`** *(optional)*: Initial reservoir
    temperature. Default `150.0`.
-   **`boundary_pressure`** *(optional)*: Reservoir boundary pressure.
    Default `1.0e7`.
-   **`boundary_temperature`** *(optional)*: Reservoir boundary
    temperature. Default `150.0`.
-   **`injection_temperature`** *(optional)*: Injected-fluid
    temperature. Default `70.0`.

#### Injection and extraction inputs

-   **`injection_point`** *(optional)*: Explicit `(x, y, z)` injection
    location. If `None`, the automatically positioned centre of the
    first synthetic fracture is used.
-   **`extraction_point`** *(optional)*: Explicit `(x, y, z)` extraction
    location. If `None`, the automatically positioned centre of the
    second synthetic fracture is used.
-   **`injection_rate`** *(optional)*: Injection source rate. Default
    `3.0`.
-   **`extraction_rate`** *(optional)*: Extraction source rate. Default
    `3.0`.
-   **`source_start_time`** *(optional)*: Time at which the
    injection/extraction sources become active. Default `0.0`.
-   **`source_end_time`** *(optional)*: Time at which the sources stop.
    Default `None`.

#### Transient-simulation inputs

-   **`start_time`** *(optional)*: Simulation start time. Default `0.0`.
-   **`end_time`** *(optional)*: Simulation end time. Default `None`.
-   **`time_sequence`** *(optional)*: Explicit sequence of solve times.
    If `None`, the builder uses its default simple-reservoir time
    sequence.
-   **`automatic_scaling`** *(optional)*: Enables MOOSE automatic
    scaling. Default `True`.
-   **`compute_scaling_once`** *(optional)*: Controls whether scaling is
    computed only once. Default `False`.

#### PETSc solver controls

The following advanced inputs expose the solver configuration used by
the generated GOLEM/MOOSE input:

-   **`coupled_petsc_options_iname`** and
    **`coupled_petsc_options_value`**: PETSc options for the coupled
    nonlinear solve.
-   **`pressure_petsc_options_iname`** and
    **`pressure_petsc_options_value`**: PETSc options for the pressure
    field solve.
-   **`temperature_petsc_options_iname`** and
    **`temperature_petsc_options_value`**: PETSc options for the
    temperature field solve.

The builder provides working defaults for all six solver-control inputs,
so they normally need to be changed only for advanced solver tuning.

#### Output and diagnostic controls

-   **`print_linear_residuals`** *(optional)*: Print linear residual
    information. Default `True`.
-   **`print_nonlinear_converged_reason`** *(optional)*: Print the
    nonlinear convergence reason. Default `True`.
-   **`perf_graph`** *(optional)*: Enable MOOSE performance-graph
    output. Default `True`.

Example call in the codebase:

``` python
reservoir_problem_model2 = GolemSimpleReservoirProblemBuilder(
    geomodel_result=structural_model_result,

    # Unstructured mesh
    mesh_size=50.0,
    curve_mesh_size=5.0,
    mapping_litho="automatic_centers",
    gmsh_flag=False,
    merge_file=None,
    z_threshold=10.0,
    unstructured_mesh_options={
        "tolerance": 50,
        "DISTANCE_THRESHOLD": 40,
        "PROJECTION_THRESHOLD": 60,
        "EXTRUSION_FACTOR": 80,
    },

    # Automatic geometry; replace None values to override it explicitly
    reservoir_extent_override=None,
    fracture_planes_override=None,
    wells_override=None,

    # Well refinement
    well_refine_size_min=5.0,
    well_refine_size_max=30.0,
    well_refine_dist_min=10.0,
    well_refine_dist_max=60.0,

    # Synthetic-fracture refinement
    fracture_refine_size_min=5.0,
    fracture_refine_size_max=30.0,
    fracture_refine_dist_min=10.0,
    fracture_refine_dist_max=60.0,
    scale_refinement_with_model=True,

    # Common reservoir/geological-fault rock properties
    rock_porosity=0.10,
    rock_permeability=1.0e-15,
    rock_density=2.6e3,
    rock_thermal_conductivity=3.0,
    rock_heat_capacity=950.0,

    # Synthetic fractures and geological faults
    fracture_porosity=1.0,
    fracture_aperture=1.0e-4,
    geological_fault_aperture=1.0,

    # Well
    well_porosity=1.0,
    well_permeability=3.12e-4,
    well_scaling_factor=0.05,

    # Fluid
    fluid_density=1.0e3,
    fluid_viscosity=1.0e-3,
    fluid_heat_capacity=4.2e3,
    fluid_thermal_conductivity=0.65,
    fluid_modulus=2.5e9,

    # Initial/boundary conditions
    initial_pressure=1.0e7,
    initial_temperature=150.0,
    boundary_pressure=1.0e7,
    boundary_temperature=150.0,
    injection_temperature=70.0,

    # Sources
    injection_point=None,
    extraction_point=None,
    injection_rate=3.0,
    extraction_rate=3.0,
    source_start_time=0.0,
    source_end_time=None,

    # Transient solve
    start_time=0.0,
    end_time=None,
    time_sequence=None,
    automatic_scaling=True,
    compute_scaling_once=False,

    # Output/diagnostics
    print_linear_residuals=True,
    print_nonlinear_converged_reason=True,
    perf_graph=True,
)
```

### GOLEM: User-supplied Problem

> **Workbench component:** `Build Custom GOLEM Problem`

A complete user-provided GOLEM/MOOSE `.i` file can be used through
`CustomGolemBuilder`.

-   **`input_file`**: Path to the custom `.i` file.
-   **`mesh_results`**: WBGeo mesh.
-   **`geomodel_result`** *(optional)*: Structural model for consistency
    checks.
-   **`print_linear_residuals`**: Controls residual output.

The custom file must refer to the WBGeo mesh using:

``` text
${GOLEM_MESH_FILE}
```

The runner replaces this placeholder with the staged Exodus mesh path
before execution.

Before Docker/MPI is started, WBGeo validates the custom GOLEM input
against the exported mesh. The validation checks the required
GOLEM/MOOSE structure and verifies that material block assignments cover
the mesh blocks expected by WBGeo. This catches common mesh/input
mismatches before the external solver starts.

Example call in the codebase:

``` python
with open(
    "examples/synthetic_examples/model2/input_data/"
    "simulation_files/custom_reservoir.i",
    encoding="utf-8",
) as f:
    custom_golem_input = f.read()

custom_reservoir_problem = CustomGolemBuilder(
    input_file_contents=custom_golem_input,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    print_linear_residuals=True,
)
```

### 2. Run the simulation

#### Run SfePy

> **Workbench component:** `Run Simulation`

Both SfePy builders connect to the same runner.

Example call in the codebase:

``` python
data_by_time = run_simulation_sfepy(builder)
```

For an auto-generated hydrothermal problem, the runner performs the
pressure stage followed by the heat stage. For a custom SfePy problem,
execution follows the supplied input.

#### Run GOLEM

> **Workbench component:** `Run GOLEM Simulation`

All three GOLEM builders connect to the same runner through the common
`GolemProblem` type.

Important runner inputs are:

-   **`problem`**: GOLEM problem produced by one of the GOLEM builders.
-   **`image_name`**: Docker image containing GOLEM.
-   **`project_dir`**: Host project directory mounted into the
    container.
-   **`docker_dir`**: Corresponding path inside the container.
-   **`output_dir`** *(optional)*: Host destination for final results.
-   **`nproc`**: Number of MPI processes.
-   **`keep_files_dir`** *(optional)*: Directory in which
    staged/generated files can be retained for debugging.

Example call in the codebase:

``` python
problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    image_name="moose-dev:fixed",
    project_dir="/path/to/projects",
    docker_dir="/projects",
    output_dir="/path/to/output",
    nproc=4,
)
```

The GOLEM runner performs the following operations:

1.  Exports and stages the WBGeo mesh as Exodus.
2.  Generates or stages the GOLEM/MOOSE `.i` input.
3.  Validates the input against the mesh and expected material/block
    structure.
4.  Replaces `${GOLEM_MESH_FILE}` with the staged mesh path.
5.  Executes GOLEM in the configured Docker environment using MPI.
6.  Collects the Exodus results.
7.  Loads the numerical result into `SimulationResults`.
8.  Cleans temporary resources while preserving explicitly requested
    output/debug files.

### 3. Export the results

> **Workbench component:** `Export Simulation Results`

SfePy and GOLEM both return `SimulationResults`, but their export paths
are different.

#### SfePy result export

For SfePy, the standard Workbench export is a zipped VTK time series.
The ZIP contains one `.vtk` file per saved time step together with a
`.pvd` collection file for loading the complete time series in ParaView.

Example call in the codebase:

``` python
buf = export_simulation_results(data_by_time, format="vtk")
with open("simulation_results.zip", "wb") as f:
    f.write(buf.getvalue())
```

#### GOLEM result output and export

GOLEM/MOOSE may produce several output files. In addition to the native
Exodus result (normally `wbgeo.e`), a model may produce CSV files or
other configured outputs.

There are two ways to keep GOLEM results in the codebase.

**Option 1 - define `output_dir` when running GOLEM.**

Example call in the codebase:

``` python
problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    image_name="moose-dev:fixed",
    project_dir="/path/to/projects",
    docker_dir="/projects",
    output_dir="/path/to/my_golem_output",
    nproc=4,
)
```

When `output_dir` is supplied, `run_simulation_golem()` copies the
preserved user-facing GOLEM outputs directly into that directory. This
normally includes `wbgeo.e`, `golem_stdout.txt`, `wbgeo_generated.i`,
`wbgeo_mesh.exo`, and any additional GOLEM/MOOSE outputs such as CSV
files when they are specified in the input model file.

**Option 2 - export the complete GOLEM output directory as a ZIP.**

The `SimulationResults` returned by `run_simulation_golem()` retains the
path to its preserved GOLEM output directory. This directory can be
saved as one Zip file using `export_simulation_results`.

Example call in the codebase:

``` python
buf = export_simulation_results(problem_golem_results, format="golem")
with open("simulation_golem_results.zip", "wb") as f:
    f.write(buf.getvalue())
```

This is useful when the complete result set needs to be downloaded,
transferred, or stored as one artifact. The archive preserves the native
`wbgeo.e` together with CSV files and the other user-facing files
produced or retained by the GOLEM runner.

If only the native GOLEM Exodus database is required, it can be exported
as a single Exodus file.

Example call in the codebase:

``` python
buf = export_simulation_results(problem_golem_results, format="exodus")
with open("simulation_golem_results.e", "wb") as f:
    f.write(buf.getvalue())
```

Here `format="exodus"` returns GOLEM's original multi-time-step Exodus
file.

> **Note:** For GOLEM, choose either saving the outputs by defining
> output directory in `run_simulation_golem` (`output_dir`) or use
> `export_simulation_results`. If
> `export_simulation_results' is used, use`format="golem"`to save the whole preserved output directory as a ZIP. Use`format="exodus"`only when the single native Exodus result is specifically desired. For SfePy, only use`export_simulation_results'
> with `format="vtk"` for saving the zipped VTK time series.

------------------------------------------------------------------------

## Visualisation and Inspection

Once a simulation has been converted to `SimulationResults`, SfePy and
GOLEM results can use the same result plotting functions.

### Plot Materials

> **Workbench component:** `Plot Materials`

This is a pre-flight inspection of the problem definition. It plots the
mesh by material so that geological and fault assignments can be checked
before running a potentially expensive simulation. This is mainly used
when SfePy simulation in combination with faults is used.

Example call in the codebase:

``` python
plot_builder_materials(builder)
```

For SfePy, this can show lithology and damage-zone assignments.

### Plot a variable at a time step

> **Workbench component:** `Plot Variable (Final Time)`

Produces a 3D visualisation of a result variable at a selected time.

Example call in the codebase:

``` python
plot_variable_at_a_time(
    simulation_result,
    "T",
    time=final_time,
    cmap="coolwarm",
    show_edges=True,
)
```

### Plot a cross-section

> **Workbench component:** `Plot Cross Section 2D`

Slices a result variable on a specified plane and allows initial and
final states to be compared.

Example call in the codebase:

``` python
plot_cross_section_2D(
    simulation_result,
    "temperature",
    origin=(500, 500, 500),
    normal=(0, 1, 0),
    cmap="coolwarm",
)
```

### View solver output (Only for Spefy)

> **Workbench component:** `View Solver Output`

For SfePy, solver output can be stored in
`SimulationResults.sfepy_stdout`, with stages such as `"pressure"`,
`"heat"`, or `"custom"`.

Example call in the codebase:

``` python
for stage, stdout in data_by_time.sfepy_stdout.items():
    print(f"=== {stage} ===")
    print(stdout)
```

### Additional codebase-only plots and inspection

The following post-processing functions are available directly in the
codebase. They operate on `SimulationResults` and can therefore be used
with either SfePy or GOLEM results when the requested variable is
available.

#### Plot the difference between two time steps

`plot_variable_difference()` displays the change in a point-data
variable between two saved time steps as `time_b - time_a`. If the two
times are not specified, the first and last saved time steps are used.
The colour scale is centred symmetrically around zero so that increases
and decreases can be distinguished directly.

Example call in the codebase:

``` python
plot_variable_difference(
    simulation_result,
    "temperature",
    time_a=initial_time,
    time_b=final_time,
    cmap="RdBu_r",
    show_edges=False,
)
```

#### Plot a cross-section difference

`plot_cross_section_difference_2D()` plots the change in a point-data
variable between two time steps on a specified 2D section. The results
are calculated as `time_b - time_a`. By default, the first and last
saved time steps are compared.

Example call in the codebase:

``` python
plot_cross_section_difference_2D(
    simulation_result,
    "temperature",
    time_a=initial_time,
    time_b=final_time,
    origin=(500, 500, 500),
    normal=(0, 1, 0),
    cmap="RdBu_r",
)
```


#### Plot a variable along a line

`plot_variable_along_line()` samples a variable along a straight line
between two points at one time step and plots the sampled values
against distance along the line.

Example call in the codebase:

``` python
plot_variable_along_line(
    simulation_result,
    "temperature",
    time=final_time,
    p0=(0, 500, 500),
    p1=(1000, 500, 500),
    n_samples=200,
)
```

Sampling is performed from the simulation mesh using PyVista.

#### Print a variable at a point

`print_variable_at_point()` samples a variable at a specified point and
time. 

Example call in the codebase:

``` python
value = print_variable_at_point(
    simulation_result,
    "temperature",
    time=final_time,
    point=(500, 500, 500),
)

print(value)
```

If the requested point lies outside the mesh, the function logs a
warning and returns `None`.

#### Plot a time series at a point

`plot_variable_time_series()` samples a variable at the same spatial
point for all saved simulation time steps and plots the resulting time
series.

Example call in the codebase:

``` python
plot_variable_time_series(
    simulation_result,
    "temperature",
    point=(500, 500, 500),
    decimals=3,
)
```

If the point cannot be sampled at a particular time step, that value is
represented as a gap in the plotted time series.

------------------------------------------------------------------------

## Output

Both solver pipelines converge to the same WBGeo result type:

``` text
SfePy:
Build Hydrothermal Problem ─┐
                            ├─> Run Simulation ──────────────┐
Build Custom SfePy Problem ─┘                                │
                                                             ├─> SimulationResults
GOLEM:                                                       │
Build GOLEM Hydrothermal Problem ─┐                          │
Build GOLEM Simple Reservoir ─────┼─> Run GOLEM Simulation ─┘
Build Custom GOLEM Problem ───────┘
                                                                  │
                                                                  ├─> Visualisation / Inspection
                                                                  └─> Export Simulation Results
```

`SimulationResults` is a time-dependent container for geometry,
topology, and numerical fields. It includes:

-   **`nodes_by_time`** --- nodal coordinates for each saved time step.
-   **`cells_by_time`** --- cell connectivity.
-   **`celltypes_by_time`** --- VTK-style cell type identifiers.
-   **`node_data_by_time`** --- node-based result variables.
-   **`cell_data_by_time`** --- cell-based result variables.
-   **`sfepy_stdout`** --- solver-specific recorded output where
    provided by the runner.

The important architectural distinction is therefore at the **build and
run stages**. SfePy executes natively in the WBGeo Python environment,
while GOLEM is staged and executed externally through Docker/MPI. After
execution, both are represented as `SimulationResults` and can use the
common WBGeo visualisation and export layer.

\`\`\`
