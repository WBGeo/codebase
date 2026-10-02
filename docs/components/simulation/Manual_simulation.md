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
input file. The custom file specifies a user-defined problem with its
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
`.i` input file. A custom run can use a WBGeo mesh already held in Python,
one level of a hierarchical WBGeo mesh, or an existing mesh file supplied
by path. The `.i` file selects the corresponding mesh source through one
of `${GOLEM_MESH_FILE}`, `${GOLEM_MESH1_FILE}`, or
`${GOLEM_MESH2_FILE}`.

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

> **Cluster execution:** The runner documented here currently implements the Docker/MPI backend only. A GOLEM installation available directly on an HPC cluster is not yet selected automatically by `run_simulation_golem()`. Native SLURM execution (for example, launching GOLEM with `srun` across allocated nodes) requires a separate SLURM/native backend in the runner. The WBGeo problem builders and custom-input staging concepts can be reused by that backend.

------------------------------------------------------------------------

## Workflow

> **Examples:** Examples 1 and 2 demonstrate how to use the two
> simulation pipelines available in WBGeo: **SfePy** and
> **GOLEM/MOOSE**. They provide practical examples of building, running,
> and working with simulation problems using these solvers.

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

The simple-reservoir builder supports two synthetic fractures, a well, an injection source,
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
`CustomGolemBuilder`. A custom problem uses **exactly one mesh source per
run**. This can be a WBGeo mesh object, one level of a hierarchical mesh,
or an already existing mesh file on disk.

The supported mesh inputs are:

| Builder input | Placeholder in the `.i` file | Behaviour |
| --- | --- | --- |
| `mesh_results=...` | `${GOLEM_MESH_FILE}` | Export the WBGeo mesh to Exodus and run a normal custom simulation. |
| `mesh1=...` | `${GOLEM_MESH1_FILE}` | Export and solve the outer/big hierarchical mesh. Hierarchy sampling can be injected for this mode. |
| `mesh2=...` | `${GOLEM_MESH2_FILE}` | Export and solve the inner/small hierarchical mesh as a normal custom simulation. |
| `mesh_file="/path/to/mesh.exo"` | `${GOLEM_MESH_FILE}` | Use an existing mesh file directly. The runner copies/stages the file and does not export it again. |

Other relevant inputs are:

-   **`input_file_contents`**: Complete text of the custom GOLEM/MOOSE
    `.i` file.
-   **`input_file_path`** *(optional)*: Path of the original `.i` file.
    For a mesh1 hierarchy run, WBGeo can save the hierarchy-augmented
    input beside it using the suffix `_hierarchy_sampling`. This path is
    not used to discover or stage auxiliary input files.
-   **`auxiliary_files_dir`** *(optional)*: Directory containing auxiliary
    files required by a custom GOLEM/MOOSE input, for example time-series
    CSV files. The directory must already exist. The runner copies its
    complete contents into a `files/` directory in the GOLEM working
    directory. The custom `.i` file should therefore reference these files
    with relative paths such as `files/injection_flowrate.csv` or
    `files/frac_aperture.csv`.
-   **`mesh_file_name`** *(optional)*: Staging name to use inside the GOLEM
    run when `mesh_file` is supplied. It must be a file name, not a path.
    If omitted, the source mesh file name is retained.
-   **`geomodel_result`** *(optional)*: Structural model when required by
    the surrounding workflow.
-   **`print_linear_residuals`**: Controls residual output.
-   **`hierarchy_sample_nodes`** *(mesh1 only)*: Hierarchical node/coordinate
    information returned by `hierarchical_mesh()` and used to construct
    GOLEM point samplers.
-   **`sample_variables`** *(optional, mesh1 hierarchy sampling)*: Variables
    written by the hierarchy samplers. The default is
    `['pore_pressure', 'temp']`.

The mesh argument and placeholder must match. For example,
`mesh1=mesh1_structural` requires `${GOLEM_MESH1_FILE}` in the `.i` file;
it must not reference `${GOLEM_MESH2_FILE}`. Likewise,
`mesh_file=...` uses `${GOLEM_MESH_FILE}`.

#### Auxiliary files for a custom GOLEM input

A custom `.i` file may depend on additional files besides its mesh, such as
CSV files defining a transient injection rate or fracture aperture. Pass
the directory containing those files explicitly with `auxiliary_files_dir`.
WBGeo does not infer this directory from `input_file_path`.

For example, with the host layout:

``` text
simulation_files/
├── transient.i
└── files/
    ├── injection_flowrate.csv
    └── frac_aperture.csv
```

build the problem as follows:

``` python
with open("/path/to/simulation_files/transient.i", encoding="utf-8") as f:
    transient_input = f.read()

problem = CustomGolemBuilder(
    input_file_contents=transient_input,
    input_file_path="/path/to/simulation_files/transient.i",
    auxiliary_files_dir="/path/to/simulation_files/files",
    mesh_file="/path/to/existing/GSB_mesh.exo",
    print_linear_residuals=True,
)
```

The `.i` file can then use paths relative to the GOLEM working directory:

``` text
files/injection_flowrate.csv
files/frac_aperture.csv
```


#### Existing mesh file by address

An existing mesh does not need to be present as a `MeshResults` object in
Python. Give its address directly with `mesh_file`:

``` python
with open("/path/to/custom_reservoir.i", encoding="utf-8") as f:
    custom_golem_input = f.read()

problem = CustomGolemBuilder(
    input_file_contents=custom_golem_input,
    input_file_path="/path/to/custom_reservoir.i",
    mesh_file="/path/to/existing/GSB_mesh.exo",
    # Optional name used when staging the mesh for GOLEM:
    mesh_file_name="reservoir.exo",
    print_linear_residuals=True,
)
```

The corresponding mesh entry in the `.i` file is:

``` text
[Mesh]
  [original_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
[]
```

The runner copies the existing file into the GOLEM staging directory and
replaces `${GOLEM_MESH_FILE}` with its container path. It does **not**
call the WBGeo Exodus exporter for this mode. Because an external file
does not carry WBGeo `MeshResults` metadata, WBGeo performs the generic
required-block validation and GOLEM/MOOSE performs the detailed
mesh/block/material compatibility checks during execution.

For the normal in-memory WBGeo case, use `mesh_results` instead:

``` python
problem = CustomGolemBuilder(
    input_file_contents=custom_golem_input,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    print_linear_residuals=True,
)
```

In this case the runner exports the `MeshResults` object to Exodus and
performs WBGeo's detailed material/block validation before Docker/MPI is
started.

#### Hierarchical custom GOLEM simulation

Hierarchical simulations are executed one mesh level at a time. A single
`CustomGolemBuilder` call does not solve `mesh1` and `mesh2` together.
The intended sequence is:

1.  Solve the outer/big mesh with `mesh1=...` and
    `${GOLEM_MESH1_FILE}`.
2.  During the mesh1 run, optionally sample the solution at boundaries or inside of the small mesh using `hierarchy_sample_nodes`. These data are considered post-processing data and the solutions are obtained on these data are saved in CSV files.
3.  Use the generated hierarchy CSV data as boundary or initial conditions in small mesh.
4.  Solve the inner/small mesh separately with `mesh2=...` and
    `${GOLEM_MESH2_FILE}`. The mesh2 run is otherwise a normal custom
    GOLEM run and does not automatically inject the mesh1 hierarchy
    samplers.

Example mesh1 solve:

``` python
problem_mesh1 = CustomGolemBuilder(
    input_file_contents=reservoir_input,
    input_file_path=reservoir_input_path,
    mesh1=mesh1_structural,
    geomodel_result=structural_model_result,
    hierarchy_sample_nodes=nodes_bcs_structural,
    sample_variables=["pore_pressure", "temp"],
    print_linear_residuals=True,
)

mesh1_results = run_simulation_golem(
    problem=problem_mesh1,
    image_name="moose-dev:fixed",
    project_dir="/path/to/projects",
    docker_dir="/projects",
    output_dir="/path/to/mesh1_output",
    nproc=4,
)
```

The mesh1 `.i` file must contain:

``` text
file = ${GOLEM_MESH1_FILE}
```

When `hierarchy_sample_nodes` is supplied, WBGeo adds GOLEM
`PointValueSampler` entries for the available hierarchy groups. These
include the `inside_nodes` group and the available boundary-face groups
from the big-mesh frame. The samplers use 
`execute_on = TIMESTEP_END`. CSV output is enabled if the input does not
already provide a CSV output.

The raw sampler `id` column is the identifier written by GOLEM's
`PointValueSampler` (the containing element id); it should not be treated
as the WBGeo hierarchy `NodeID`. The transfer locations themselves come
from the hierarchy coordinate arrays supplied to the builder. The retained hierarchy data are written in both the big-mesh and small-mesh coordinate systems.



A separate mesh2 run uses:

``` python
problem_mesh2 = CustomGolemBuilder(
    input_file_contents=small_input,
    input_file_path=small_input_path,
    mesh2=mesh2_structural,
    geomodel_result=structural_model_result,
    print_linear_residuals=True,
)
```

and the corresponding `.i` file contains:

``` text
file = ${GOLEM_MESH2_FILE}
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

Important runner inputs for the current **Docker backend** are:

-   **`problem`**: GOLEM problem produced by one of the GOLEM builders.
-   **`image_name`**: Docker image containing GOLEM.
-   **`project_dir`**: Host project directory mounted into the
    container.
-   **`docker_dir`**: Corresponding path inside the container.
-   **`output_dir`** *(optional)*: Host destination for final results.
-   **`nproc`**: Number of MPI processes.
-   **`keep_files_dir`** *(optional)*: Directory in which
    staged/generated files can be retained for debugging.

The current function signature is Docker-specific: `image_name`, `project_dir`, and `docker_dir` are required by this backend. Native SLURM execution is not yet an alternative backend in the runner. When such a backend is added, WBGeo should be started once by the batch job and only the GOLEM executable should be launched with `srun`/MPI across the allocated tasks.

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

1.  Selects the single configured mesh source. WBGeo `mesh_results`,
    `mesh1`, and `mesh2` objects are exported to Exodus; `mesh_file` is
    copied directly without re-export.
2.  Generates or stages the GOLEM/MOOSE `.i` input. For a custom problem
    with `auxiliary_files_dir`, copies the contents of that directory into
    `files/` in the GOLEM working directory.
3.  For a custom `mesh1` run with hierarchy sampling data, injects the
    hierarchy `PointValueSampler` definitions and ensures CSV output is
    available.
4.  Validates the input. WBGeo meshes receive detailed material/block
    validation; external mesh files receive generic input-structure
    validation because no `MeshResults` metadata is available.
5.  Replaces the matching mesh placeholder (`${GOLEM_MESH_FILE}`,
    `${GOLEM_MESH1_FILE}`, or `${GOLEM_MESH2_FILE}`) with the staged
    container path.
6.  Executes GOLEM in the configured Docker environment using MPI.
7.  Copies the complete GOLEM working output back to `output_dir`, while
    removing internal `*_cp` checkpoint directories.
8.  Post-processes hierarchy sampler CSV files when hierarchy sampling is
    active (described below).
9.  Loads the resulting Exodus database into `SimulationResults` and
    preserves the native Exodus/output-directory paths for later export.
10. Cleans temporary host/container resources while retaining the
    user-facing outputs.

#### Hierarchical GOLEM CSV output and post-processing

For a mesh1 hierarchy run, GOLEM/MOOSE writes one sampler
CSV per hierarchy group **for each written time step**. Typical names are:

``` text
hierarchy_inside_0001.csv
hierarchy_top_0001.csv
...
```

The runner keeps only the highest-numbered hierarchy CSV for each sampling group, so the retained file represents the last written sampler time step. The hierarchy groups come from the available transfer
points: `inside` plus boundary groups such as `top`, `bottom`, `left`,
`right`, `front`, and `back` when present. These data are provided in the coordinate systems of both the big and small meshes in separate directories.

The resulting hierarchy part of `output_dir` therefore has the form:

``` text
output_dir/
├── big_frame/
│   ├── hierarchy_inside_....csv
│   ├── hierarchy_top_....csv
│   └── ...
└── small_frame/
    ├── hierarchy_inside_....csv
    ├── hierarchy_top_....csv
    └── ...
```

`big_frame` and `small_frame` contain the **same sampled solution rows**
expressed in the two coordinate frames.



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

> **Note:** Model 1 and Model 2 in the examples demonstrate how to use
> the **GOLEM/MOOSE** and **SfePy** simulation workflows. For the
> **GOLEM Simple Reservoir Problem**, the simulation results can be
> exported as an **Exodus** file and opened in **ParaView**. For
> improved visualisation of the reservoir results, use the provided
> ParaView state file `reservoir_visualisation.pvsm`.

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
normally includes the native Exodus result, `golem_stdout.txt`,
`wbgeo_generated.i`, the staged mesh file (`wbgeo_mesh.exo`,
`wbgeo_mesh1.exo`, `wbgeo_mesh2.exo`, or the chosen external
`mesh_file_name`), and additional GOLEM/MOOSE outputs such as CSV files.
For hierarchy sampling, the final retained sampler CSVs are additionally
organized under `big_frame/` and `small_frame/` as described above.

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
> `export_simulation_results' is used, use`format="golem"`to save the whole preserved output directory as a ZIP. Use`format="exodus"`only when the single native Exodus result is specifically desired. For SfePy, use `export_simulation_results` with `format="vtk"` to save the zipped VTK time series.

------------------------------------------------------------------------

## Visualisation and Inspection

Once a simulation has been converted to `SimulationResults`, SfePy and
GOLEM results can use the same result plotting functions.

For the **GOLEM Simple Reservoir Problem**, the saved ParaView state
file `reservoir_visualisation.pvsm` can be opened in **ParaView** for
more detailed visualisation of the simulation results. The state file
provides a prepared visualisation setup for inspecting the reservoir
simulation output and can be used in addition to the plotting functions
available directly in WBGeo.

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

### View solver output (SfePy only)

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
between two points at one time step and plots the sampled values against
distance along the line.

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
while the current GOLEM runner stages and executes GOLEM externally through Docker/MPI. Native SLURM execution is a planned/alternative runner backend rather than functionality provided by the current implementation. After execution, both solver pipelines are represented as `SimulationResults` and can use the common WBGeo visualisation and export layer.

\`\`\`
