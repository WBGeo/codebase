# Simulation

The simulation component connects the Workbench to numerical solvers.
Since the meshing component exports the mesh to different formats, WBGeo
can interface with various simulation packages. The current simulation
architecture provides two solver pipelines:

-   **SfePy** (Simple Finite Elements in Python) for Python-native
    finite-element simulations, supporting the built-in hydrothermal
    workflow as well as user-supplied SfePy problems.
-   **GOLEM** for hydrothermal and simple-reservoir simulations,
    as well as user-supplied GOLEM problems, executed either in a
    Docker container or by a GOLEM installation on the host.

Both pipelines follow the same general Workbench pattern:

`Build problem -> Run simulation -> SimulationResults -> Visualise / Export`


------------------------------------------------------------------------

## Concepts

Before a model can be solved, it needs a **problem definition**: the
mesh together with material properties, boundary and initial conditions,
solver settings, and the solver-specific description of the governing
equations.

The Workbench currently provides five problem-building routes.

### SfePy problem definitions

**SfePy Hydrothermal Problem** --- auto-generated from the mesh and structural
model. Material regions and boundary regions are inferred from the
lithology mapping and mesh information. The built-in formulation solves
simplified single-phase Darcy flow followed by advective-conductive heat
transport.

**Custom SfePy Problem** --- wraps a complete user-supplied SfePy `.py`
input file. The custom file specifies a user-defined problem with its
variables, materials, equations and solvers. 


### GOLEM problem definitions

**GOLEM Hydrothermal Problem** --- auto-generates a GOLEM
hydrothermal problem from a WBGeo mesh, structural model, rock
properties, fluid properties, boundary conditions, and optional fault
properties.

**GOLEM Simple Reservoir** --- builds a dedicated reservoir
model for GOLEM, including the reservoir mesh and support for synthetic
fractures, a well, and injection/extraction source behaviour.

**GOLEM Custom Problem** --- wraps a complete user-supplied GOLEM
`.i` input file. A custom run can use a WBGeo mesh already held in Python,
or an existing mesh file supplied by path. The `.i` file uses `${GOLEM_MESH_FILE}` to reference the mesh.


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

SfePy and GOLEM also differ at execution time.

**SfePy** runs inside the WBGeo Python environment. For the built-in
hydrothermal problem, pressure and temperature are solved sequentially:
first the Darcy pressure problem, then transient heat transport using
the resulting Darcy velocity. A custom SfePy problem instead follows the
execution defined by the supplied Python input.

**GOLEM** runs externally to the WBGeo Python environment. The GOLEM runner
prepares the mesh and input files, validates their compatibility, and executes
the simulation either in a Docker container or directly on the host.

> **Execution environments:** `run_simulation_golem()` supports two execution modes:
>
> - **Docker execution:** When `image_name` is provided, GOLEM runs inside a
>   temporary Docker container created from the specified image. This mode
>   is available in the codebase. The required
>   input files are staged in the container, and the simulation results are
>   copied back to the host. The container is removed after execution.
> - **Host execution:** GOLEM runs directly on the host.
>   The workbench with visual editor uses this mode with a remote mpiexec.
>
> In codebase, the `golem_binary` parameter specifies the GOLEM executable path in the
> selected environment. The `workspace_dir`, `output_dir`, `nproc`, and
> `temporary` parameters control the working directory, output location,
> MPI process count, and temporary-file cleanup, respectively.

------------------------------------------------------------------------

## Workflow

> **Examples:** Examples 1 and 2 in **codebase** demonstrate how to use the two
> simulation pipelines available in WBGeo: **SfePy** and
> **GOLEM**. They provide practical examples of building, running,
> and working with simulation problems using these solvers.

### 1. Build the problem definition

Choose the builder corresponding to the solver and model type.

### SfePy: Auto-generated Hydrothermal Problem

> **Workbench component:** `Build SfePy Hydrothermal Problem`

In the Workbench, this component requires connections to both a mesh and
the structural model used to generate that mesh.

The component accepts the following parameters:
-   **`options`** (by clicking on **Update Controls**): Per-rock-unit properties (porosity,
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
-   **`p_top`**, **`p_bottom`** *(optional)*: Pressure boundary values
    at the model top/bottom, in Pa. Default `p_top=0.0`, `p_bottom=1e6`.
-   **`t_top`**, **`t_bottom`** *(optional)*: Temperature boundary
    values at the model top/bottom, in °C. Default `t_top=10.0`,
    `t_bottom=60.0`.
-   **`initial_temperature`** *(optional)*: Constant initial temperature
    in °C. Default empty: a linear profile from `t_bottom` to `t_top`.
-   **`start_time`**, **`end_time`**, **`num_steps`** *(optional)*:
    Start time, end time, and number of uniform steps for the transient
    heat stage. Default `start_time=0.0`, `num_steps=3`; `end_time`
    defaults to the model's thermal diffusion timescale (roughly the time
    to reach thermal equilibrium). Same names and defaults as the GOLEM
    hydrothermal builder.
-   **`print_linear_residuals`** *(optional)*: Include the linear solver
    residuals in the solver output. Default `False`.
-   **`linear_solver`** *(optional)*: `"iterative"` (default) scales to
    large meshes; `"direct"` is exact but does not scale well --- use it
    for small or debug meshes.
-   **`linear_solver_i_max`**, **`linear_solver_eps_r`** *(optional)*:
    Maximum iterations and relative residual tolerance for the iterative
    linear solver. Default `500` and `1e-8`.

In the codebase, the mesh and structural model are provided through two
additional parameters:
-   **`mesh_results`**: The mesh to solve on (implicit, structured, or
    unstructured).
-   **`geomodel_result`**: The structural model.

Example call in the codebase:

``` python
rock_properties = {
    "basement": RockUnitProperties(name="basement", porosity=0.10, permeability=1e-13, k_solid=50, rho_c_solid=3900000),
    "rock1": RockUnitProperties(name="rock1", porosity=0.30, permeability=1e-11, k_solid=10, rho_c_solid=800000),
    "rock2": RockUnitProperties(name="rock2", porosity=0.20, permeability=1e-12, k_solid=20, rho_c_solid=3400000),
}
fluid = FluidProperties(mu=1.0e-3, k_fluid=15, rho_c_fluid=2400000)

builder = SfepyHydrothermalProblemBuilder(
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    include_flow=True,
    end_time=5e11,
    num_steps=3,
)
```

Here, `rock_properties` gives each lithology its own porosity,
permeability, thermal conductivity, and heat capacity; `fluid` describes
the pore fluid. `end_time=5e11` overrides the default diffusion-timescale
estimate for the heat stage's end time, solved over `num_steps=3` steps.

A faulted model additionally enables the damage zone:

``` python
fault_zone_properties = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2300000
)

builder = SfepyHydrothermalProblemBuilder(
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
-   Engineering objects such as wells and point sources cannot be
    physically solved by this built-in workflow.

### SfePy: User-supplied Problem

> **Workbench component:** `Build SfePy Custom Problem`

This component brings a complete SfePy `.py` input into the Workbench.
In the Workbench, this component requires connections to both a mesh and
the structural model used to generate that mesh.

The component accepts the following parameters:
-   **`input_file`**: Path to a complete SfePy input file (`.py`).
-   **`fault_zone_n_voxels`** *(optional)*: If set, gives cells within
    this many structural-grid voxels of any active fault's trace their
    own region id --- the same damage-zone cell selection
    `Build SfePy Hydrothermal Problem` uses, referenceable from the custom
    file as `cells of group N`.

In the codebase, the mesh and structural model are provided through two
additional parameters:

-   **`mesh_results`**: The mesh to solve on. Meshes containing wells or
    sources are rejected --- not supported by this component.
-   **`geomodel_result`**: The structural model.

In the codebase, the properties configured through **`options`** in the
Workbench are provided through three separate parameters:

- **`rock_properties`**: Per-rock-unit hydrothermal properties,
  defined using `RockUnitProperties`.
- **`fluid`**: Fluid properties, defined using `FluidProperties`.
- **`fault_zone_properties`** *(optional)*: Properties of the
  volumetric fault damage zone, defined using `RockUnitProperties`.

The builder validates the mesh type, lithology mapping, and referenced cell
groups. The equations, material definitions, and solver settings are validated by 
SfePy during simulation execution.

Example call in the codebase:

``` python
with open("examples/synthetic_examples/model1/input_data/simulation_files/custom_hydrothermal.py", encoding="utf-8") as f:
    custom_input_file_contents = f.read()

custom_builder = SfepyCustomBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
)
```

**SfePy custom problem limitations:**

-   Faults can only be represented as a combined volumetric damage zone
    rather than explicit interface elements.
-   Engineering objects such as wells and point sources cannot be
    physically solved. Only 3D elements can be solved. 

### GOLEM: Auto-generated Hydrothermal Problem

> **Workbench component:** `Build GOLEM Hydrothermal Problem`

This component constructs a `GolemHydrothermalProblemBuilder` from a
WBGeo mesh and structural model. It uses rock and fluid properties
analogous to the SfePy hydrothermal workflow, while supporting explicit
GOLEM fault properties when suitable 2D fault blocks are present in an
unstructured mesh.
In the Workbench, this component requires connections to both a mesh and
the structural model used to generate that mesh.
Important inputs and controls in Workbench include:
-   **`options`** (by clicking on **Update Controls**): Per-rock-unit properties (porosity,
    permeability, thermal conductivity, volumetric heat capacity),
    global fluid properties, and fault-zone properties.
-   **`p_top`**, **`p_bottom`**: Pressure boundary values.
-   **`t_top`**, **`t_bottom`**: Temperature boundary values.
-   **initial pressure / temperature**: Initial-state controls.
-   **`start_time`**, **`end_time`**, **`num_steps`**: Time controls,
    with the same names and defaults as the SfePy hydrothermal builder.
    The step size is `(end_time - start_time) / num_steps`; an empty
    `end_time` uses the model's thermal diffusion timescale.
-   **`output_interval`**: Result-output frequency.
-   **`print_linear_residuals`**: Include GOLEM's linear residuals in
    the solver output. Default `False`.

In the codebase, the mesh and structural model are provided through two
additional parameters:
-   **`mesh_results`**: Mesh used by GOLEM (implicit, structured, or
    unstructured).
-   **`geomodel_result`**: Structural model.
In the codebase, the properties configured through **`options`** in the
Workbench are provided through three separate parameters:
-   **rock properties**: Geological-unit hydrothermal properties.
-   **fluid properties**: Properties of the fluid phase.
-   **fault properties**: `GolemFaultProperties` for explicit 2D fault
    blocks.

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


### GOLEM: Simple Reservoir

> **Workbench component:** `Build GOLEM Simple Reservoir`

The simple-reservoir builder supports two synthetic fractures, a well, an injection source,
geological faults already present in the structural model, local mesh
refinement, material/fluid properties, transient controls, and low-level
PETSc solver settings.


In the Workbench, this component requires connections to a
structural model. The unstructured mesh is created in the builder.

It works for any structural model large enough for the layout. The
reservoir mesh fills the model's extent, and a doublet of physical size
is placed in it: two square fracture planes (`fracture_size`, default
100 m) normal to x, `fracture_spacing` apart (default 200 m), centred on
`layout_center` (default: the model centre), with a well at the second
plane. Injection and extraction act at the plane centres. Sizes follow
what they belong to:

-   the fracture and well refinement is in metres around those objects,
    the same in every model;
-   the background `mesh_size` defaults to 3 % of the model's smallest
    axis (30 m for a 1 km model), so large models don't produce huge
    meshes.

The layout plus its refinement zone must fit into the model on every
axis; otherwise the builder stops before meshing and names the axis and
the missing space.

The component accepts the following parameters in the Workbench:
-   **`mesh_size`** *(optional)*: Target volume-mesh size (m). Default
    empty: 3 % of the model's smallest axis.
-   **`curve_mesh_size`** *(optional)*: Target mesh size on well curves.
    Default `5.0`.
-   **`well_refine_size_min`** *(optional)*: Minimum target mesh size
    close to the well. Default `5.0`.
-   **`well_refine_size_max`** *(optional)*: Maximum target mesh size
    away from the well. Default `30.0`.
-   **`well_refine_dist_min`** *(optional)*: Inner distance for well
    refinement. Default `10.0`.
-   **`well_refine_dist_max`** *(optional)*: Outer distance for well
    refinement. Default `60.0`.
-   **`fracture_refine_size_min`** *(optional)*: Minimum target mesh
    size close to the synthetic fractures. Default `5.0`.
-   **`fracture_refine_size_max`** *(optional)*: Maximum target mesh
    size away from the fractures. Default `30.0`.
-   **`fracture_refine_dist_min`** *(optional)*: Inner distance for
    fracture refinement. Default `10.0`.
-   **`fracture_refine_dist_max`** *(optional)*: Outer distance for
    fracture refinement. Default `60.0`.
-   **`scale_refinement_with_model`** *(optional, codebase only)*: When `True`, scales well and fracture refinement sizes and distances with the smallest model axis relative to a 1000 m reference model (the old behaviour). Default `False`: the refinement values are metres.
-   **`fracture_size`** *(optional)*: Edge length of the two square fracture planes (m). Default `100.0`.
-   **`fracture_spacing`** *(optional)*: Distance between the two fracture planes along x (m). Default `200.0`.
-   **`layout_center`** *(optional)*: `(x, y, z)` centre of the fracture pair. Default: the model centre.
-   **`injection_point`**, **`extraction_point`** *(optional)*: `(x, y, z)` source locations. Default: the centres of the first and second fracture.
-   **`wells`** *(optional)*: Well path(s), e.g. from `Load Well` (a plain list, not hierarchical well data). Default: the model 1 well shape at the second fracture.
-   **`rock_porosity`** *(optional)*: Rock porosity. Default `0.1`.
-   **`rock_permeability`** *(optional)*: Rock permeability. Default
    `1.0e-15`.
-   **`rock_density`** *(optional)*: Rock density. Default `2.6e3`.
-   **`rock_thermal_conductivity`** *(optional)*: Rock thermal
    conductivity. Default `3.0`.
-   **`rock_heat_capacity`** *(optional)*: Rock heat capacity. Default
    `950.0`.
-   **`fracture_porosity`** *(optional)*: Porosity assigned to the two
    internally generated synthetic fractures. Default `1.0`.
-   **`fracture_aperture`** *(optional)*: Aperture of the synthetic
    fractures. Default `1.0e-4`.
-   **`geological_fault_aperture`** *(optional)*: Aperture/scaling
    required by GOLEM for exported lower-dimensional geological-fault
    blocks. Geological faults otherwise use the common rock properties
    above. Default `1.0`.
-   **`well_porosity`** *(optional)*: Well porosity. Default `1.0`.
-   **`well_permeability`** *(optional)*: Well permeability. Default
    `3.12e-4`.
-   **`well_scaling_factor`** *(optional)*: GOLEM scaling factor used
    for the well block. Default `0.05`.
-   **`fluid_density`** *(optional)*: Fluid density. Default `1.0e3`.
-   **`fluid_viscosity`** *(optional)*: Fluid dynamic viscosity. Default
    `1.0e-3`.
-   **`fluid_heat_capacity`** *(optional)*: Fluid heat capacity. Default
    `4.2e3`.
-   **`fluid_thermal_conductivity`** *(optional)*: Fluid thermal
    conductivity. Default `0.65`.
-   **`fluid_modulus`** *(optional)*: Fluid modulus. Default `2.5e9`.
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
-   **`injection_rate`** *(optional)*: Injection source rate. Default
    `3.0`.
-   **`extraction_rate`** *(optional)*: Extraction source rate. Default
    `3.0`.
-   **`start_time`** *(optional)*: Simulation start time. Default `0.0`.
-   **`end_time`** *(optional)*: Simulation end time. Default `None`.


In the codebase, the structural model results and the following additional parameters are accepted:

#### Structural-model input

-   **`geomodel_result`**: Required structural model used to determine
    the model extent and construct the reservoir geometry.

#### Unstructured-mesh inputs

-   **`mapping_litho`** *(optional)*: Lithology mapping method. Default
    `"automatic_centers"`.
-   **`gmsh_flag`** *(optional)*: Option to save the original mesh created in Gmsh. Default `False`.
-   **`merge_file`** *(optional)*: It is required if `mapping_litho` is set to `manual`. Default `None`.
-   **`unstructured_mesh_options`** *(optional)*: Dictionary of
    additional keyword arguments accepted by
    `create_unstructured_mesh_data()`. Default `None`.

For example, additional meshing controls can be supplied as:

``` python
unstructured_mesh_options={
    "tolerance": 50,
}
```

#### Optional geometry overrides

Normally, the builder uses the complete structural-model extent and
places the doublet described above. In the codebase, the reservoir extent
and the fracture planes can also be given as explicit coordinates:

-   **`reservoir_extent_override`** *(optional)*: Explicit
    `(xmin, xmax, ymin, ymax, zmin, zmax)` reservoir extent. Default
    `None`.
-   **`fracture_planes_override`** *(optional)*: Explicit
    synthetic-fracture plane coordinates. Default `None`.
-   **`wells_override`** *(optional)*: Explicit well coordinates (the
    Workbench input `wells`). Default `None`.

#### Injection and extraction inputs

-   **`injection_point`** *(optional)*: Explicit `(x, y, z)` injection
    location. If `None`, the automatically positioned centre of the
    first synthetic fracture is used.
-   **`extraction_point`** *(optional)*: Explicit `(x, y, z)` extraction
    location. If `None`, the automatically positioned centre of the
    second synthetic fracture is used.
-   **`source_start_time`** *(optional)*: Time at which the
    injection/extraction sources become active. Default `0.0`.
-   **`source_end_time`** *(optional)*: Time at which the sources stop.
    Default `None`; when omitted, the builder sets it to the resolved simulation `end_time`, so the sources remain active until the simulation ends.

#### Transient-simulation inputs

-   **`time_sequence`** *(optional)*: Explicit sequence of solve times.
    If `None`, the builder uses its default simple-reservoir time
    sequence.
-   **`automatic_scaling`** *(optional)*: Enables MOOSE automatic
    scaling. Default `True`.
-   **`compute_scaling_once`** *(optional)*: Controls the GOLEM/MOOSE executioner option of the same name, determining whether automatic scaling is computed only once rather than updated during the solve. Default `False`.

#### PETSc solver controls

The following advanced inputs expose the solver configuration:

-   **`coupled_petsc_options_iname`** and
    **`coupled_petsc_options_value`**: PETSc options for the coupled
    nonlinear solve.
-   **`pressure_petsc_options_iname`** and
    **`pressure_petsc_options_value`**: PETSc options for the pressure
    field solve.
-   **`temperature_petsc_options_iname`** and
    **`temperature_petsc_options_value`**: PETSc options for the
    temperature field solve.

#### Output and diagnostic controls

-   **`print_linear_residuals`** *(optional)*: Print linear residual
    information. Default `True`.
-   **`print_nonlinear_converged_reason`** *(optional)*: Print the
    nonlinear convergence reason. Default `True`.
-   **`perf_graph`** *(optional)*: Enables GOLEM/MOOSE performance-graph reporting in the generated output configuration. Default `True`.

Example call in the codebase:

``` python
reservoir_problem_model2 = GolemSimpleReservoirBuilder(
    geomodel_result=structural_model_result,

    # Unstructured mesh
    mesh_size=50.0,
    curve_mesh_size=5.0,
    mapping_litho="automatic_centers",
    gmsh_flag=False,
    merge_file=None,
    unstructured_mesh_options={
        "tolerance": 50,
    },

    # Automatic geometry;
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

    # Output/diagnostics
    print_linear_residuals=True,
    print_nonlinear_converged_reason=True,
)
```

> **Note:** model1 and model2 in the examples demonstrate how to use
> the **GOLEM** and **SfePy** simulation workflows. For the
> **GOLEM Simple Reservoir**, the simulation results can be
> exported as an **Exodus** file and opened in **ParaView**. For
> improved visualisation of the reservoir results, use the provided
> ParaView state file `model1_reservoir_visualisation.pvsm` and `model2_reservoir_visualisation.pvsm`.


### GOLEM: User-supplied Problem

> **Workbench component:** `Build GOLEM Custom Problem`

It accepts a complete user-provided GOLEM `.i` file.
In the Workbench, this component requires connections to both a mesh and
the structural model used to generate that mesh.


The component accepts the following parameters in the Workbench:

-   **`input_file`**: Path (relative to the codebase root) to the
    complete GOLEM/MOOSE `.i` file.
-   **`print_linear_residuals`** *(optional)*: Include GOLEM's linear
    residuals in the solver output. Default `True`.

In the codebase, the builder takes the file content instead of a path,
plus the mesh and the following additional parameters:

-   **`input_file_contents`**: Complete text of the custom GOLEM/MOOSE
    `.i` file.
-   **`input_file_path`** *(optional)*: Path of the original `.i` file.
-   **`mesh_results`** or **`mesh_file`**: The WBGeo mesh, or the path
    of an existing Exodus mesh file (see the table below).
-   **`auxiliary_files_dir`** *(optional)*: Directory containing auxiliary
    files required by a custom GOLEM input, for example, CSV files defining transient injection rate or fracture aperture.
     The directory must already exist. The runner copies its
    complete contents into a `files/` directory in the GOLEM working
    directory. The custom `.i` file should therefore reference these files
    with relative paths such as `files/injection_flowrate.csv` or
    `files/frac_aperture.csv`.
-   **`mesh_file_name`** *(optional)*: Specifies the filename used to stage an existing 
    Exodus mesh in the GOLEM run directory. This supports simulations using meshes 
    created externally or previously exported from the Workbench. The value must be a filename, not a path.
-   **`geomodel_result`** *(optional)*: Structural model results.

The following mesh inputs are supported:

| Builder input | Placeholder in the `.i` file | Behaviour |
| --- | --- | --- |
| `mesh_results=...` | `${GOLEM_MESH_FILE}` | Use the WBGeo mesh. |
| `mesh_file="/path/to/mesh.exo"` | `${GOLEM_MESH_FILE}` | Use an existing mesh file. |

Regardless of the selected mesh source, use `${GOLEM_MESH_FILE}` in the
 `.i` file to reference the mesh:

``` text
[Mesh]
  [original_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
[]
```


Example of a custom builder that requires the auxiliary files `injection_flowrate.csv` and `frac_aperture.csv` to run the custom-defined simulation successfully.


``` python
# input file
reservoir_input_trans_path = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/transient_state.i"
)

with open(reservoir_input_trans_path, "r") as f:
    reservoir_input_trans = f.read()

# Auxiliary CSV files 
reservoir_auxiliary_files_dir = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/files"
)

model_transient = GolemCustomBuilder(
    input_file_contents=reservoir_input_trans,
    input_file_path=reservoir_input_trans_path,
    auxiliary_files_dir=reservoir_auxiliary_files_dir,
    mesh_file=cwd
    + "/examples/case_studies/Groß_Schoenebeck/"
      "GS_steady_state/wbgeo_out.e",
    geomodel_result=structural_model_result,
    print_linear_residuals=True,
)
```

In the input `.i` file, the paths relative to the GOLEM working directory should be defined:

``` text
files/injection_flowrate.csv
files/frac_aperture.csv
```

### 2. Run the simulation

#### Run SfePy

> **Workbench component:** `Run SfePy Simulation`

Both SfePy builders connect to the same runner.

Example call in the codebase:

``` python
data_by_time = run_simulation_sfepy(builder)
```

For an auto-generated hydrothermal problem, the runner performs the
pressure stage followed by the heat stage. For a custom SfePy problem,
execution follows the supplied input by users.

In scripts, `run_simulation_sfepy(builder, stream_output=True)` also
prints SfePy's output live while it runs, as `stream_output` does for
GOLEM. The output is stored in the result either way.

#### Run GOLEM

> **Workbench component:** `Run GOLEM Simulation`

All three GOLEM builders connect to the same runner through the common
`GolemProblem` type.

In the Workbench, the component accepts the following parameter:

-   **`nproc`**: Number of MPI processes. Default `4`.

In the codebase, the `run_simulation_golem()` function executes a GOLEM problem created by one of the supported GOLEM builders. It supports execution either inside a Docker container or directly on the host.

Important runner inputs are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `problem` | — | GOLEM problem created by `GolemHydrothermalProblemBuilder`, `GolemSimpleReservoirBuilder`, or `GolemCustomBuilder`. |
| `image_name` | `None` | Docker image containing GOLEM. If provided, GOLEM runs inside a container; otherwise, it runs on the host. |
| `golem_binary` | `"/golem/golem-opt"` | Path to the GOLEM executable inside the Docker container or on the host. |
| `output_dir` | `<workspace_dir>/golem_output` | Host directory where simulation results and generated files are saved. |
| `nproc` | `4` | Number of MPI processes used to run GOLEM. |
| `temporary` | `True` | If `True`, stages the simulation in a temporary `golem_run_*` subdirectory of `workspace_dir` and removes it afterward. The generated input and mesh files are preserved in `output_dir`. If `False`, stages and runs directly in `workspace_dir` and retains the files for debugging. |
| `workspace_dir` | `"/project"` | Host directory used for staging the simulation files and, for local execution, as the working directory. |
| `stream_output` | `False` | If `True`, prints GOLEM output during execution. The output is also saved to `golem_stdout.txt` after successful execution. |


After a successful simulation, the runner preserves the generated results in `output_dir`.

When `temporary=True`, the generated input and mesh files are also copied into this directory before the temporary working directory is removed.

Example of simulation run in a Docker:

```python
problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    output_dir="/path/to/output",
    nproc=4,
)
```

### Choose where GOLEM runs: Docker or host

The same GOLEM problem builder can be used in both execution modes. Only the arguments passed to `run_simulation_golem()` differ.

The examples below assume that `problem_golem` has already been constructed.

| Environment | Arguments | Required setup |
| --- | --- | --- |
| GOLEM in a Docker container | `image_name=...` and optionally `golem_binary=...` | Docker CLI, running Docker daemon, and a compatible GOLEM image. |
| GOLEM installed on the host | `image_name=None` and `golem_binary=...` | `mpiexec` available on `PATH` and an executable GOLEM binary. |

**Example A — GOLEM in Docker**

```python
from pathlib import Path
from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem

problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    output_dir=str(Path.cwd() / "golem_output"),
    nproc=4,
)
```


**Example B — GOLEM installed on the host**

```python
from pathlib import Path
from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem

problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    golem_binary="/absolute/path/to/golem-opt",
    output_dir=str(Path.cwd() / "golem_native_output"),
    nproc=4,
)
```

### GOLEM installation

To run GOLEM directly on the host, follow the installation guidelines in the [GOLEM GitLab repository](https://git.gfz-potsdam.de/moose/golem).

To obtain access to the GitLab repository, please contact one of the GOLEM maintainers.

After installation, provide the path to the GOLEM executable using the `golem_binary` parameter when calling `run_simulation_golem()`, as shown in **Example B — GOLEM installed on the host** above.

### 3. Export the results

> **Workbench component:** `Export Simulation Results`

Both simulators export the same way. The result is self-contained: it
holds the fields, the solver's CSV tables, the input file(s) and the
console output, so an export needs no output folder on disk.

`export_simulation_results()` returns one ZIP containing:

- the fields, either as one `.vtk` file per saved time step plus a
  `results.pvd` collection for ParaView (`format="vtk"`), or as one
  multi-time-step `results.e` Exodus file with named blocks
  (`format="exodus"`);
- `tables/`: the solver's CSV output (for GOLEM, its postprocessor
  tables);
- `input/`: the input file(s) the solver ran;
- `solver_output/`: the solver's console output per stage.

`format="auto"` (the default) uses Exodus for GOLEM results and VTK
otherwise. Both formats work for both simulators.

Example call in the codebase:

``` python
buf = export_simulation_results(data_by_time, format="vtk")
with open("simulation_results.zip", "wb") as f:
    f.write(buf.getvalue())

buf = export_simulation_results(problem_golem_results, format="exodus")
with open("simulation_golem_results.zip", "wb") as f:
    f.write(buf.getvalue())
```

In scripts, GOLEM's original output files can additionally be kept by
passing `output_dir` to `run_simulation_golem()`:

``` python
problem_golem_results = run_simulation_golem(
    problem=problem_golem,
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    output_dir="/path/to/my_golem_output",
    nproc=4,
)
```

The Workbench's Run GOLEM Simulation component deletes GOLEM's output
folder right after the run, since the result already contains everything.


------------------------------------------------------------------------

## Visualisation and Inspection

Once the execution of a simulation has been done, SfePy and
GOLEM results can use the same result plotting functions. 

For the **GOLEM Simple Reservoir**, the saved ParaView state
file `model1_reservoir_visualisation.pvsm` and `model2_reservoir_visualisation.pvsm`
can be opened in **ParaView** for
more detailed visualisation of the simulation results. The state files
provide a prepared visualisation setup for inspecting the reservoir
simulation output and can be used in addition to the plotting functions
available directly in WBGeo.

### Plot Materials

> **Workbench inspectors:** `Plot Materials` (SfePy problems) and
> `Plot GOLEM Materials` (GOLEM problems)

This inspector provides a visual inspection of material assignments and
properties in the simulation mesh before running the simulation. It helps
verify the spatial distribution of geological units, fault damage zones,
and material properties.

It supports both predefined and custom SfePy and GOLEM problem builders.

The component can display either categorical material assignments or
the spatial distribution of numerical material properties, such as
porosity, permeability, and thermal conductivity.

The relevant inputs in the codebase are:

- **`builder`**: The SfePy or GOLEM problem builder containing the mesh
  and material definitions.
- **`style`** *(default: `"surface"`)*: PyVista style of the material
  meshes (for example `"surface"` or `"wireframe"`), for both simulators.
- **`show_edges`** *(default: `False`)*: Controls whether mesh cell edges
  are displayed.
- **`property_name`** *(optional, default: `None`)*: Specifies a numerical
  material property to visualize, such as `"porosity"`, `"permeability"`,
  or `"k_solid"` (or a custom input file's own coefficient names). Fluid
  properties are shared by all materials and can be given as
  `"fluid.<name>"` or as `"viscosity"`, `"mu"`, `"rho_c_fluid"` and
  similar. If omitted, the component displays categorical material
  assignments.
- **`cmap`** *(default: `"viridis"`)*: Colormap used when visualizing
  numerical material properties.
- **`opacity`** *(default: `0.65`)*: Transparency of the rock materials,
  from `0` (fully transparent) to `1` (fully opaque). Faults, fractures
  and wells are always drawn opaque so they stay visible inside the rock.

Both simulators use the same colors and legend: lithologies get their
structural-model colors (basement grey), faults are black, and the legend
is grouped like the structural model plot. For a custom GOLEM input, each
3D block also shows the lithology it belongs to, when the builder has a
structural model.

Example calls in the codebase:

```python
# Visualize material assignments
plot_builder_materials(builder)

# Visualize permeability distribution
plot_builder_materials(
    builder,
    property_name="permeability",
    cmap="viridis",
    opacity=0.7,
    show_edges=False,
)
```

**Note:** The material-property visualization uses numerical values
defined in the model before simulation. It does not display computed
simulation results such as temperature or pressure. Available properties
depend on the selected problem builder and its material definitions.



### Plot a variable at a time step

> **Workbench inspector:** `Plot Variable (Time Step)`

This inspector provides a 3D visualization of a simulation result
variable at a selected time step. It displays the spatial distribution
of variables such as temperature or pressure on the simulation mesh.

By default, the final saved time step is displayed. The variable
defaults to the temperature if available, or the first available
numerical scalar field.

Results keep the variable names their solver writes: SfePy writes `T`
and `p`, GOLEM writes `temperature` and `pore_pressure`, and a custom
builder writes whatever its input file defines. For the two hydrothermal
variables, all plotting functions accept either name, so `var_name="T"`
also works for a GOLEM result and `var_name="temperature"` for a SfePy
one.

The relevant parameters in the codebase are:

- **`sim`**: The `SimulationResults` object containing the simulation
  results.
- **`var_name`** *(optional, default: `None`)*: Name of the variable
  to visualize.
- **`time`** *(optional, default: `None`)*: Time step to display.
  If omitted, the final saved time step is used.
- **`cmap`** *(default: `"viridis"`)*: Colormap used for the variable.
- **`show_edges`** *(default: `False`)*: Controls whether mesh cell
  edges are displayed.
- **`edge_color`** *(default: `"grey"`)*: Color of the mesh edges.
- **`edge_line_width`** *(default: `0.5`)*: Width of the mesh edges.
- **`show_contours`** *(default: `False`)*: Displays contour lines
  of the selected variable on the 3D surface.
- **`n_contours`** *(default: `10`)*: Number of contour levels.
- **`contour_color`** *(default: `"black"`)*: Color of contour lines.
- **`scale`** *(default: `(1, 1, 1)`)*: Scaling factors for the
  X, Y, and Z directions.
- **`off_screen`** *(default: `False`)*: Enables off-screen rendering.
- **`window_size`** *(optional, default: `None`)*: Size of the
  visualization window.
- **`anti_aliasing`** *(default: `True`)*: Controls anti-aliasing
  during rendering.

Example call in the codebase:

```python
plot_variable_at_a_time(
    simulation_result,
    var_name="T",
    time=final_time,
    cmap="coolwarm",
    show_edges=True,
    show_contours=False,
)
```

### Plot a cross-section

> **Workbench inspector:** `Plot Cross Section 2D`

This inspector visualizes a simulation result variable on a specified
2D cross-section. It displays the variable at two saved time steps
side by side, allowing their spatial distributions to be compared.

By default, the first and last saved time steps are used. Both plots
share the same color scale for direct comparison.

The relevant parameters in the codebase are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`** *(optional, default: `None`)*: Variable to visualize.
  Defaults to `"T"` if available, or the first available scalar field.
- **`time_a`** *(optional, default: `None`)*: First time step.
  Defaults to the first saved time step.
- **`time_b`** *(optional, default: `None`)*: Second time step.
  Defaults to the last saved time step.
- **`origin`** *(default: `(0, 0, 0)`)*: A point defining the
  cross-section plane.
- **`normal`** *(default: `(0, 1, 0)`)*: Normal vector defining
  the orientation of the cross-section plane.
- **`cmap`** *(default: `"coolwarm"`)*: Colormap used for the variable.
- **`n_contours`** *(default: `15`)*: Number of contour levels.
- **`resolution`** *(default: `200`)*: Resolution of the interpolated
  2D grid used for plotting.

Example call in the codebase:

```python
plot_cross_section_2D(
    simulation_result,
    var_name="T",
    time_a=initial_time,
    time_b=final_time,
    origin=(500, 500, 500),
    normal=(0, 1, 0),
    cmap="coolwarm",
)
```
#### Plot a variable along a line
> **Workbench inspector:** `Plot Variable Along Line`

`plot_variable_along_line()` samples a result variable along a
straight line between two specified points at one time step and
plots its values against distance along the line.

The relevant parameters are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`**: Variable to sample.
- **`time`**: Time step used for sampling.
- **`p0`**: Starting point of the line `(x, y, z)`.
- **`p1`**: Ending point of the line `(x, y, z)`.
- **`n_samples`** *(default: `200`)*: Number of sampling intervals
  along the line.

Example call in the codebase:

```python
plot_variable_along_line(
    simulation_result,
    var_name="T",
    time=final_time,
    p0=(0, 500, 500),
    p1=(1000, 500, 500),
    n_samples=200,
)
```

#### Plot a time series at a point
> **Workbench inspector:** `Plot Variable Time Series`

`plot_variable_time_series()` samples a result variable at the same
spatial point across all saved simulation time steps and plots the
resulting time series.

The relevant parameters are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`**: Variable to sample.
- **`point`**: Spatial coordinates `(x, y, z)`.
- **`decimals`** *(default: `3`)*: Number of decimal places used
  for formatting the vertical axis.

Example call in the codebase:

```python
plot_variable_time_series(
    simulation_result,
    var_name="T",
    point=(500, 500, 500),
    decimals=3,
)
```

If the point cannot be sampled at a particular time step, the
corresponding value is represented as a gap in the time-series plot.



### View solver output

> **Workbench inspector:** `View Solver Output`

This inspector shows the text output of the solver for a simulation
result, for both SfePy and GOLEM runs. It is useful for checking solver
progress, convergence information, warnings, and errors, also after a
successful run.

Example access in the codebase:

```python
# One transcript per stage: "pressure"/"heat" or "custom" for SfePy, "golem" for GOLEM
for stage, output in simulation_result.solver_output.items():
    print(f"=== {stage} ===")
    print(output)
```



### Additional simulation result plots and inspection

The following post-processing functions are available in the codebase
and can be used to inspect `SimulationResults`. They support both
SfePy and GOLEM results when the requested variables and required
data are available.

#### Plot the difference between two time steps

`plot_variable_difference()` displays the change in a variable
between two saved time steps in 3D, calculated as `time_b - time_a`.

The color scale is centered symmetrically around zero to distinguish
increases and decreases.

The relevant parameters are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`**: Name of the point-data variable.
- **`time_a`** *(optional, default: `None`)*: First time step.
  Defaults to the first saved time step.
- **`time_b`** *(optional, default: `None`)*: Second time step.
  Defaults to the last saved time step.
- **`cmap`** *(default: `"RdBu_r"`)*: Colormap for the difference.
- **`show_edges`** *(default: `False`)*: Controls mesh edge visibility.
- **`scale`** *(default: `(1, 1, 1)`)*: Scaling factors for the
  X, Y, and Z directions.

Example call in the codebase:

```python
plot_variable_difference(
    simulation_result,
    var_name="T",
    time_a=initial_time,
    time_b=final_time,
    cmap="RdBu_r",
    show_edges=False,
)
```

#### Plot a cross-section difference

`plot_cross_section_difference_2D()` visualizes the change in a
point-data variable between two time steps on a specified 2D plane.
The difference is calculated as `time_b - time_a`.

The relevant parameters are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`**: Name of the point-data variable.
- **`time_a`** *(optional, default: `None`)*: First time step.
- **`time_b`** *(optional, default: `None`)*: Second time step.
- **`origin`** *(default: `(0, 0, 0)`)*: Point on the section plane.
- **`normal`** *(default: `(1, 0, 0)`)*: Normal vector of the plane.
- **`cmap`** *(default: `"RdBu_r"`)*: Difference colormap.
- **`n_contours`** *(default: `15`)*: Number of contour levels.
- **`resolution`** *(default: `200`)*: Resolution of the
  interpolated 2D grid.

Example call in the codebase:

```python
plot_cross_section_difference_2D(
    simulation_result,
    var_name="T",
    time_a=initial_time,
    time_b=final_time,
    origin=(500, 500, 500),
    normal=(0, 1, 0),
    cmap="RdBu_r",
)
```


#### Print a variable at a point

`print_variable_at_point()` samples a result variable at a specified
spatial point and time step. 

The relevant parameters are:

- **`sim`**: The `SimulationResults` object.
- **`var_name`**: Variable to sample.
- **`time`**: Time step used for sampling.
- **`point`**: Spatial coordinates `(x, y, z)`.

Example call in the codebase:

```python
value = print_variable_at_point(
    simulation_result,
    var_name="T",
    time=final_time,
    point=(500, 500, 500),
)

print(value)
```


------------------------------------------------------------------------

## Output structure

Both solver pipelines converge to the same WBGeo result type:

``` text
SfePy:
Build SfePy Hydrothermal Problem ─┐
                                  ├─> Run SfePy Simulation ─┐
Build SfePy Custom Problem ───────┘                         │
                                                            ├─> SimulationResults
GOLEM:                                                      │        │
Build GOLEM Hydrothermal Problem ─┐                         │        ├─> Visualisation / Inspection
Build GOLEM Simple Reservoir ─────┼─> Run GOLEM Simulation ─┘        └─> Export Simulation Results
Build GOLEM Custom Problem ───────┘
```

`SimulationResults` is a time-dependent container for geometry,
topology, and numerical fields. It includes:

-   **`nodes_by_time`** --- nodal coordinates for each saved time step.
-   **`cells_by_time`** --- cell connectivity.
-   **`celltypes_by_time`** --- VTK-style cell type identifiers.
-   **`node_data_by_time`** --- node-based result variables.
-   **`cell_data_by_time`** --- cell-based result variables.
-   **`solver`** --- `"sfepy"` or `"golem"`.
-   **`solver_output`** --- the solver's console output per stage.
-   **`solver_tables`** --- tabular solver output per file and column,
    e.g. GOLEM's postprocessor CSVs.
-   **`input_files`** --- the input file(s) the solver ran.
-   **`block_names`** --- the name of each element block, by block id.
