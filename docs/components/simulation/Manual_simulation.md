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

**Sfepy Hydrothermal Problem** --- auto-generated from the mesh and structural
model. Material regions and boundary regions are inferred from the
lithology mapping and mesh information. The built-in formulation solves
simplified single-phase Darcy flow followed by advective-conductive heat
transport.

**Sfepy Custom SfePy Problem** --- wraps a complete user-supplied SfePy `.py`
input file. The custom file specifies a user-defined problem with its
variables, materials, equations and solvers. 


### GOLEM problem definitions

**GOLEM Hydrothermal Problem** --- auto-generates a GOLEM
hydrothermal problem from a WBGeo mesh, structural model, rock
properties, fluid properties, boundary conditions, and optional fault
properties.

**GOLEM Simple Reservoir Problem** --- builds a dedicated reservoir
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
>   temporary Docker container created from the specified image. This is valid
>   for the codebase. The required
>   input files are staged in the container, and the simulation results are
>   copied back to the host. The container is removed after execution.
> - *Workbench execution:** GOLEM runs directly on the host. 
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

> **Workbench component:** `Build Sfepy Hydrothermal Problem`

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
-   Engineering objects such as wells and point sources cannot be
    physically solved by this built-in workflow.

### SfePy: User-supplied Problem

> **Workbench component:** `Build Custom SfePy Problem`

This component brings a complete SfePy `.py` input into the Workbench.
In the Workbench, this component requires connections to both a mesh and
the structural model used to generate that mesh.

The component accepts the following parameters:
-   **`input_file`**: Path to a complete SfePy input file (`.py`).
-   **`fault_zone_n_voxels`** *(optional)*: If set, gives cells within
    this many structural-grid voxels of any active fault's trace their
    own region id --- the same damage-zone cell selection
    `Build Hydrothermal Problem` uses, referenceable from the custom
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
-   **`start_time`**, **`end_time`**, **`dt`**, **`num_steps`**: Time
    controls.
-   **`output_interval`**: Result-output frequency.
-   **`print_linear_residuals`**: Enables linear residual reporting.

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


### GOLEM: Simple Reservoir Problem

> **Workbench component:** `Build GOLEM Simple Reservoir Problem`

The simple-reservoir builder supports two synthetic fractures, a well, an injection source,
geological faults already present in the structural model, local mesh
refinement, material/fluid properties, transient controls, and low-level
PETSc solver settings.


In the Workbench, this component requires connections to a
structural model. The unstructured mesh is created in the builder. 

The component accepts the following parameters in the Workbench:
-   **`mesh_size`** *(optional)*: Target volume-mesh size. Default
    `30.0`.
-   **`curve_mesh_size`** *(optional)*: Target mesh size on well curves.
    Default `5.0`.
-   **`z_threshold`** *(optional)*: Optional z-direction threshold used
    by the unstructured meshing workflow. Default `None`.
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

------------------------------------------------------------------------

## Output structure

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
