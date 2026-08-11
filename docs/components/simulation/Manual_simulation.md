# Simulation

The simulation component connects the Workbench to numerical solvers. Since the meshing component exports to multiple standard formats, the Workbench can interface with a range of commercial and open-source simulation packages. In the current version, **SfePy** (Simple Finite Elements in Python) is integrated as an open-source, Python-based multi-physics finite element framework. Meshes generated within the Workbench are passed directly to SfePy at runtime, without manual file management.

> **Note:** The current version supports single-phase Darcy flow coupled with advective-conductive heat transport (a simplified hydrothermal model), plus arbitrary physics via a user-supplied SfePy input file. Faults are represented only as a "damage zone" of reassigned material properties, not as explicit geometric discontinuities, and engineering objects (wells, sources, shafts) are not solved for physically even where the mesh contains them.

---

## Concepts

Before a model can be solved, it needs a **problem definition** — the mesh plus everything SfePy needs to actually run: material properties, boundary conditions, and solver settings. The Workbench offers two ways to build one:

**Hydrothermal Problem** — auto-generated from the mesh and its structural model. Material regions and boundary regions are inferred directly from the lithology mapping and the mesh's top/bottom point sets; the physics is a simplified, single-phase Darcy flow coupled to advective-conductive heat transport. No SfePy scripting required — only physical properties per rock unit and a handful of solver settings.

**Custom SfePy Problem** — wraps a complete, hand-written SfePy input file with its own physics, staging, and sequencing logic. Not restricted to hydrothermal physics — any SfePy problem can be brought in this way. It is validated against the mesh (known mesh type, known lithology mapping, and that every region the file references actually exists), but the physics itself — material properties, equations, solver configuration — is entirely up to the file and is not checked; SfePy's own execution is the final authority.

Both produce the same problem-definition type and connect to the same `Run Simulation` component, so a workflow can be built once and switched between an auto-generated and a custom problem without rewiring anything downstream.

**Materials.** Each lithology in the structural model becomes its own material region, with its own porosity, permeability, thermal conductivity, and volumetric heat capacity. A single set of fluid properties (viscosity, thermal conductivity, heat capacity) is shared across the whole model — a single-phase assumption. Reasonable defaults (water at roughly 20°C, typical sedimentary rock properties) are used for anything not specified explicitly.

**Fault representation.** SfePy has no dedicated fault or interface element, and only unstructured meshing explicitly meshes fault surfaces — implicit and structured meshes carry no fault geometry at all, only a per-cell lithology id. Faults are therefore represented as a **damage zone**: a band of mesh cells within a chosen distance of each fault's trace is reassigned its own material, distinct from the surrounding lithology. This is a standard, simplified proxy for fault hydraulics (a sealing gouge zone or a fracture-enhanced conduit, depending on the properties assigned) rather than an exact geometric fault. All faults in a model share one combined damage-zone region — faults are not modeled individually.

**Two-stage solve (Hydrothermal Problem only).** Pressure and temperature are solved as two sequential, single-physics stages rather than one coupled system: first a steady-state Darcy pressure solve, then a transient heat solve that uses the Darcy velocity computed from the pressure solution as an advection term. Temperature therefore depends on the flow field, but the flow field never depends on temperature (no buoyancy-driven convection). Segregating the two stages also keeps each one numerically well-conditioned — combining pressure (order 10⁶ Pa) and temperature (tens of °C) into a single linear system is not. Set `include_flow=False` to skip the pressure stage entirely and solve pure heat conduction instead.

---

## Workflow

### 1. Build the problem definition

Two alternative components build a problem definition — pick the one that fits your model. Both connect to the same `Run Simulation` component in step 2.

#### Auto-generated: Build Hydrothermal Problem

> **Workbench component:** `Build Hydrothermal Problem`

Requires a mesh with a known origin (produced by one of the meshing components) and a per-cell lithology mapping — always present for implicit and structured meshes, and present for unstructured meshes only when they were generated with automatic lithology mapping.

- **`mesh_results`**: The mesh to solve on (implicit, structured, or unstructured).
- **`geomodel_result`**: The structural model `mesh_results` was generated from — supplies rock unit names, fault geometry, and boundary point sets.
- **`options`** *(optional)*: Per-rock-unit properties (porosity, permeability, thermal conductivity, volumetric heat capacity), global fluid properties, and fault-zone properties. In the visual interface this is a form generated per rock unit once a structural model is connected; in the codebase it is a plain dictionary of `RockUnitProperties`. Any rock unit not given explicit properties falls back to the defaults.
- **`include_flow`** *(optional)*: If `False`, skips the pressure stage and solves pure heat conduction only. Default `True`.
- **`enable_fault_zone`** *(optional)*: If `True`, gives cells near any fault their own material (the fault-zone properties from `options`) instead of treating them as ordinary lithology. Has no effect if the structural model has no faults. Default `False`.
- **`fault_zone_n_voxels`** *(optional)*: Width of the fault damage zone, in structural-grid voxels. Default `1`.
- **`t0`**, **`t1`**, **`num_steps`** *(optional)*: Start time, end time, and number of steps for the transient heat stage. Default `t0=0.0`, `num_steps=1`; `t1` defaults to a data-driven estimate of the model's thermal diffusion timescale if not given.
- **`p_top`**, **`p_bottom`** *(optional)*: Pressure boundary values at the model top/bottom, in Pa. Default `p_top=0.0`, `p_bottom=1e6`.
- **`t_top`**, **`t_bottom`** *(optional)*: Temperature boundary values at the model top/bottom, in °C. Default `t_top=10.0`, `t_bottom=60.0`.
- **`linear_solver`** *(optional)*: `"iterative"` (default) scales to large meshes; `"direct"` is exact but does not scale well — use it for small or debug meshes.
- **`linear_solver_i_max`**, **`linear_solver_eps_r`** *(optional)*: Maximum iterations and relative residual tolerance for the iterative linear solver. Default `500` and `1e-8`.

Example call in the codebase:

```python
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

Here, `rock_properties` gives each lithology its own porosity, permeability, thermal conductivity, and heat capacity; `fluid` describes the pore fluid. `t1=5e11` overrides the default diffusion-timescale estimate for the heat stage's end time, solved over `num_steps=3` steps.

A faulted model additionally enables the damage zone:

```python
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

This gives the fault a low porosity/permeability — a sealing fault. Raising `permeability` above the surrounding rock instead represents a fracture-enhanced conduit.

**Limitations:**

- **Single-phase fluid.** One set of fluid properties is shared across the whole model — no multiphase flow.
- **Segregated, one-way coupling.** Temperature depends on the flow field, but the flow field never depends on temperature — there is no buoyancy-driven (free) convection.
- **Structured meshes do not support faulted models.** Use an implicit or unstructured mesh for a faulted structural model instead (see the meshing manual).
- **No engineering objects.** Wells, point sources, and mining shafts can be embedded in an unstructured mesh for geometric refinement (see the meshing manual), but their mesh cells are excluded from the solved domain entirely — there is no well injection/production term and no point-source term.
- **Fault zones are a single combined region.** All faults in a model share one damage-zone material, not modeled individually, and the damage zone is a fixed-width band around each fault's trace rather than an exact geometric split.

#### User-supplied: Build Custom SfePy Problem

> **Workbench component:** `Build Custom SfePy Problem`

Brings a complete, hand-written SfePy input file into the Workbench instead of letting the physics be auto-generated.

- **`input_file`**: Path to a complete SfePy input file (`.py`). In the visual interface, the file is picked from a folder of user-provided files made available to the Workbench.
- **`mesh_results`**: The mesh to solve on. Meshes containing wells or sources are rejected — not supported by this component.
- **`geomodel_result`** *(optional)*: The structural model, used for an additional sanity check (comparing the number of referenced regions against the number of lithologies) and required if `fault_zone_n_voxels` is set.
- **`fault_zone_n_voxels`** *(optional)*: If set, gives cells within this many structural-grid voxels of any active fault's trace their own region id — the same damage-zone cell selection `Build Hydrothermal Problem` uses, referenceable from the custom file as `cells of group N`. No material or equation is generated for it automatically; it is entirely up to the custom file what to do with these cells.

The mesh and region bookkeeping are validated at build time — known mesh type, known lithology mapping, and every `cells of group N` selector the file references actually existing on the mesh — but this cannot catch a wrong material property or a malformed equation; SfePy's own solve remains the final check.

Two worked examples, covering both a from-scratch derivation and a bit-for-bit reproduction of the equivalent Hydrothermal Problem, are provided for each of the synthetic example models:

- `examples/synthetic_examples/model1/input_data/simulation_files/custom_hydrothermal_fromscratch_implicit.py` and `custom_hydrothermal_reproduction_implicit.py` — for model1's unfaulted, implicit mesh.
- `examples/synthetic_examples/model2/input_data/simulation_files/custom_hydrothermal_fromscratch_faulted_unstructured.py` and `custom_hydrothermal_reproduction_faulted_unstructured.py` — for model2's faulted, unstructured mesh, including its fault damage zone.

The "from-scratch" files build the same physics from first principles, without relying on any internals of `Build Hydrothermal Problem`; the "reproduction" files instead reconstruct its generated input bit-for-bit. Both are a practical starting point for writing a custom problem against a new mesh.

Example call in the codebase:

```python
with open("examples/synthetic_examples/model1/input_data/simulation_files/custom_hydrothermal_fromscratch_implicit.py") as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
)
```

**Limitations:**

- **No engineering objects.** A mesh containing wells or sources is rejected outright at build time — not supported yet.
- **Structural validation only.** Only the mesh and referenced regions are checked; material properties and equations are not — a malformed custom file is only caught when SfePy actually runs it.
- **Fault zones are a single combined region**, the same limitation `Build Hydrothermal Problem` has (see above) — if `fault_zone_n_voxels` is used.

### 2. Run the simulation

> **Workbench component:** `Run Simulation`

Runs the problem definition from either build step above. The two-stage pressure/heat solve is used for a Hydrothermal Problem; a single SfePy run is used for a Custom SfePy Problem — the correct one is selected automatically based on which build component produced the input.

Example call in the codebase:

```python
data_by_time = run_simulation_sfepy(builder)
```

### 3. Export the results

> **Workbench component:** `Export Simulation Results`

Exports every solved time step to a downloadable, zipped VTK time series — one legacy `.vtk` file per time step plus a ParaView `.pvd` collection file tying them together with their actual time values, so the result opens directly in ParaView with full time-stepping.

> **Note:** VTK is currently the only export format implemented for simulation results.

Example call in the codebase:

```python
buf = export_simulation_results(data_by_time, format="vtk")
with open("simulation_results.zip", "wb") as f:
    f.write(buf.getvalue())
```

---

## Visualisation and Inspection

> **Workbench:** Where available, these are exposed as inspector functions with limited functionality in the visual interface (visual DSL) — parameters are fixed; the codebase functions below expose the full set of options.

### Plot Materials

> **Workbench component:** `Plot Materials`

A pre-flight check, run on the problem definition before solving: plots the mesh colored one solid color per material (lithologies, plus the fault damage zone if active), with a legend of the properties each material actually uses. Catches a wrong material or fault-zone assignment before running a solve that can take a while. Works for both a Hydrothermal Problem (legend entries summarize porosity/permeability/conductivity/heat capacity) and a Custom SfePy Problem (legend entries are bare material names, since a custom file has no structured properties object to summarize).

- **`builder`**: A `HydrothermalProblemBuilder` or `CustomSfepyBuilder` (not yet run).
- **`style`** *(optional)*: PyVista mesh style — `"surface"`, `"wireframe"`, or `"points"`. Default `"surface"`.
- **`show_edges`** *(optional)*: If `True`, displays mesh element boundaries. Default `True`.

```python
plot_builder_materials(builder)
```

### Plot a variable at a time step

> **Workbench component:** `Plot Variable (Final Time)`

Produces a 3D visualisation of a selected simulation variable at a specific time step.

- **`sim`**: The `SimulationResults` object.
- **`var_name`** *(optional)*: Variable to visualise, e.g. `"T"` or `"p"`. Defaults to `"T"` if present, otherwise whichever field the result actually has — a Custom SfePy Problem is not restricted to hydrothermal variable names.
- **`time`** *(optional)*: Time step at which to plot. Defaults to the final saved step.
- **`cmap`** *(optional)*: Colormap, default `"viridis"`.
- **`show_edges`** *(optional)*: If `True`, displays mesh element boundaries. Default `False`.
- **`show_contours`** *(optional)*: If `True`, overlays isolines of `var_name` on the 3D surface. Default `False`.
- **`scale`** *(optional)*: Tuple `(sx, sy, sz)` to scale the geometry along each axis. Default `(1, 1, 1)`.

```python
plot_variable_at_a_time(data_by_time, "T", time=0, cmap="coolwarm", show_edges=True)
```

### Plot a cross-section

> **Workbench component:** `Plot Cross Section 2D`

Slices a variable at two time steps (initial and final by default) on the same plane and plots them side by side, sharing one color scale for direct visual comparison.

- **`sim`**: The `SimulationResults` object.
- **`var_name`** *(optional)*: Variable to visualise, same default as above.
- **`time_a`**, **`time_b`** *(optional)*: The two time steps to compare. Default to the first and last saved steps.
- **`origin`** *(optional)*: A 3D point the slicing plane passes through. Default `(0, 0, 0)`.
- **`normal`** *(optional)*: A 3D vector defining the orientation of the slicing plane. Default `(0, 1, 0)`.
- **`cmap`** *(optional)*: Colormap, default `"coolwarm"`.

```python
plot_cross_section_2D(data_by_time, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")
```

### View solver output

> **Workbench component:** `View Solver Output`

Prints the raw SfePy solver output for a result, per stage (`"pressure"`/`"heat"` for a Hydrothermal Problem, `"custom"` for a Custom SfePy Problem) — lets solver convergence and quality be checked on a normal, successful run too, not only via the error raised when a solve fails to converge.

```python
for stage, stdout in data_by_time.sfepy_stdout.items():
    print(f"=== {stage} ===")
    print(stdout)
```

### Additional codebase-only plots

The following are not currently exposed as Workbench inspectors, but are available for further analysis in the codebase.

**Plot the change between two time steps** — `plot_variable_difference(sim, var_name, time_a=None, time_b=None, cmap="RdBu_r", show_edges=False, scale=(1, 1, 1))`: a 3D plot of `value(time_b) - value(time_a)` using a diverging colormap centered on zero, isolating what actually changed rather than relying on eyeballing two separate absolute snapshots.

**Plot a cross-section of the change between two time steps** — `plot_cross_section_difference_2D(sim, var_name, time_a=None, time_b=None, origin=(0, 0, 0), normal=(1, 0, 0), cmap="RdBu_r", n_contours=15, resolution=200)`: the 2D cross-section analogue of the above.

**Plot a variable along a line** — `plot_variable_along_line(sim, var_name, time, p0, p1, n_samples=200)`: a 2D profile of a variable versus distance along a straight line between two points, for quantitative comparison along a transect.

```python
plot_variable_along_line(data_by_time, "p", 0, p0=(500, 20, 0), p1=(500, 20, 1000), n_samples=20)
```

**Print a variable at a point** — `print_variable_at_point(sim, var_name, time, point)`: samples and logs the value of a variable at a specific spatial point and time step.

**Plot a time series at a point** — `plot_variable_time_series(sim, var_name, point, decimals=3)`: tracks the temporal evolution of a variable at a fixed spatial location across every saved time step.

```python
plot_variable_time_series(data_by_time, "p", point=(500, 20, 500))
```

---

## Output

> **Workbench:** The `problem` output port of `Build Hydrothermal Problem` or `Build Custom SfePy Problem` connects directly to `Run Simulation`, and can be inspected in-place using `Plot Materials`. The `simulation_result` output port of `Run Simulation` carries the solved result — it can be connected to `Export Simulation Results`, or inspected in-place using `Plot Variable (Final Time)`, `Plot Cross Section 2D`, and `View Solver Output`.

`run_simulation_sfepy()` returns a `SimulationResults` object — a time-dependent container for geometry, topology, and result fields:

- **`nodes_by_time`** — nodal coordinates for each solved time step, shaped `(n_nodes, 3)`.
- **`cells_by_time`** — cell connectivity for each solved time step.
- **`celltypes_by_time`** — VTK-style cell type identifiers for each element.
- **`node_data_by_time`** — node-based variables (e.g. `"T"`, `"p"`), as `{variable_name: array}` dictionaries per time step.
- **`cell_data_by_time`** — element-based variables, in the same structure as `node_data_by_time`.
- **`sfepy_stdout`** — raw solver output, keyed by stage (`"pressure"`/`"heat"`, or `"custom"`).

Time steps are keyed by their saved step index (`0`, `1`, `2`, ...), not by physical simulation time — one file is saved per actually-solved step, so the fully evolved final state is always at `max(sim.nodes_by_time.keys())`.

`SimulationResults` is the `sim` object used throughout Visualisation and Inspection above, and the direct input to `Export Simulation Results`.
