# Process Simulation — implementation reference

Hydrothermal process simulation pipeline: single-phase Darcy flow + advective-
conductive heat transport, solved with SfePy on a mesh produced by
`core/meshing_components/`. Two ways to define the SfePy problem, one shared
run/visualization/export path for both.

---

# 1. Architecture

- **`simulation_packages/sfepy/sfepy_hydrothermal_builder.py`**
  - `HydrothermalProblemBuilder` — takes a `MeshResults`, `StructuralModelResults`,
    and physical properties (`RockUnitProperties` per lithology, `FluidProperties`)
    and auto-generates the SfePy problem-description `.py` files (regions,
    materials, equations, solvers).
  - `CustomSfepyBuilder` — wraps a complete, user-supplied SfePy input file
    instead of auto-generating one (see §2). Validated against the mesh at
    construction time (known mesh type, known lithology mapping, no
    wells/sources, every `'cells of group N'` referenced actually exists).
  - `SfepyProblem` — thin envelope (`Optional[HydrothermalProblemBuilder]` +
    `Optional[CustomSfepyBuilder]`, exactly one set) so both builders' outputs
    register as one Workbench type and share downstream components.
  - Shared helpers used by both builder types: `map_mat_id_to_lithology`,
    `enumerate_rock_units`, `lithology_to_group_idx`,
    `compute_fault_zone_cell_mask`, `check_mesh_has_known_type`,
    `check_mesh_has_lithology_mapping`, `check_mesh_has_no_engineering_objects`,
    `check_custom_sfepy_regions`.
- **`simulation_packages/sfepy/sfepy_hydrothermal_run.py`** — `run_simulation_sfepy(problem)`
  dispatches on builder type to `_run_hydrothermal_problem` (two-stage) or
  `_run_custom_sfepy_problem` (one `sfepy-run` call), both returning the same
  `SimulationResults`. Converts the mesh to SfePy's expected format (Exodus →
  Medit via meshio) and runs `sfepy-run` as a subprocess. Also
  `export_simulation_results()` (VTK only).
- **`simulation_visualization/simulation_visualization.py`** — plotting for
  `SimulationResults` (3D snapshot, cross-section, before/after diff, point/line
  sampling) and `plot_builder_materials(builder)`, a pre-flight material/fault-
  zone check before running a solve. Works for either builder type.
- **`simulation_workbench_components.py`** — `@wbgeo_component` wrappers (§4).
- **`output_format/vtk/unified_format_vtk.py`** — `load_vtk_results()`, reads
  SfePy's per-timestep VTK output into a `SimulationResults`.

# 2. Two problem-definition paths

**`HydrothermalProblemBuilder`** auto-generates a coupled Darcy-flow +
heat-transport problem: steady-state pressure solved first, then transient
temperature (advection from the pressure stage's Darcy velocity + conduction),
as two sequential single-physics SfePy runs — segregated because temperature
depends on the flow field but pressure never depends on temperature, and mixing
the two in one system is badly conditioned. `include_flow=False` skips the
pressure stage for pure-conduction cases. `t1` defaults to a data-driven
diffusion timescale (`L²/α`) rather than an arbitrary constant. Linear solver is
`scipy_direct` or `scipy_iterative` (BiCGStab + ILU) — no PETSc/MPI dependency
anywhere in this pipeline.

**`CustomSfepyBuilder`** takes a complete, hand-written SfePy input file instead —
**this is the same pattern the original (pre-rewrite) implementation used**
(hand-edited static `.py` input files), now formalized as a proper, validated
Workbench component rather than the only available option. Not restricted to
hydrothermal physics; only mesh/region bookkeeping is validated, not the
equations. Gets the identical opt-in fault damage-zone mechanism as
`HydrothermalProblemBuilder` (`fault_zone_n_voxels`).

Both builders flow through the same `run_simulation_sfepy()`, same
`SimulationResults`, same plotting/export functions.

**Verified equivalence**: for both example models, a hand-written
`CustomSfepyBuilder` file reproducing `HydrothermalProblemBuilder`'s own
generated sequence matches it to `0.0` max difference, node-for-node (model1:
implicit mesh, `include_flow=True`; model2: unstructured mesh, real fault,
64,688-cell damage zone). Both files are committed as worked examples
(`examples/synthetic_examples/model{1,2}/input_data/simulation_files/`), wired
into `WBGeo1.0_model{1,2}.py` right after each script's own
`HydrothermalProblemBuilder` call.

# 3. Fault support

`fault_zone_properties`/`fault_zone_n_voxels` gives cells near a fault their own
material — a damage-zone band, computed on the structural grid via each fault's
`get_domain_mask()` and resampled onto mesh cells. Respects `fault_activity`
(a fault restricted to specific stratigraphic groups only gets a zone there).
Identical mechanism for both builder types (`compute_fault_zone_cell_mask`); a
custom file references the resulting cells via `'cells of group {fault_group_id}'`.
See §6 for why this is the only fault representation available.

# 4. Meshing integration

Both builders resolve lithology via `MeshResults.cell_data["block_id"]`
(`map_mat_id_to_lithology`), trusted directly from meshing for all three mesh
types (implicit, structured, unstructured) — not re-derived. `MeshResults` now
carries its own `mesh_type` (`MeshType` enum), set automatically by all three
mesh-creation functions. Structured meshing rejects a faulted structural model
outright (`pre_check_no_faults_for_structured_mesh`) — structured+fault does not
reliably converge when solved.

# 5. Workbench components

- **Build Hydrothermal Problem** (`build_hydrothermal_problem`) → `SfepyProblem`
- **Build Custom SfePy Problem** (`build_custom_sfepy_problem`) → `SfepyProblem` —
  `input_file` is a `RemoteFile`-controlled path (place under `own_data/`, pick
  via the file browser), not a direct upload.
- **Run Simulation** (`run_simulation`) — one component for either builder,
  `SfepyProblem -> SimulationResults`.
- **Export Simulation Results** (`export_simulation_results`) — VTK.
- **Plot Materials** (pre-flight, either builder), **Plot Variable (Final Time)**,
  **Plot Cross Section 2D** — inspectors on `SfepyProblem`/`SimulationResults`.
- `HydrothermalProblemBuilder`'s per-rock-unit/fluid/fault-zone properties are
  editable via a SmartInput sidebar (`options`); `CustomSfepyBuilder` has no
  equivalent since a custom file's own `materials` block already carries
  whatever properties its author wrote.

# 6. Limitations

## 6.1 Custom builder: no shared multi-stage orchestration

`run_simulation_sfepy()` calls either `_run_hydrothermal_problem` (built-in
two-stage pressure→heat) or `_run_custom_sfepy_problem` (a single `sfepy-run`
invocation, no stage handoff) — there is no shared staging mechanism between
them. **A user wanting multi-stage physics (or any sequencing) with
`CustomSfepyBuilder` must implement it themselves, entirely inside their own
input file.** This works because `sfepy-run` genuinely imports the input file
as a Python module, executing every top-level statement (including, e.g.,
solving an earlier stage in-process via SfePy's own library API) before
reading `regions`/`materials`/`equations` — a real, documented mechanism (both
worked examples in §2 use it), not a workaround. There is no lower-effort path
today than writing that sequencing by hand each time.

## 6.2 Faults: 3D damage zone only, not a real fault surface

Neither builder can represent a fault as a true 2D interface with its own jump
condition. SfePy's interface-jump term (`dw_jump`) requires a **non-conforming**
mesh — independent, duplicated DOFs on each side of the fault plane. This
pipeline's unstructured meshing (`create_unstructured_mesh_data`, via
`gmsh.model.occ.fragment`) produces a **conforming**, watertight mesh by
construction — confirmed empirically on a real faulted mesh: zero near-duplicate
node coordinates, 100% of fault-surface nodes shared by tetrahedra on both
sides. **Not fixable from the simulation side** — it would require a materially
different meshing approach (a non-conforming mesh with duplicated interface
DOFs), which is a meshing-architecture change, not a simulation one. Until/unless
that changes, both builders are limited to one shared damage-zone-band material
per model: no per-fault properties, no exact hanging-wall/footwall split.

## 6.3 Other, minor

- `export_simulation_results()` only supports `format="vtk"` (no Exodus).
- The convergence check (`_check_convergence`) only catches SfePy self-reporting
  non-convergence (`cond` 1/2) — it cannot catch a numerically-converged but
  physically-wrong solve.
- Structured mesh + fault is blocked at mesh-generation time (§4); the
  underlying non-convergence issue itself isn't root-caused.
- `CustomSfepyBuilder` rejects any mesh with a well/source block outright
  (`check_mesh_has_no_engineering_objects`) — stricter than
  `HydrothermalProblemBuilder`, which silently leaves such blocks unsolved.

---

# 7. Fixes made (this rewrite, vs. the previous implementation)

**Simulation core**
- Subprocess return code was never checked (a crashed SfePy run looked like
  success) — now checked, raises with captured output on failure.
- stdout/stderr were never captured — now captured via `subprocess.run(capture_output=True)`.
- The real process-global `os.environ` was mutated and never restored — now a
  local `env` dict passed to the subprocess only.
- Solver convergence was never checked, only the return code — `report_status`
  added to the Newton solver config; every nonlinear solve's `cond` is scanned
  and a non-convergence raises with the specific failing line(s).
- `Exo_format.py`'s unstructured export used a hardcoded `NUM_SIDE_BLOCKS = 6`
  positional slice to separate volume from non-volume blocks — broke silently
  for any mesh with more than 6 non-volume blocks (multiple faults/wells/
  sources). Now classifies by actual cell dimensionality.
- `export_mesh_results`'s Exodus branch defaulted to `UNSTRUCTURED` type
  regardless of the mesh's real type — now reads `MeshResults.mesh_type`.
- A real `MemoryError` in unstructured meshing's surface-fitting step
  (`scipy.interpolate.Rbf` building one dense eval matrix in a single call) at
  fine structural-grid resolutions — now evaluated in memory-bounded chunks.
- Unstructured meshing computed a lithology-per-block mapping internally but
  never wrote it to `MeshResults.cell_data["block_id"]` — fixed at the source
  (removed a downstream cKDTree fallback that existed only because of this gap).
- GMSH on Windows breaks bare-executable-name subprocess launches for the rest
  of the process after `gmsh.initialize()`/`finalize()` has run once — `sfepy-run`
  is now resolved to its absolute path before launching, sidestepping it.
- `plot_variable_at_a_time`/etc. rebuilt a full VTK grid from scratch on every
  call — including twice per two-timestep plot (diff/cross-section functions)
  and once per saved step in the time-series loop — even though a fixed-mesh
  solve's geometry never changes between saved steps. Now builds each grid
  once and reuses it across timesteps (checked via `_same_geometry`, not
  assumed). `load_vtk_results` has the same fix one level down: reuses the same
  node/cell/celltype array objects across timesteps instead of storing an
  independent copy per saved step.
- Duplicate/mismatched scalar-bar rendering in every 3D plot — fixed via
  `add_mesh`'s own `scalar_bar_args` instead of a separate `add_scalar_bar()` call.
- `extract_surface()`'s default algorithm silently produced an empty mesh for
  this pipeline's grids — fixed via `extract_surface(algorithm=None)`.

**Workbench/DSL**
- `input_file: BasicallyABufferedFile` (a direct-upload widget) didn't work at
  all in the live frontend ("Unsupported input type file_import_export") — the
  only place in this codebase that type was ever used as a component *input*
  rather than an export/download. Switched to a `RemoteFile`-controlled path
  (`CustomSfepyFileDataType`), the same mechanism every other file-input
  component in this codebase already uses successfully.
- A `Union[HydrothermalProblemBuilder, CustomSfepyBuilder]`-typed port silently
  registered as only the union's first member's type (no real union semantics)
  — replaced with the `SfepyProblem` envelope so both builders' outputs share
  one real Workbench type.
- `is_object_type` was being auto-inferred `True` for `Build Custom SfePy
  Problem` purely from having a file-type parameter, collapsing it into a
  single node instead of the normal Build/Result presentation — fixed via an
  explicit `is_object_type=False` override.
- Dependency cleanup: removed `mpi4py`/`petsc4py`/`petsc` from `requirements.txt`
  — no petsc solver path exists anywhere in this pipeline (only
  `scipy_direct`/`scipy_iterative`), so these were dead weight (and by far the
  slowest packages to build).

# 8. Testing

`tests/tests_simulation_components/`: 96 tests, all passing (fast construction/
validation/dispatch tests plus real end-to-end `sfepy-run` invocations —
plain, pure-conduction-must-be-exactly-static, fault+flow, custom-builder fault
zone). Full repository suite: 580 passed, 1 pre-existing unrelated skip, 0
failed. Plotting render bodies are deliberately not unit-tested (matches this
codebase's existing convention); pure-computation helpers (grid construction,
geometry-reuse checks, axis projection) are.

Manual/physical sanity checks beyond the automated suite: each example run
end-to-end and checked for physically sensible results (temperature gradients
relaxing toward expected steady states); fault-zone properties set identical
to the surrounding rock and confirming the result is unchanged (isolates the
region-splitting/velocity-handoff plumbing from the physics itself); the
exact-equivalence demonstration in §2.
