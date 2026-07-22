  # Process Simulation — implementation notes for reviewers

This directory replaces the previous `core/simulation_components` implementation
entirely, and is now wired into the Workbench. This document is structured for two
different reviewers:

1. Whoever reviews the simulation implementation itself (start at "1. Implementation").
2. Whoever owns meshing (`core/meshing_components/`) and the Workbench/DSL layer
   (`py_api_wbgeo`) — sections 3 and 4 are written specifically for you, so you can
   review just the parts that touch your area without reading the whole thing.

This revision folds in everything since the previous review pass (through commit
`a344086`): a second problem-definition path (`CustomSfepyBuilder`, section 2.1 —
was "researched, not implemented," is now shipped), a fault-representation
investigation with a concrete, proven answer (section 1.4), an Exodus export bug
this branch's own `CustomSfepyBuilder` work surfaced and fixed (section 3.4 — this
was the "export_exodus regression" review comment), and a real MemoryError found
and fixed in unstructured meshing's surface-fitting step (section 3.6, new).

---

# 1. Implementation

## 1.1 Why a rewrite instead of a patch

The previous implementation had three concrete correctness bugs (see "Bugs found
and fixed" below) and a hand-written, static SfePy input file (`Hydro_thermal.py` /
`Thermal.py`) that every example copied and edited by hand whenever rock properties
or boundary conditions needed to change. Rather than patch around those issues, the
input-file generation, execution, and result handling were rebuilt from scratch,
matching the Workbench's component/connector philosophy more directly: a builder
component that turns typed inputs (mesh, structural model, rock properties) into
SfePy problem files, and a separate run component that executes them and returns a
typed result.

## 1.2 Architecture

Four files, one of which (the builder) now defines **two** problem-definition
classes sharing one runner:

- **`simulation_packages/sfepy/sfepy_hydrothermal_builder.py`** —
  - `HydrothermalProblemBuilder`. Takes a `MeshResults`, `StructuralModelResults`, and
    physical properties (`RockUnitProperties` per lithology, `FluidProperties`), and
    generates the actual SfePy problem-description `.py` files (regions, materials,
    equations, solvers) as strings written to disk. This is the input-data-generator
    component — its constructor arguments are the component's configurable inputs,
    matching how other Workbench components work. It is itself a `@wbgeo_type`
    (a pydantic dataclass) — see section 4.2.
  - `CustomSfepyBuilder` (new, section 2.1) — wraps a user-supplied, already-complete
    SfePy input file instead of auto-generating one. Validated against the mesh at
    construction time (known mesh type, known lithology mapping, no wells/sources,
    every `'cells of group N'` the file references actually exists), then runs
    through the same `run_simulation_sfepy()` as `HydrothermalProblemBuilder`. Also a
    `@wbgeo_type` dataclass, same reason as above.
  - Several standalone module-level functions factored out so both classes (and the
    Workbench pre-check layer, section 4.4) can reuse the same logic without one
    depending on the other: `map_mat_id_to_lithology`, `enumerate_rock_units`,
    `lithology_to_group_idx`, `compute_fault_zone_cell_mask`,
    `check_mesh_has_known_type`, `check_mesh_has_lithology_mapping`,
    `check_mesh_has_no_engineering_objects`, `check_custom_sfepy_regions`.
    `HydrothermalProblemBuilder`'s own equivalent private methods are now one-line
    wrappers around these — a pure refactor, zero behavior change (re-verified via
    the existing `HydrothermalProblemBuilder` test suite passing unmodified).
- **`simulation_packages/sfepy/sfepy_hydrothermal_run.py`** — `run_simulation_sfepy(problem)`,
  the run component. Dispatches on `isinstance(problem, CustomSfepyBuilder)`
  (`_run_hydrothermal_problem` for the auto-generated two-stage case,
  `_run_custom_sfepy_problem` for a custom file — one `sfepy-run` invocation, no
  stage handoff) so both problem types share one runner, one result type, and the
  Workbench needs only one "Run" component per builder rather than diverging
  behavior to keep in sync. Also converts the mesh to the format SfePy actually
  reads (Exodus → Medit, via meshio), and `export_simulation_results()`, a
  "Download Simulation Results" component (VTK only so far).
- **`simulation_visualization/simulation_visualization.py`** — plotting functions for
  `SimulationResults` (3D snapshots, cross-sections, before/after diffs) and a
  pre-flight material-assignment check, `plot_builder_materials(builder)`, that lets
  a wrong material/fault-zone assignment be caught by eye before running a solve
  that can take a while. Accepts either builder type now: for
  `HydrothermalProblemBuilder` it shows the full `RockUnitProperties` summary per
  material (porosity/permeability/k/rho·c) in the legend, same as before; for
  `CustomSfepyBuilder` there's no such structured properties object to draw from (a
  custom file writes its `materials` block directly in raw SfePy syntax), so the
  legend falls back to bare material names — same coloring/grouping/fault-zone
  section otherwise, both paths share one function body with an internal branch,
  not two separate plot functions. Also includes three point/line post-processing
  utilities ported from the previous implementation (`plot_variable_along_line`,
  `print_variable_at_point`, `plot_variable_time_series`) — initially dropped in the
  rewrite, restored on request since they're useful post-processing tools not
  covered by the snapshot/cross-section plots. `print_variable_at_point`'s
  out-of-bounds detection was fixed while porting it: the original checked
  `sampled.n_points == 0`, which never actually triggers for a single-point
  `PolyData` (PyVista's probe filter always returns one output point per input
  point, valid or not); now checks `sampled["vtkValidPointMask"]` instead, which is
  what actually distinguishes a valid sample from an out-of-bounds one.
- **`simulation_workbench_components.py`** (`core/simulation_components/`, not
  under `simulation_packages/`) — the actual `@wbgeo_component`-decorated Workbench
  components (thin wrappers around the three files above). See section 4.

## 1.3 Physics model

Coupled single-phase Darcy flow + advective-conductive heat transport, solved as
**two sequential single-physics stages** (pressure first, then temperature), not one
combined system: mixing pressure (~1e6 Pa) and temperature (~10s of °C) in one
matrix is badly conditioned for iterative solvers, regardless of preconditioning.
Segregating matches the physics anyway: temperature depends on the flow field, but
pressure never depends on temperature in this model. Each stage is a
well-conditioned single-physics problem on its own once segregated. This
segregation, and everything else in this section, is specific to
`HydrothermalProblemBuilder`'s own auto-generated physics — a `CustomSfepyBuilder`
file can implement any physics/staging it wants (section 2.1).

`t1` (the transient stage's end time) defaults to a data-driven thermal diffusion
timescale (`L² / α`) rather than an arbitrary constant. An arbitrarily short `t1`
relative to this timescale (this model's own rock properties and cell sizes
typically imply a diffusion time on the order of years) makes the heat equation's
transient term dwarf the diffusive term by many orders of magnitude in the
assembled matrix — an internal scale mismatch that defeats preconditioning
regardless of solver choice.

`include_flow=False` skips the pressure/Darcy stage entirely, for pure-conduction
validation against a known analytical steady-state profile before trusting the
coupled result.

## 1.4 Fault support (limited, opt-in) — and why a full fault model isn't feasible today

`fault_zone_properties` (a `RockUnitProperties`) gives cells near any fault their
own separate material — a "damage zone" band `fault_zone_n_voxels` grid cells wide,
computed on the structural model's grid via each fault's `get_domain_mask()`, then
resampled onto mesh cells. Off by default; does not change behavior when unset.
`CustomSfepyBuilder` gets the identical mechanism via its own `fault_zone_n_voxels`
field (`compute_fault_zone_cell_mask`, shared with `HydrothermalProblemBuilder` —
see 1.2) — a custom file references the resulting cell group exactly like any
lithology group, via `'cells of group {fault_group_id}'`; no material/equation is
auto-generated for it, that part is left entirely to the custom file's author.

This is deliberately not a complete fault model: all faults in a structural model
share one combined material region rather than per-fault properties, and only a
single band-around-the-trace geometry is supported (not an exact hanging-wall/
footwall split). It correctly respects `fault_activity` though — a fault that only
affects some stratigraphic groups (finite faults, e.g. model9) only gets a fault
zone in the groups it's actually active for, gated by the same `group_idx >=
youngest_idx` rule the real `lith_block` computation uses
(`effective_domain_components_for_group` in `general.py`). Verified on model2:
restricting a fault to only its older stratigraphic group correctly excludes the
younger group's cells from the fault zone (764 → 596 affected cells at test
resolution), confirmed both numerically and visually via `plot_builder_materials`.
Also verified at the official example's full resolution (125×50×50): 64,688 active
fault-zone cells, reproduced bit-for-bit identically through both
`HydrothermalProblemBuilder` and an independently-authored `CustomSfepyBuilder` file
solving the same problem (section 1.7).

**Why not a real fault surface (a 2D interface with its own jump condition), instead
of a 3D damage-zone band?** This was investigated in depth for the
`CustomSfepyBuilder` work and explicitly rejected, not just left unexplored:

- SfePy itself has no fault/interface element for flow problems as a named,
  pre-built term — checked the `terms/` and `examples/` directories directly; the
  only contact-mechanics module that exists is for elastic solid-body mechanical
  contact, not applicable here.
- It *does* have a generic interface jump term, `dw_jump` (`SurfaceJumpTerm` in
  `sfepy/terms/terms_surface.py`, `integration = 'facet'`) — the standard way to
  represent a sealing/leaking fault plane as ∫ c·q·(p₁−p₂) across an internal
  boundary. This is the term a real fault-surface representation would need.
- `dw_jump` requires a **non-conforming** mesh: independent, duplicated degrees of
  freedom on each side of the interface, not a single shared node. This was checked
  empirically, not assumed — on a real faulted unstructured mesh (model2, ~1.7M
  elements): zero near-duplicate node coordinates anywhere in the mesh (checked via
  a cKDTree nearest-neighbor search), and 100% of the fault surface's own nodes are
  shared by tetrahedra on *both* sides of the fault (checked per-node, via a
  PCA-derived local normal to classify which side each neighboring tetrahedron is
  on).
- That's a direct consequence of how `create_unstructured_mesh_data` builds the
  mesh: `gmsh.model.occ.fragment(...)` (OpenCASCADE boolean fragmentation, in
  `create_grid_fragment_surface.py`) produces a watertight, conforming partition by
  design — shared nodes at every internal boundary, including the fault plane. That
  is the entire point of using `occ.fragment` (a single consistent volume mesh
  across all lithology/fault domains), and changing it would be a different, far
  larger meshing-architecture project, not a simulation-side fix.
- So a real `dw_jump`-based fault surface is not obtainable from this pipeline's
  mesh output today — not a "harder to implement," a "the mesh this depends on is
  structurally the wrong shape for it." The damage-zone band (a 3D volume material,
  no different in kind from any lithology's own volume term) has no such
  requirement, which is why it's the only fault representation this pipeline
  supports, for both builder types.

## 1.5 Lithology-to-mesh mapping — trusted from meshing, not re-derived

Both builder types need to know which lithology each mesh block corresponds to,
via the shared `map_mat_id_to_lithology` (section 1.2), which trusts
`MeshResults.cell_data["block_id"]` directly for all three mesh types, with one
explicit `ValueError` at construction time if it's missing entirely. This used to
require an independent cKDTree-based fallback for unstructured meshes specifically,
because meshing computed the mapping internally but never wrote it to the output —
that gap is now fixed at the source; see section 3.2 for the meshing-side details.

## 1.6 Bugs found and fixed vs. the previous implementation

- **Subprocess return code was never checked.** A crashed/errored SfePy run
  silently looked like success; the only symptom was zero output files loaded
  several call frames away, logged at INFO level. Fixed: checks the return code,
  raises with the captured output on failure.
- **stdout/stderr were not captured**, so the actual SfePy error just scrolled past
  in the console with no way to recover it programmatically. Fixed: captured via
  `subprocess.run(capture_output=True)`.
- **The real process-global `os.environ` was mutated** (`os.environ["SFEpy_OUTPUT_DIR"]
  = ...`) and never restored. Fixed: builds a local `env` dict passed via
  `subprocess.run(env=...)`, which only affects the child process.
- A duplicate/redundant scalar bar bug in the visualization functions (every plot
  was calling both `add_mesh(scalars=...)`, which auto-adds its own scalar bar, and
  a separate `add_scalar_bar()` — two visibly mismatched-colormap bars stacked on
  screen). Fixed by setting the title via `add_mesh`'s own `scalar_bar_args`
  instead.
- `extract_surface()`'s default algorithm silently produced an empty mesh (0
  points) for this pipeline's SfePy-exported grids. Fixed by using
  `extract_surface(algorithm=None)` everywhere (matches the deprecated
  `extract_geometry()`'s actual behavior).
- **Solver convergence was never checked**, only the subprocess return code was.
  A returncode of 0 only means SfePy didn't crash — it says nothing about whether
  the nonlinear (Newton) solve actually reached tolerance within `i_max`
  iterations. Fixed: `'report_status': True` was added to the shared Newton solver
  config (`HydrothermalProblemBuilder._newton_ts_options_blocks`, used by both the
  pressure and heat stages), which makes SfePy print a definitive
  `cond: N, iter: ..., err0: ..., err: ...` line per nonlinear solve (`N`: 0 =
  converged, 1 = max iterations reached, 2 = linesearch gave up — SfePy's own
  codes, from `conv_test()` in `sfepy/solvers/nls.py`). `_run_sfepy_input_file`
  now scans stdout for every such line (a transient run does one nonlinear solve
  per time step, so an earlier step failing while a later one happens to succeed
  must not go unnoticed) and raises with the specific failing line(s) if any
  `cond != 0`. This check applies uniformly to both builder types — it lives in
  `_run_sfepy_input_file`, which both `_run_hydrothermal_problem` and
  `_run_custom_sfepy_problem` call.

Two more bugs were found and fixed later, while building `CustomSfepyBuilder` —
both are meshing-side, so they're written up in section 3 (3.4 and 3.6) for the
meshing reviewer, with only a cross-reference here: an Exodus export block-splitting
bug with a hardcoded assumption that broke for non-standard block counts, and a
real MemoryError in unstructured meshing's surface-fitting step at high structural
grid resolution.

## 1.7 Verification approach

**Automated test suite**: `tests/tests_simulation_components/` (95 tests, all
passing — 88 fast + 7 slow/end-to-end). Coverage (via `coverage run
--source=core.simulation_components -m pytest tests/tests_simulation_components/`):
`sfepy_hydrothermal_builder.py` 98%, `sfepy_hydrothermal_run.py` 90%,
`simulation_workbench_components.py` 97%, `simulation_visualization.py` 40% (the
rest is `plot_*` rendering bodies, deliberately not unit-tested — matches this
codebase's existing convention of not asserting on plot output). Covers:
construction validation for both builder types, rock-unit/fault-zone logic
(including the exact 764/596/64,688 fault-cell-count regressions from section
1.4), generated/uploaded SfePy input-file text (validated via `compile()`, not
always a real solve, where possible), the JSON round-trip every `@wbgeo_type` gets
between Workbench component executions, and several real end-to-end `sfepy-run`
invocations (plain, pure-conduction-must-be-exactly-static, fault+flow, and a
custom-builder fault-zone case). `test_custom_sfepy_builder.py` alone has 12 tests;
`test_hydrothermal_run.py` and `test_simulation_workbench_components.py` add 8 more
covering the dispatcher, the pre-checks, and end-to-end custom-file solves.

**Full repository suite**: 579 passed, 1 pre-existing unrelated skip, 0 failed
(`python -m pytest`, ~10 minutes, all markers including slow/integration tests
across every component, not just simulation).

**Manual/physical sanity checks**, still relevant beyond what the automated suite
covers: running each example end-to-end and checking the result is physically
sensible (temperature gradients relaxing toward expected steady states, boundary
conditions matching configured values); a specific correctness check for the
fault-zone feature (setting `fault_zone_properties` identical to the surrounding
homogeneous rock properties and confirming the result is unchanged to floating-
point/solver-tolerance noise — isolates the region-splitting and Darcy-velocity-
handoff plumbing from whether the physics itself is right); and the lithology-
mapping accuracy comparison in section 3.2 (ground-truth cross-check via
independent nearest-neighbor lookup, not just "did it run without error").

**Exact-equivalence demonstration (new)**: for `CustomSfepyBuilder` specifically,
"does this actually work, for real" was checked by literally reproducing
`HydrothermalProblemBuilder`'s own generated input files by hand-equivalent means —
writing a self-contained custom SfePy file (importing only `sfepy`/`numpy`/
`pyvista`, zero WBGeo imports) that performs the same two-stage
pressure→velocity→heat sequence as a module-level Python side effect (exploiting
that `sfepy-run` genuinely `__import__()`s its input file, executing all top-level
code before reading `regions`/`materials`/etc.) — then running it through the real
`CustomSfepyBuilder`/`run_simulation_sfepy` pipeline and diffing against the
reference. Done for both example models, at each model's own full official
resolution and real (non-default) rock/fluid/fault-zone properties — not a
simplified stand-in:
- **model1** (no fault, implicit structured mesh, 50×50×50, `include_flow=True`,
  3 heat steps): `T` matched to `0.0` max difference, node-for-node.
- **model2** (real fault, active damage zone, unstructured mesh, 125×50×50,
  `include_flow=False`, pure conduction, 2 heat steps): same, `0.0` max difference,
  including the 64,688-cell fault zone.

Both files are committed as worked examples
(`examples/synthetic_examples/model{1,2}/input_data/simulation_files/`), wired into
`WBGeo1.0_model1.py`/`WBGeo1.0_model2.py` right after each script's own
`HydrothermalProblemBuilder` call, running both builders against the identical
`structural_model_result`/mesh objects and printing both results' `T` mean for a
direct side-by-side comparison. This is the concrete answer to "could a real user
actually replicate what the builder does with the custom path" (section 2.1) —
not a hypothetical.

The convergence check itself (`_check_convergence`) is unit-tested against
synthetic stdout strings (converged, non-converged, linesearch-failure, and a
mixed multi-step case where only one of several nonlinear solves fails, confirming
it scans every match, not just the last), and was originally validated against a
real reproduced failure too: `include_flow=True` with a deliberately too-short
`t1` genuinely diverges on the second time step, and the check correctly raises
with that exact line. That reproduction needed `include_flow=True` specifically:
with `include_flow=False`, the heat stage's initial condition is already the exact
pure-conduction steady state (see `ic_temp`'s docstring in the builder), so there
is nothing to diffuse and the solve trivially converges in one Newton iteration
regardless of `t1` — the scale-mismatch problem only shows up once advection
perturbs the field away from that steady state.

---

# 2. Limitations

## 2.1 Custom SfePy input file support — now implemented (`CustomSfepyBuilder`)

A user can now bring a complete, hand-written SfePy input file instead of using
`HydrothermalProblemBuilder`'s auto-generated one — `CustomSfepyBuilder` (section
1.2), wired into the Workbench as **Build Custom SfePy Problem** / **Run Custom
SfePy Simulation** (section 4.1). It is not restricted to hydrothermal physics
either — the file can define any SfePy problem; only the mesh/region bookkeeping
is validated (known mesh type, known lithology mapping, no wells/sources, every
`'cells of group N'` referenced actually exists on the mesh — `check_custom_sfepy_regions`,
a text-based scan of the file's source for that selector pattern, not an `exec()`
of the file at construction time — see that function's docstring for why: SfePy
input files can use either a module-level `regions = {...}` dict or a top-level
`define()` function, and a text search works for both without ever executing
untrusted uploaded code before an explicit "Run" action).

**The key remaining caveat, by design, not oversight**: `run_simulation_sfepy()`
dispatches on type and calls either `_run_hydrothermal_problem` (the auto-generated
two-stage pressure→heat sequence) or `_run_custom_sfepy_problem` (one `sfepy-run`
invocation, no stage handoff at all) — there is no shared multi-stage orchestration
mechanism between the two. A user wanting to replicate `HydrothermalProblemBuilder`'s
own two-stage pattern (or any other multi-stage physics) with `CustomSfepyBuilder`
must write that staging themselves, inside their own input file, as ordinary
top-level Python — which works because `sfepy-run` (via
`sfepy.base.conf.ProblemConf.from_file` → `sfepy.base.base.import_file`) genuinely
imports the input file as a Python module, executing every top-level statement
(including, e.g., solving an earlier stage in-process via SfePy's own library API —
`ProblemConf.from_file` + `Problem.from_conf` + `.solve()` — and stashing the result
somewhere the "official" conf below it can read) before ever reading the
`regions`/`materials`/`equations` variables `sfepy-run` actually resolves. This is
not a workaround or a trick specific to one example — it's a direct, general
consequence of how SfePy loads input files, documented and demonstrated in the two
worked examples referenced in section 1.7. It does mean the orchestration logic
lives in the user's file rather than being reusable/composable the way
`HydrothermalProblemBuilder`'s two Python methods (`build_pressure_input_file`,
`build_heat_input_file`) are — there is no lower-effort path today for a multi-stage
custom problem than writing that sequencing by hand each time.

An earlier design alternative — a shared `Stage`/`get_stages()` interface that both
builder types would implement, with `run_simulation_sfepy()` reduced to one generic
loop over `.get_stages()` — was considered but not built. What's implemented instead
(a simple `isinstance` dispatch, one non-generic runner function per builder type)
is less abstract but was sufficient for what actually shipped; `Stage`/`get_stages()`
remains a reasonable direction if a second built-in multi-stage physics ever needs
to share plumbing with `HydrothermalProblemBuilder`, but nothing today depends on it.

`CustomSfepyBuilder` also supports the same opt-in fault damage-zone mechanism as
`HydrothermalProblemBuilder` (`fault_zone_n_voxels`, section 1.4) — a real fault
surface representation is not available to either builder, for the meshing reason
explained there, not a builder-specific gap.

## 2.2 Fault support: damage-zone only, for both builder types

See section 1.4 for the concrete investigation and proof: SfePy's own fault/
interface term (`dw_jump`) requires a non-conforming mesh, and
`create_unstructured_mesh_data`'s use of `occ.fragment` produces a conforming one
by construction — not fixable from the simulation side without a materially
different meshing approach. Both `HydrothermalProblemBuilder` and
`CustomSfepyBuilder` are limited to the same damage-zone-band representation as a
result: one shared material region for all faults, no per-fault properties, no
exact hanging-wall/footwall split.

## 2.3 No Exodus export for `SimulationResults`

`export_simulation_results()` only supports `format="vtk"`. Exodus (mirroring
`export_mesh_results_to_exodus`, already used internally for the mesh side) would
be the natural next format if needed.

## 2.4 Convergence check only catches SfePy self-reported non-convergence

`_check_convergence` only catches SfePy reporting `cond` 1 or 2 (max iterations
reached / linesearch gave up). It cannot catch a solve that satisfies Newton's
tolerance trivially against a physically-wrong state — e.g. a badly-scaled system
where the initial guess already nearly zeroes the residual for the wrong reason.
Judging physical correctness, not just numerical convergence, still requires the
sanity-checking approach described in section 1.7. Applies identically to both
builder types (the check lives in the shared `_run_sfepy_input_file`, section 1.6).

## 2.5 Structured mesh + fault: blocked, not just discouraged

Cross-checking all three mesh types against each other on a faulted model at
matched settings: `implicit` and `unstructured` agree well (domain-mean T within
~1%, identical min/max, close std), but `structured` genuinely fails to converge
(the convergence check catching a real numerical issue, not a false positive) —
confirmed specific to the fault+structured-mesh combination, not structured
meshing generally (the identical builder settings on an unfaulted model converge
cleanly on a structured mesh). Not root-caused (suspected: structured mesh's cell
geometry/connectivity interacting badly with the fault-zone region split).

This is no longer just a documentation note: `create_structured_mesh_data` now
actively rejects a faulted `geomodel_result` (see section 3.3) — the underlying
non-convergence problem still isn't fixed, but the combination can no longer be
constructed through the normal Workbench path, for either builder type (the block
happens at mesh-generation time, before either builder ever sees the mesh). A
residual gap: `mesh_results` and `geomodel_result` are still two independent inputs
at the builder level, so a structured mesh built before this check existed (or from
any other source) could still be paired with a faulted `geomodel_result` there,
recreating the problem. A second, redundant check at that level was proposed and
deliberately not implemented (judged a fringe case for now) — see section 4.4.

## 2.6 `CustomSfepyBuilder` does not support wells/sources

`check_mesh_has_no_engineering_objects` rejects any mesh with a well/source block
(`dim < 2`) outright, for `CustomSfepyBuilder` only. This is stricter than
`HydrothermalProblemBuilder`, which never rejects such a mesh — it simply never
generates any equation for those blocks, so they're silently left unsolved (an
existing, unchanged behavior, not something this pass touched). For a custom file,
where there is no way to know whether the file's author expected an engineering
object to be usable, a loud rejection at construction time was judged safer than
replicating that same silent gap for a case where the user has direct control over
the input file and might reasonably expect it to matter. Not supported yet, not
architecturally blocked — a future extension if there's a real use case.

---

# 3. Changes in meshing (`core/meshing_components/`, `core/object_components.py`)

This section is for whoever owns meshing — it lists every change this branch made
outside `core/simulation_components/`, and why.

## 3.1 `MeshResults.mesh_type` — mesh provenance moved onto the base model

`HydrothermalProblemBuilder` used to require a separate `mesh_type` constructor
argument ("implicit"/"structured"/"unstructured") alongside `mesh_results`, purely
because `MeshResults` (`core/object_components.py`) didn't record which meshing
component produced it — two places that had to agree, with nothing checking they
actually did. Fixed at the source: `MeshResults` now has a
`mesh_type: Optional[MeshType] = None` field, `MeshType` a new `StrEnum`
(`IMPLICIT`/`STRUCTURED`/`UNSTRUCTURED`) defined right next to `MeshResults`.
`create_implicit_structured_mesh`, `create_structured_mesh_data`, and
`create_unstructured_mesh_data` (the only three ways a `MeshResults` gets built for
real) all set it automatically now.

`Optional` rather than required: ~13 existing meshing-format tests
(`tests/tests_meshing_components/mesh_formats/`) construct bare `MeshResults(nodes=...,
elements=...)` directly to test export logic in isolation, with no need for
provenance — making the field required would have forced pointless changes to all
of them for no benefit. Full `tests/tests_meshing_components/` suite (240+ tests,
including the additions from section 3.2/3.3) passes.

`export_mesh_results_to_exodus`'s own `type` parameter
(`core/meshing_components/mesh_format/exodus/Exo_format.py`) keeps its own
separate `MeshType` short-code enum (`"imp"`/`"str"`/`"unstr"`, not the same class
as the new one) rather than being unified with `MeshResults.mesh_type` —
deliberately: the two represent different things. `MeshResults.mesh_type` is
factual provenance; `Exo_format`'s `type` is a configuration knob selecting which
export code path to run, and the two can legitimately diverge (see 3.4). Existing
`Exo_format.MeshType` values are also depended on explicitly by several existing
tests, so left unchanged; one new value, `MeshType.VOLUME_ONLY`, was added there
(see 3.4).

## 3.2 Unstructured meshing: `block_id` propagation + lithology mapping mode rename

Two related fixes/changes in `core/meshing_components/explicit/unstructured/mesh_data.py`:

**`cell_data["block_id"]` propagation (real gap, now fixed).** Unstructured meshing
computed a lithology-per-block mapping internally but never wrote it to the
output `MeshResults.cell_data` — implicit and structured meshing already did.
This is why `HydrothermalProblemBuilder` used to carry its own independent
cKDTree-based fallback for unstructured meshes specifically (resampling
individual cell centroids and taking a majority vote against the structural
model's `lith_block`). That fallback is gone now that the mapping is fixed at the
source: `mesh_generator` now builds `cell_data["block_id"]` from the same
per-block lithology assignment it already computes internally, with a `-1`
sentinel for every block that isn't a lithology volume (shafts, wells, sources,
fault surfaces, triangulated/boundary surfaces) — left out entirely for
`mapping_litho="manual"`/`"none"`, which have no inherent per-block lithology
mapping to propagate.

**`mapping_litho` rename + new default.** The two automatic-mapping modes were
renamed to describe what they actually vote on: `"auto"` → `"automatic_corners"`
(votes on each raw block's *node* coordinates against the structural grid) and the
new `"automatic_dev"` → `"automatic_centers"` (votes on each block's *cell
centroids* instead). Old values are no longer accepted at all (confirmed no live
call site anywhere in this repo — examples, tests — passed the old values, only
comments referencing them, all updated). `"automatic_centers"` is now the
**default** for both `mesh_generator` and `create_unstructured_mesh_data`.

Why centers over corners: nodes sit on block boundaries — exactly where a fault
plane tends to be — so a fault-adjacent block's node vote can be genuinely
ambiguous regardless of algorithm quality; cell centroids are guaranteed-interior
points, avoiding that failure mode. Verified on a deliberately mismatched
structural-grid/mesh resolution (structural grid voxel ~3x the mesh size, to
stress-test this): overall cell-level accuracy against ground truth was 73.3% for
centers vs. 68.1% for corners (~16% relative reduction in misclassified cells) —
not a clean win on every individual lithology, but better overall, and produced
fewer/less-severe low-confidence blocks. At matched structural-grid/mesh
resolution, both methods were equivalent. Neither mode fully fixes the case where
the structural grid is much coarser than the mesh — that's a real, separate
problem no purely-geometric voting method solves; `_check_lithology_mapping_reliability`
(new, in the same file) logs warnings when it detects this (voxel size vs. mesh
size ratio, more merged blocks than expected lithologies, low vote-agreement
ratio per lithology) rather than trying to fix it.

## 3.3 Structured meshing: rejects faulted structural models

New `pre_check_no_faults_for_structured_mesh(geomodel_result)` in
`core/meshing_components/explicit/structured/mesh_data.py`, wired both as
`create_structured_mesh_data`'s `@wbgeo_component` `input_checks=` (blocks the
connection in the Workbench UI before execution) and as an explicit call inside
the function itself (pre-checks can be skipped by a direct Python caller — see
docs/developers/components.md's "Pre-checks" section). Raises `ValueError` if
`geomodel_result.structural_frame.fault_frame` has any fault elements. Motivated by
section 2.5's non-convergence finding: structured meshing builds a mesh without
error for a faulted model (it has no fault-awareness at all), but that combination
does not reliably converge when later solved with SfePy.

## 3.4 `Exo_format.py`: hardcoded block-splitting bug — found and fixed

`ExodusInput.write()`'s unstructured-mesh branch used to strip a hardcoded
`NUM_SIDE_BLOCKS = 6` trailing blocks via a positional slice to separate volume
blocks from boundary/fault/well/source blocks — correct only when there were
exactly 6 non-volume blocks total. A model with multiple faults, wells, or sources
could easily exceed 6, silently corrupting the exported mesh (a mixed
`{'triangle', 'tetra'}` "volume" block, wrong element count) for any caller with a
non-standard block count. **This is the "export_exodus regression" flagged in
review** — found while building `CustomSfepyBuilder` (which, unlike
`HydrothermalProblemBuilder`, has no fixed assumption about how many non-volume
blocks a mesh might have) and fixed at the source: `write()` now classifies blocks
by their actual dimensionality (`volume_blocks = [b for b in all_blocks if b.dim
== 3]`, `boundary_blocks = [b for b in all_blocks if b.dim == 2]`) instead of a
fixed count. Verified on a real faulted model2 mesh: the old slice-based logic
would have produced exactly the corrupted-block symptom described above; the new
logic correctly isolates the tetra-only volume blocks regardless of how many
boundary/fault/well/source blocks exist alongside them.

Added `Exo_format.MeshType.VOLUME_ONLY = "volume_only"` alongside this fix —
purely additive, behaves exactly like `STRUCTURED`/`IMPLICIT` in `write()`
(`volume_blocks = all_blocks`, no splitting needed). Needed because
`_run_sfepy_input_file()` pre-filters an unstructured mesh down to its tetra-only
volume blocks before export (see 3.5), and needed a way to say "this mesh is
already volume-only" without mislabeling its true `mesh_type` as `"imp"` just to
get that code path. Doesn't touch or reorder any existing enum values, so no risk
to existing tests depending on `"imp"`/`"str"`/`"unstr"`.

`export_mesh_results`'s Exodus branch had a related bug fixed at the same time: it
called `export_mesh_results_to_exodus(mesh)` with no `type=` argument at all, which
silently defaulted to `UNSTRUCTURED` regardless of the mesh's true type — crashing
for implicit/structured meshes routed through that generic entry point. Fixed by
adding `exodus_type_for(mesh_results) -> MeshType`, which reads the mesh's own
`MeshResults.mesh_type` (section 3.1) to pick the right `Exo_format.MeshType`
automatically.

## 3.5 GMSH + Windows `CreateProcess` conflict — root-caused, worked around in simulation code

Not a meshing-code change, but the root cause is in GMSH (used by
`create_unstructured_mesh_data`), so meshing should know about it: on Windows,
after GMSH's `initialize()`/`finalize()` has run anywhere earlier in the same
process, launching a subprocess by **bare executable name** (e.g. `"sfepy-run"`)
fails with `FileNotFoundError: [WinError 2]`, even though `PATH` is untouched and
`shutil.which()` still resolves it correctly. Reproduced with the minimum possible
case: `import gmsh; gmsh.initialize(); gmsh.finalize()` alone breaks every
subsequent bare-name subprocess launch in that process — confirmed
`initialize()`/`finalize()` are already properly paired in
`create_unstructured_mesh_data`'s try/finally, so it isn't simply "GMSH left
initialized." It's specifically Win32 `CreateProcess`'s own executable-search step
(only used for bare names) that GMSH's `initialize()` breaks at the OS/DLL level.

**Fixed on the simulation side only**: `_run_sfepy_input_file` resolves
`"sfepy-run"` to its absolute path via `shutil.which()` before calling
`subprocess.run`, sidestepping `CreateProcess`'s search entirely. This does **not**
fix the underlying GMSH/Windows behavior — any *other* code in the same process
that launches a subprocess by bare name after GMSH has run would still fail. If
`py_runner` is a long-lived process serving multiple requests, one user triggering
unstructured meshing could break bare-name subprocess calls elsewhere in that same
process for every subsequent request until restarted — worth confirming with
whoever owns that infrastructure. A more thorough (unimplemented) fix would be for
`create_unstructured_mesh_data` to run GMSH in a child process, so the parent
process's `CreateProcess` behavior is never affected in the first place.

## 3.6 Unstructured meshing: real MemoryError in surface-fitting — found and fixed (new)

`create_surface_grid` (`core/meshing_components/explicit/unstructured/
create_grid_fragment_surface.py`) fits a `scipy.interpolate.Rbf` per surface, then
evaluates it over a grid to hand GMSH a B-spline-fittable point cloud.
`Rbf.__call__` allocates one dense `(n_eval_points × n_source_points)` float64
distance matrix internally, in a single call — at a fine-enough structural grid
resolution, this produced a real, reproducible `MemoryError`
(`Unable to allocate 2.08 GiB for an array with shape (25000, 11180)`) while
building a model2 test mesh: the eval grid is capped at `250×100 = 25,000` points
(`max_n_gx`/`max_n_gy`), but the number of *source* points (deduplicated raw
surface points) scales directly with structural-grid resolution and is
**uncapped** — 11,180 source points here, from a 125×50×50 structural grid.

**Fixed**: `_evaluate_rbf_chunked(rbf, n_source_points, *coord_grids)` evaluates the
grid in batches sized to keep each dense allocation under ~256 MB, instead of one
allocation for the whole eval grid — same fitted weights, same distance function,
just computed in smaller pieces and concatenated. Verified numerically equivalent
to a direct (unchunked) call to float64 machine precision (differences on the
order of 1e-10–1e-14, from summation-order non-associativity across chunk
boundaries — not a correctness change). Re-verified the originally-crashing case:
building the model2 mesh at full 125×50×50 resolution with real mesh settings
(`tolerance=50, mesh_size=20, curve_mesh_size=5, ...`, matching
`WBGeo1.0_model2.py`) now completes without error, on a machine with less free RAM
(~3.5 GB) than the amount that originally crashed (~6.2 GB).

**Residual risk, not fixed**: this only bounds the *evaluation*-side allocation.
`Rbf`'s own *fit* step (solving for interpolation weights) still builds a dense
`(n_source × n_source)` matrix internally — at 11,180 source points that's already
~1 GB, and since source-point count is uncapped, a sufficiently fine structural
grid could still hit a MemoryError at fit time, before evaluation is ever reached.
Not yet observed in practice (fit-side cost only exceeded eval-side cost once
`n_source` approached the eval-grid cap in the case that was fixed), but a
denser/larger model than either example here could reach it. Two options if this
becomes a real problem: cap/bin the number of source points before fitting (the
existing dedup is exact-coincidence-only, not spatial downsampling), or switch from
the legacy `scipy.interpolate.Rbf` to `scipy.interpolate.RBFInterpolator` with a
`neighbors=` parameter (local/sparse RBF using only the k nearest source points per
evaluation point, avoiding both the `O(N²)` fit and `O(M·N)` eval matrices
entirely) — the latter would change results from exact to a well-controlled
approximation, not a drop-in swap.

---

# 4. Workbench wiring (`py_api_wbgeo` / DSL layer)

This section is for whoever reviews the Workbench/DSL side specifically.

## 4.1 Components

`core/simulation_components/simulation_workbench_components.py` (new file — the
old `simulation_workbench_components.py`-equivalent was deleted along with the
previous implementation) holds every `@wbgeo_component` for this pipeline, per the
project's "one workbench file per step" convention — thin wrappers only, no logic
of its own:

- **Build Hydrothermal Problem** (`build_hydrothermal_problem`) — takes
  `mesh_results`, `geomodel_result`, the SmartInput `options` (4.3), and plain
  scalar settings (`include_flow`, `enable_fault_zone`, `fault_zone_n_voxels`,
  `t0`/`t1`/`num_steps`, boundary values, solver settings), builds a
  `HydrothermalProblemBuilder` and returns it wrapped in `SfepyProblem`
  (`SfepyProblem(hydrothermal=...)`).
- **Build Custom SfePy Problem** (`build_custom_sfepy_problem`, new) — takes an
  uploaded `input_file`, `mesh_results`, optional `geomodel_result`, and optional
  `fault_zone_n_voxels`; decodes and validates the file (UTF-8, mesh/region
  bookkeeping, section 2.1), builds a `CustomSfepyBuilder` and returns it wrapped
  in `SfepyProblem` (`SfepyProblem(custom=...)`). Not restricted to hydrothermal
  physics.
- **`SfepyProblem`** (`sfepy_hydrothermal_builder.py`) — a thin `@wbgeo_type`
  envelope, `Optional[HydrothermalProblemBuilder]` + `Optional[CustomSfepyBuilder]`
  fields with exactly one set (enforced in `__post_init__`), plus an `.inner()`
  accessor. Exists purely so both Build components' outputs register as the
  identical Workbench type: this DSL matches a connector port by its one
  registered type, not by structural/union compatibility. A
  `Union[HydrothermalProblemBuilder, CustomSfepyBuilder]`-typed port was tried
  first and confirmed live (Docker) to silently register with only the Union's
  first member as its type — no actual union semantics, and it would have
  rejected a connection from the other builder's output. `SfepyProblem` sidesteps
  that at the Workbench boundary only; internally `HydrothermalProblemBuilder`/
  `CustomSfepyBuilder` are unchanged, still built/validated exactly as before.
- **Run Simulation** (`run_simulation`) — one component, `problem: SfepyProblem
  -> SimulationResults`, calling `problem.inner()` then the same
  `run_simulation_sfepy` dispatcher (section 1.2) either builder already used.
  Originally shipped as two separate components (`run_hydrothermal_simulation`/
  `run_custom_sfepy_simulation`) for the same "distinct typed connector socket"
  reason `build_*` is still split in two — unified once `SfepyProblem` made a
  single shared input type possible.
- **Export Simulation Results** (`export_simulation_results`) — thin wrapper over
  the run-file function of the same name.
- **Plot Materials** (`inspect_sfepy_problem_plot_materials`), attached to
  `SfepyProblem` — pre-flight material/fault-zone check via
  `plot_builder_materials(problem.inner())`, before running the (possibly slow)
  solve. One inspector for both builder kinds, same reasoning as `run_simulation`.
- Two more inspectors: **Plot Variable (Final Time)** and **Plot Cross Section
  2D**, attached to `SimulationResults`, matching the existing
  `StructuralModelResults`/`MeshResults` inspector pattern.

## 4.2 `HydrothermalProblemBuilder`/`CustomSfepyBuilder` as `@wbgeo_type`s

Converted from a plain Python class to a pydantic dataclass decorated with
`@wbgeo_type` (needed so it can flow as a typed value between Workbench component
executions — every `@wbgeo_type` gets serialized/deserialized via
`pydantic_core.to_jsonable_python`/reconstruction between executions, see
`examples/pydantic_nodesapi_simulation.py`'s mock backend). The old `__init__`'s
parameter list became declared dataclass fields (identical names/types/defaults);
the derived/computed attributes (`mesh_type`, `rock_units`,
`mat_id_to_lithology`, `fault_zone_cell_mask`, `fault_group_id`, the resolved
`t1`) are plain non-declared instance attributes assigned in `__post_init__` —
they correctly drop out of the JSON entirely (pydantic's serializer is
schema/field-driven, not `__dict__`-driven) and get recomputed by `__post_init__`
running again on reconstruction. `RockUnitProperties`/`FluidProperties` were
converted from stdlib `@dataclass` to plain pydantic `BaseModel` for the same
reason (so `Dict[str, RockUnitProperties]` round-trips through JSON) — matching
how the structural-modeling interpolator parameter types (`RBFParams`, etc.) are
plain `BaseModel`s too, not `@wbgeo_type`s themselves.

`CustomSfepyBuilder` is a `@wbgeo_type` dataclass of the same shape and for the
same reason — its `input_file_contents: str` field (not a file path or open
handle) is exactly what survives that JSON round trip; a temp file path assigned
at construction time would not (this is also why `_run_custom_sfepy_problem`
writes the content to a fresh temp file at *run* time, not construction time, same
as `HydrothermalProblemBuilder`'s own generated files). Its own derived attributes
(`mesh_type`, `mat_id_to_lithology`, `fault_zone_cell_mask`, `fault_group_id`)
follow the identical non-declared/recomputed-in-`__post_init__` pattern.

## 4.3 SmartInput sidebar for rock/fluid/fault-zone properties

`build_hydrothermal_problem`'s `options` parameter is a `SmartInput`-annotated
type (`SmartHydrothermalOptions`), giving a "Modify Parameters" sidebar for
per-rock-unit properties, fluid properties, and fault-zone properties — mirroring
`structural_workbench_components.py`'s `structural_modeling_smart_options` pattern
for per-group interpolation parameters (same shape of problem: the set of rock
unit names is only known once `geomodel_result` is connected, so the form has to
be built dynamically from it). `build_custom_sfepy_problem` has no equivalent —
by design, a custom file's own `materials` block already carries whatever
properties its author wrote directly into the SfePy syntax, so there is nothing
for a dynamic form to configure.

`enable_fault_zone` is a **plain checkbox parameter**, not an in-form toggle
(unlike the interpolation-method selector, which does use an in-form `CtrlIf`) —
this deliberately avoids relying on `CtrlIf`'s condition-path scoping against a
dict-of-groups layout (the per-rock-unit properties), which isn't exercised
anywhere else in this codebase and could not be verified end-to-end in this
development environment. If `enable_fault_zone=True` and no `options` is
connected at all, a shared module-level default (`_DEFAULT_FAULT_ZONE_PROPERTIES`)
is used, so the fault zone still works with no wiring required, matching how
`include_flow` needs none either.

**Dependency note**: this required `py_api_wbgeo>=0.8.8` — the environment this
was developed in had a stale `0.8.2` initially, which is missing
`smartcontrols.smart_group_from_type`/`form_data_as_dict`/`form_data_to_wbgeo_type_data`
(used by the *existing* structural-modeling SmartInput code too, so this was a
latent gap, not something new). It's published at `github.com/WBGeo/nodesapi` (not
PyPI); installable via
`pip install "git+https://github.com/WBGeo/nodesapi.git@<version>#subdirectory=py_api"`
(the installable package lives in a `py_api/` subdirectory, not the repo root).

## 4.4 Blocked-path pre-checks (`input_checks`)

All follow the `@wbgeo_component(input_checks=[...])` pattern from
docs/developers/components.md — blocking a Workbench connection that runs without
error but was never actually intended, rather than leaving it as a silent trap.
Every one of these is wired as `input_checks=` **and** called explicitly inside the
component function body (pre-checks can be skipped by a direct Python caller).

1. **Structured meshing rejects faulted structural models** — section 3.3.
2. **`build_hydrothermal_problem` rejects a mesh with no known `mesh_type` or no
   lithology mapping** (`cell_data["block_id"]` missing — e.g. an unstructured
   mesh built with `mapping_litho="manual"`/`"none"`). Both were already hard
   errors deep inside `HydrothermalProblemBuilder.__post_init__`/
   `_map_mat_id_to_lithology`; extracted into standalone
   `check_mesh_has_known_type`/`check_mesh_has_lithology_mapping` functions so the
   same validation is reused as a proper Workbench pre-check instead of only
   surfacing as a runtime construction error.
3. **`build_custom_sfepy_problem` reuses both checks above, plus a third
   (new)**: `check_mesh_has_no_engineering_objects` — section 2.6. Note this third
   check is *not* applied to `build_hydrothermal_problem` — that path's existing,
   unchanged behavior is to silently leave engineering-object blocks unsolved
   rather than reject them, and this pass didn't change that (see 2.6 for why the
   two paths differ here).

   A related but distinct validation, `check_custom_sfepy_regions` (the
   `'cells of group N'` sanity check, section 2.1), is **not** wired as a Workbench
   `input_checks=` pre-check — it runs unconditionally inside
   `CustomSfepyBuilder.__post_init__` instead, since it needs the *decoded file
   content* (only available after `build_custom_sfepy_problem` has already read
   the uploaded stream), not just the typed inputs a pre-check receives before the
   component body runs.

**Proposed, not implemented**: a redundant check specifically for "structured mesh
+ faulted `geomodel_result`" at the `build_hydrothermal_problem`/
`build_custom_sfepy_problem` level too, since `mesh_results`/`geomodel_result` are
two independent inputs there with nothing verifying they correspond to each
other — see section 2.5. Deferred as a fringe case for now.

## 4.5 Test coverage for all of the above

`tests/tests_simulation_components/test_simulation_workbench_components.py` covers
`build_hydrothermal_problem`'s options-application logic, the SmartInput form
build/parse round trip, `build_custom_sfepy_problem`'s file-decoding and
pre-checks, and the thin wrappers via `monkeypatch` (no real solve needed for
those). `tests/tests_simulation_components/test_custom_sfepy_builder.py` (12
tests) covers `CustomSfepyBuilder` construction/validation directly: basic
construction, optional `geomodel_result`, each rejection path (unknown mesh type,
no lithology mapping, unknown referenced group, well/source block present), the
JSON round-trip, fault-zone-off-by-default and its `geomodel_result`-required
guard, fault-zone detection on a real faulted model, and a custom file
successfully referencing the fault group. `test_hydrothermal_run.py` adds the
dispatcher test (`run_simulation_sfepy` routes to the right runner by type),
fault-zone-args passthrough, and two real end-to-end `sfepy-run` invocations (plain
custom file, and custom file with an active fault zone).
`tests/tests_meshing_components/explicit/structured_mesh/test_structured_mesh.py`,
`.../unstructured_mesh/test_lithology_mapping_mode.py`, and
`.../unstructured_mesh/test_creare_surface_grid.py` (the last one covering
`_evaluate_rbf_chunked`, section 3.6) cover the meshing-side pieces from section 3.

Two incidental test-infrastructure issues surfaced and are worth knowing about:
- `tests/test_component_signatures.py` had a real import-order fragility (fixed):
  it never re-imported `py_api_wbgeo.smartcontrols` under its own mock backend
  instance, so whichever test happened to import it first (this pipeline's new
  test file, since nothing previously imported a `*_workbench_components.py`-style
  SmartInput module directly in a test) would permanently decorate `CtrlGroup`
  without the mock, breaking every SmartInput component's return-type check for a
  reason unrelated to that component. Fixed by adding
  `py_api_wbgeo.smartcontrols` to the same evict/reimport treatment already used
  for `core/` files.
- A GMSH physical-group-tag state bleed across separate
  `create_unstructured_mesh_data` calls within one pytest process (same family of
  issue as section 3.5) — not fixed, worked around by avoiding a second
  independent real-meshing call per test file rather than chasing the GMSH
  internals.
