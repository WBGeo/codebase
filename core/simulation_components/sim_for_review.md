  # Process Simulation — implementation notes for reviewers

This directory replaces the previous `core/simulation_components` implementation
entirely, and is now wired into the Workbench. This document is structured for two
different reviewers:

1. Whoever reviews the simulation implementation itself (start at "1. Implementation").
2. Whoever owns meshing (`core/meshing_components/`) and the Workbench/DSL layer
   (`py_api_wbgeo`) — sections 3 and 4 are written specifically for you, so you can
   review just the parts that touch your area without reading the whole thing.

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

Four files:

- **`simulation_packages/sfepy/sfepy_hydrothermal_builder.py`** —
  `HydrothermalProblemBuilder`. Takes a `MeshResults`, `StructuralModelResults`, and
  physical properties (`RockUnitProperties` per lithology, `FluidProperties`), and
  generates the actual SfePy problem-description `.py` files (regions, materials,
  equations, solvers) as strings written to disk. This is the input-data-generator
  component — its constructor arguments are the component's configurable inputs,
  matching how other Workbench components work. It is itself a `@wbgeo_type`
  (a pydantic dataclass) — see section 4.2.
- **`simulation_packages/sfepy/sfepy_hydrothermal_run.py`** — `run_simulation_sfepy(builder)`,
  the run component. Takes a builder, converts its mesh to the format SfePy actually
  reads (Exodus → Medit, via meshio), launches `sfepy-run` as a subprocess, loads
  the result back into a `SimulationResults` object. Also
  `export_simulation_results()`, a "Download Simulation Results" component (VTK
  only so far).
- **`simulation_visualization/simulation_visualization.py`** — plotting functions for
  `SimulationResults` (3D snapshots, cross-sections, before/after diffs) and a
  pre-flight material-assignment check for `HydrothermalProblemBuilder`
  (`plot_builder_materials`) that lets a wrong material/fault-zone assignment be
  caught by eye before running a solve that can take a while. Three plotting
  utilities from the previous implementation (`plot_variable_along_line`,
  `print_variable_at_point`, `plot_variable_time_series`) were dropped rather than
  reimplemented — not currently used anywhere, but flagging in case they're
  actually wanted back.
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
well-conditioned single-physics problem on its own once segregated.

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

## 1.4 Fault support (limited, opt-in)

`fault_zone_properties` (a `RockUnitProperties`) gives cells near any fault their
own separate material — a "damage zone" band `fault_zone_n_voxels` grid cells wide,
computed on the structural model's grid via each fault's `get_domain_mask()`, then
resampled onto mesh cells. Off by default; does not change behavior when unset.

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

SfePy itself has no fault/interface element for flow problems — checked the
`terms/` and `examples/` directories directly; the only contact-mechanics module
that exists is for elastic solid-body mechanical contact, not applicable here. The
only real lever is per-region material constants, the same mechanism already used
for lithology.

## 1.5 Lithology-to-mesh mapping — trusted from meshing, not re-derived

`HydrothermalProblemBuilder` needs to know which lithology each mesh block
corresponds to, and now trusts `MeshResults.cell_data["block_id"]` directly for all
three mesh types, with one explicit `ValueError` at construction time if it's
missing entirely. This used to require an independent cKDTree-based fallback for
unstructured meshes specifically, because meshing computed the mapping internally
but never wrote it to the output — that gap is now fixed at the source; see
section 3.2 for the meshing-side details.

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
  `cond != 0`.

## 1.7 Verification approach

**Automated test suite**: `tests/tests_simulation_components/` (61 tests). Coverage
(via `coverage run --source=core.simulation_components -m pytest
tests/tests_simulation_components/`): `sfepy_hydrothermal_builder.py` 99%,
`sfepy_hydrothermal_run.py` 91%, `simulation_workbench_components.py` 96%,
`simulation_visualization.py` 50% (the rest is `plot_*` rendering bodies,
deliberately not unit-tested — matches this codebase's existing convention of not
asserting on plot output). Covers: construction validation, rock-unit/fault-zone
logic (including the exact 764/596 fault-cell-count regression from section 1.4),
generated SfePy input-file text (validated via `compile()`, not a real solve, where
possible), the JSON round-trip every `@wbgeo_type` gets between Workbench component
executions, and several real end-to-end `sfepy-run` invocations (plain, pure-
conduction-must-be-exactly-static, and fault+flow). Full repository suite: 534
passed, 1 pre-existing unrelated skip, 0 failed.

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

## 2.1 Custom SfePy input file support — researched, not implemented

A user cannot currently upload their own raw SfePy input file instead of using
`HydrothermalProblemBuilder` — only the built-in two-stage Darcy+heat physics is
supported. This was researched in depth and explicitly deferred, not overlooked:

- `_run_sfepy_input_file()` is already close to generic — it takes a file *path*,
  doesn't inspect content, and does the mesh conversion independent of what's in
  the file. `load_vtk_results()` is fully generic too. What's *not* generic is
  `run_simulation_sfepy()`, which hardcodes the two-stage pressure→heat sequence
  with the Darcy-velocity handoff in between.
- SfePy itself has two mechanisms that would fit a "bring your own file" flow: the
  `-d "key: value"` / `--define` CLI flag (calls a `define(**kwargs)` function in
  the input file — SfePy's own idiomatic parameterization mechanism), and
  `--solve-not --save-regions-as-groups` (builds the real SfePy `Problem` object,
  resolving every region, without actually solving — a genuine pre-flight
  validation using SfePy's own region-resolution code).
- The natural design is a shared `Stage`/`get_stages()` interface that both
  `HydrothermalProblemBuilder` and a hypothetical `CustomSfepyInput` class
  implement (each stage knows how to produce its own input file and optionally
  what to hand to the next stage), so `run_simulation_sfepy()` becomes one loop
  over `.get_stages()` with zero knowledge of which kind of "problem" produced
  them. A custom upload becomes a single trivial stage (hands back the given
  file, no handoff). This would also make `--solve-not` pre-flight validation
  available uniformly, for builder-generated files too, not just custom ones.
- Important scope note: a custom upload is *not* how someone would replicate what
  `HydrothermalProblemBuilder` already does — they would just use the builder
  directly. The custom path only ever covers a single SfePy file / single physics
  problem with no inter-stage dependency; the builder's own two-stage
  pressure→heat pattern (stage 2 needs the Darcy velocity from stage 1) cannot be
  expressed by one uploaded file at all. Replicating that pattern with genuinely
  custom physics would mean writing a small Python class against the
  `get_stages()` interface once it exists — a developer task, not a Workbench
  upload.
- Not implemented because it is a real restructuring (the stage-sequencing logic
  needs to move from the runner into the builder) with no immediate use case
  driving it. Start from the `Stage`/`get_stages()` refactor whenever this is
  picked up.

## 2.2 Two-stage physics only / partial fault support

The pressure→heat segregation with a Darcy-velocity handoff between stages is
specific to this builder (see 2.1 for the custom-physics story). Fault support is
intentionally partial (see 1.4) — one shared material for all faults, no exact
hanging-wall/footwall geometry.

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
sanity-checking approach described in section 1.7.

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
constructed through the normal Workbench path. A residual gap: `mesh_results` and
`geomodel_result` are still two independent inputs to
`HydrothermalProblemBuilder`/`build_hydrothermal_problem`, so a structured mesh
built before this check existed (or from any other source) could still be paired
with a faulted `geomodel_result` at that stage, recreating the problem. A second,
redundant check there was proposed and deliberately not implemented (judged a
fringe case for now) — see section 4.4.

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
of them for no benefit. Full `tests/tests_meshing_components/` suite (240 tests,
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

## 3.4 `Exo_format.py`: new `VOLUME_ONLY` export type + a residual known risk

Added `Exo_format.MeshType.VOLUME_ONLY = "volume_only"` — purely additive, behaves
exactly like `STRUCTURED`/`IMPLICIT` in `write()` (`volume_blocks = all_blocks`, no
slicing). Needed because `_run_sfepy_input_file()` pre-filters an unstructured
mesh down to its tetra-only volume blocks before export (see 3.5), and needed a
way to say "this mesh is already volume-only" without mislabeling its true
`mesh_type` as `"imp"` just to get that code path. Doesn't touch or reorder any
existing enum values, so no risk to existing tests depending on
`"imp"`/`"str"`/`"unstr"`.

**Residual risk, not fixed**: `Exo_format.py`'s unstructured export path strips a
hardcoded `NUM_SIDE_BLOCKS = 6` trailing blocks to get the real volume mesh —
correct only when there are exactly 6 non-volume blocks total. A model with
multiple faults, wells, or sources can easily exceed 6, silently corrupting the
exported mesh for any caller relying on that path with a non-standard block count.
This branch's own use of unstructured export works around it (see 3.5) but does
not fix `Exo_format.py` itself.

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

---

# 4. Workbench wiring (`py_api_wbgeo` / DSL layer)

This section is for whoever reviews the Workbench/DSL side specifically.

## 4.1 New components

`core/simulation_components/simulation_workbench_components.py` (new file — the
old `simulation_workbench_components.py`-equivalent was deleted along with the
previous implementation) holds every `@wbgeo_component` for this pipeline, per the
project's "one workbench file per step" convention — thin wrappers only, no logic
of its own:

- **Build Hydrothermal Problem** (`build_hydrothermal_problem`) — takes
  `mesh_results`, `geomodel_result`, the SmartInput `options` (4.3), and plain
  scalar settings (`include_flow`, `enable_fault_zone`, `fault_zone_n_voxels`,
  `t0`/`t1`/`num_steps`, boundary values, solver settings), returns a
  `HydrothermalProblemBuilder`.
- **Run Hydrothermal Simulation** (`run_hydrothermal_simulation`) — thin wrapper
  over `run_simulation_sfepy`.
- **Export Simulation Results** (`export_simulation_results`) — thin wrapper over
  the run-file function of the same name.
- Two inspectors: **Plot Variable (Final Time)** and **Plot Cross Section 2D**,
  attached to `SimulationResults`, matching the existing
  `StructuralModelResults`/`MeshResults` inspector pattern.

## 4.2 `HydrothermalProblemBuilder` as a `@wbgeo_type`

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

## 4.3 SmartInput sidebar for rock/fluid/fault-zone properties

`build_hydrothermal_problem`'s `options` parameter is a `SmartInput`-annotated
type (`SmartHydrothermalOptions`), giving a "Modify Parameters" sidebar for
per-rock-unit properties, fluid properties, and fault-zone properties — mirroring
`structural_workbench_components.py`'s `structural_modeling_smart_options` pattern
for per-group interpolation parameters (same shape of problem: the set of rock
unit names is only known once `geomodel_result` is connected, so the form has to
be built dynamically from it).

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

## 4.4 Two new blocked-path pre-checks (`input_checks`)

Both follow the `@wbgeo_component(input_checks=[...])` pattern from
docs/developers/components.md — blocking a Workbench connection that runs without
error but was never actually intended, rather than leaving it as a silent trap.
Both are wired as `input_checks=` **and** called explicitly inside the component
function body (pre-checks can be skipped by a direct Python caller).

1. **Structured meshing rejects faulted structural models** — section 3.3.
2. **`build_hydrothermal_problem` rejects a mesh with no known `mesh_type` or no
   lithology mapping** (`cell_data["block_id"]` missing — e.g. an unstructured
   mesh built with `mapping_litho="manual"`/`"none"`). Both were already hard
   errors deep inside `HydrothermalProblemBuilder.__post_init__`/
   `_map_mat_id_to_lithology`; extracted into standalone
   `check_mesh_has_known_type`/`check_mesh_has_lithology_mapping` functions so the
   same validation is reused as a proper Workbench pre-check instead of only
   surfacing as a runtime construction error.

**Proposed, not implemented**: a redundant check specifically for "structured mesh
+ faulted `geomodel_result`" at the `build_hydrothermal_problem` level too, since
`mesh_results`/`geomodel_result` are two independent inputs there with nothing
verifying they correspond to each other — see section 2.5. Deferred as a fringe
case for now.

## 4.5 Test coverage for all of the above

`tests/tests_simulation_components/test_simulation_workbench_components.py` covers
`build_hydrothermal_problem`'s options-application logic, the SmartInput form
build/parse round trip, the two `input_checks` pre-checks (at the underlying
function level — `@wbgeo_component`'s `input_checks=` wiring itself can't be
introspected outside a real Workbench backend instance, since the decorator is a
no-op when none is registered), and the thin wrappers via `monkeypatch` (no real
solve needed for those). `tests/tests_meshing_components/explicit/structured_mesh/test_structured_mesh.py`
and the new `.../unstructured_mesh/test_lithology_mapping_mode.py` cover the
meshing-side pieces from section 3.

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
