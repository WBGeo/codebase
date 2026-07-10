# Process Simulation — implementation notes for reviewers

This directory replaces the previous `core/simulation_components` implementation
entirely. This document explains what changed, why, how it was verified, and what's
still missing — written for whoever reviews this branch.

## Why a rewrite instead of a patch

The previous implementation had three concrete, verified correctness bugs (see
"Bugs found and fixed" below) and a hand-written, static SfePy input file
(`Hydro_thermal.py` / `Thermal.py`) that every example copied and edited by hand
whenever rock properties or boundary conditions needed to change. Rather than patch
around those issues, the input-file generation, execution, and result handling were
rebuilt from scratch, matching the Workbench's component/connector philosophy more
directly: a builder component that turns typed inputs (mesh, structural model, rock
properties) into SfePy problem files, and a separate run component that executes
them and returns a typed result.

## Architecture

Three files:

- **`simulation_packages/sfepy/sfepy_hydrothermal_builder.py`** —
  `HydrothermalProblemBuilder`. Takes a `MeshResults`, `StructuralModelResults`, and
  physical properties (`RockUnitProperties` per lithology, `FluidProperties`), and
  generates
  the actual SfePy problem-description `.py` files (regions, materials, equations,
  solvers) as strings written to disk. This is the input-data-generator component —
  its constructor arguments are the component's configurable inputs, matching how
  other Workbench components work.
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
  caught by eye before running a solve that can take a while.

## `MeshResults.mesh_type` — mesh provenance moved onto the base model

`HydrothermalProblemBuilder` used to require a separate `mesh_type` constructor
argument ("implicit"/"structured"/"unstructured") alongside `mesh_results`, purely
because `MeshResults` (`core/object_components.py`) didn't record which meshing
component produced it — a real redundancy: two places that had to agree, with
nothing checking they actually did. Fixed at the source instead of worked around
downstream: `MeshResults` now has a `mesh_type: Optional[MeshType] = None` field,
`MeshType` a new `StrEnum` (`IMPLICIT`/`STRUCTURED`/`UNSTRUCTURED`) defined right
next to `MeshResults`. `create_implicit_structured_mesh`,
`create_structured_mesh_data`, and `create_unstructured_mesh_data` (the only three
ways a `MeshResults` gets built for real) all set it automatically now.
`HydrothermalProblemBuilder` no longer takes `mesh_type` at all — it reads
`mesh_results.mesh_type`, raising a clear `ValueError` if it's `None` (e.g. a
manually-constructed `MeshResults` in a test that never set it, rather than a
confusing failure several calls downstream).

`Optional` rather than required: ~13 existing meshing-format tests
(`tests/tests_meshing_components/mesh_formats/`) construct bare `MeshResults(nodes=...,
elements=...)` directly to test export logic in isolation, with no need for
provenance — making the field required would have forced pointless changes to all
of them for no benefit. Verified this doesn't regress anything: full
`tests/tests_meshing_components/` suite (234 tests) still passes unchanged.

`export_mesh_results_to_exodus`'s own `type` parameter (`core/meshing_components/
mesh_format/exodus/Exo_format.py`) keeps its own separate `MeshType` short-code
enum (`"imp"`/`"str"`/`"unstr"`, not the same class as the new one) rather than
being unified with `MeshResults.mesh_type` — deliberately: the two represent
different things. `MeshResults.mesh_type` is factual provenance (which meshing
component actually produced this mesh); `Exo_format`'s `type` is a
configuration knob selecting which export code path to run, and the two can
legitimately diverge (see "GMSH + Windows `CreateProcess` conflict" below,
where an unstructured-origin mesh is deliberately exported via a different
code path after being pre-filtered). Existing `Exo_format.MeshType` values are
also depended on explicitly by several existing tests, so left unchanged;
one new value, `MeshType.VOLUME_ONLY`, was added there (purely additive, same
behavior as `STRUCTURED`/`IMPLICIT`) so that divergence has an honestly-named
option instead of reusing `IMPLICIT` to mean something it doesn't.
`HydrothermalProblemBuilder`'s `MESH_TYPE_CODES` dict still does the one
remaining readable-name-to-short-code translation at the point it calls that
function.

## Physics model

Coupled single-phase Darcy flow + advective-conductive heat transport, solved as
**two sequential single-physics stages** (pressure first, then temperature), not one
combined system. An earlier monolithic version (both fields in one equations dict)
was badly conditioned for iterative solvers — pressure (~1e6 Pa) and temperature
(~10s of °C) in the same matrix — which no amount of preconditioner tuning fixed
(verified: relative residual stuck near 1.0 with ILU + diagonal scaling regardless
of settings). Segregating matches the physics anyway: temperature depends on the
flow field, but pressure never depends on temperature in this model. Each stage is
a well-conditioned single-physics problem on its own once segregated.

`t1` (the transient stage's end time) defaults to a data-driven thermal diffusion
timescale (`L² / α`) rather than an arbitrary constant. The original hand-written
example used a fixed 700 seconds, which is negligible next to this model's actual
diffusion timescale (order of years) — the heat equation's transient term then
dwarfed the diffusive term by 5-6 orders of magnitude in the assembled matrix, an
internal scale mismatch that stalled the iterative solver at relative residual
~0.79 regardless of preconditioning. Using the diffusion-timescale default instead
converged to ~1e-13 with identical solver settings.

`include_flow=False` skips the pressure/Darcy stage entirely, for pure-conduction
validation against a known analytical steady-state profile before trusting the
coupled result.

## Fault support (limited, opt-in)

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
(`effective_domain_components_for_group` in `general.py`). This was a real bug in
an earlier version of this feature — verified fixed: on model2, restricting a fault
to only its older stratigraphic group correctly excludes the younger group's cells
from the fault zone (764 → 596 affected cells at test resolution), confirmed both
numerically and visually via `plot_builder_materials`.

SfePy itself has no fault/interface element for flow problems — checked the
`terms/` and `examples/` directories directly; the only contact-mechanics module
that exists is for elastic solid-body mechanical contact, not applicable here. The
only real lever is per-region material constants, the same mechanism already used
for lithology.

## Lithology-to-mesh mapping — now trusted from meshing, not re-derived

`HydrothermalProblemBuilder` needs to know which lithology each mesh block
corresponds to. Implicit and structured meshing already tagged this reliably via
`MeshResults.cell_data["block_id"]`. Unstructured meshing did not — it computed a
lithology-per-block mapping internally but never wrote it to the output, so this
builder used to carry its own independent cKDTree-based fallback
(`_resample_lithology_per_block`, sampling individual cell centroids and taking a
majority vote against the structural model's `lith_block`).

That fallback is gone. Instead, `core/meshing_components/explicit/unstructured/mesh_data.py`
now propagates `cell_data["block_id"]` for its `AUTO`/`AUTOMATIC_DEV` lithology-mapping
modes (a real mesh-level gap, fixed at the source rather than worked around
downstream — see that file's own docstrings/comments for the mechanism). This
builder now trusts `cell_data["block_id"]` for all three mesh types, with one
explicit `ValueError` at construction time if it's missing entirely (e.g.
unstructured mesh built with `mapping_litho="manual"`/`"none"`, which don't
produce one). A `-1` sentinel (non-volume blocks: fault surface, well, source,
boundary/"extended" surfaces) used to also raise, but is now handled instead —
see "GMSH + Windows `CreateProcess` conflict" below for that fix.

Also added, in the same meshing file: `mapping_litho="automatic_dev"`, an
opt-in alternative to the existing `"auto"` mode. `"auto"` votes each raw mesh
block's lithology using its *node* coordinates against the structural grid; nodes
sit on block boundaries, exactly where a fault plane tends to be, so the vote can
be genuinely ambiguous there regardless of algorithm quality. `"automatic_dev"`
votes on cell centroids instead (guaranteed-interior points). Verified on model2 at
a deliberately mismatched structural-grid/mesh resolution: 73.3% cell-level
accuracy against ground truth vs. 68.1% for `"auto"` (~16% relative reduction in
misclassified cells) — not a complete fix (neither mode is reliable when the
structural grid is far coarser than the mesh; that's a real, separate problem no
purely-geometric voting method solves), but a measurable improvement.
`"automatic_dev"` is not the default and isn't used by any example — worth a
conversation with whoever owns that file before it becomes one.

## GMSH + Windows `CreateProcess` conflict — root-caused and worked around

Every example in this codebase uses `mesh_type="implicit"`. `"structured"` also
works but isn't exercised. `"unstructured"` used to fail, not during mesh
generation — the mesh generates cleanly — but at the very next step, launching
`sfepy-run` as a subprocess, which failed with `FileNotFoundError` at
`_winapi.CreateProcess`:

```
File "...\Python312\Lib\subprocess.py", line 1538, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

Root-caused to the minimum possible reproduction: `import gmsh;
gmsh.initialize(); gmsh.finalize()` alone — no mesh, no structural model, nothing
else — already breaks every subsequent `subprocess.run`/`Popen` call **that passes
a bare executable name** (e.g. `"sfepy-run"`) in that Python process.
`gmsh.initialize()`/`finalize()` are confirmed properly paired in a try/finally in
`create_unstructured_mesh_data`, so it isn't simply "GMSH left initialized."

Narrowed further: `PATH` itself is untouched by GMSH (verified identical before
and after), and `shutil.which("sfepy-run")` still resolves it correctly
afterward — so it's specifically Win32 `CreateProcess`'s own executable-search
step (used only when the executable is passed as a bare name) that GMSH's
`initialize()` breaks at the OS/DLL level, not anything Python-visible like
`PATH` or the filesystem. **Fixed**: `_run_sfepy_input_file` now resolves
`"sfepy-run"` to its absolute path via `shutil.which()` before calling
`subprocess.run`, sidestepping `CreateProcess`'s own search entirely. Verified by
generating a real unstructured mesh (which runs GMSH `initialize()`/`finalize()`)
and then running a normal implicit-mesh simulation through the real
`run_simulation_sfepy` in the same process — succeeds where it previously raised
`FileNotFoundError`.

This fixes `sfepy-run`'s subprocess call specifically. It does **not** fix the
underlying GMSH/Windows `CreateProcess` breakage itself — any *other* code in the
same process that launches a subprocess by bare executable name after GMSH has
run would still fail. If `py_runner` is a long-lived process serving multiple
requests (as the installation docs' mention of restarting it to pick up code
changes suggests, though this wasn't confirmed from this codebase — `py_runner`'s
actual implementation lives in a separately-built image, not here), one user
triggering unstructured meshing could still break bare-name subprocess calls
elsewhere in that same process for every subsequent request, until restarted.
Worth confirming with whoever owns that infrastructure — the same
resolve-to-absolute-path fix applies anywhere else this pattern shows up. A more
thorough (but unimplemented) fix would be to have `create_unstructured_mesh_data`
run GMSH in a child process (matching the temp-file/subprocess handoff pattern
this code already uses for the SfePy call itself), so the parent process's
`CreateProcess` behavior is never affected in the first place — this would be
entirely internal to that one component, invisible to everything downstream.

**`mesh_type="unstructured"` now works end-to-end**, including with an active
fault zone. Getting there needed a second, independent fix on top of the
subprocess one above — `HydrothermalProblemBuilder` used to raise at
construction if any mesh block had `block_id == -1` (a non-volume block: fault
surface, well, source, or a boundary/"extended" surface block, none of which
are real lithology volumes — every unstructured mesh has several). Checked in
the actual mesh: these are always `triangle` (2D) blocks trailing the real
`tetra` (3D) volume blocks, never mixed in among them (meshing always builds
the lithology volumes first). **Fixed**, two parts:

1. `_map_mat_id_to_lithology` (and every other place that used to assume
   "every raw block index has a lithology" — `_region_lines`,
   `build_pressure_input_file`, `build_heat_input_file`,
   `compute_darcy_velocity`, `compute_cell_materials`) now skips `block_id ==
   -1` blocks instead of raising, iterating `mat_id_to_lithology`'s keys
   rather than `range(len(mesh_results.elements))`.
2. A second, non-obvious problem surfaced once a *fault* was involved
   specifically: `Exo_format.py`'s Exodus writer, for `mesh_type="unstr"`,
   strips a hardcoded `NUM_SIDE_BLOCKS = 6` trailing blocks to get the real
   volume mesh — correct only when there are exactly 6 non-volume blocks
   total. With a fault present there are 7+ (one extra triangle block per
   fault), so that fixed-size slice wrongly left a fault-surface triangle
   block mixed in with real tetrahedra, which SfePy then failed on
   (`ValueError: Size of label 'j' for operand 1 (4) does not match previous
   terms (3)` — a 3-node/4-node element mismatch inside one region, during
   the heat solve). Fixed in `_run_sfepy_input_file`: for
   `mesh_type_code="unstr"`, it now filters `mesh_results.elements` down to
   `dim == 3` blocks itself before export, and passes
   `type=MeshType.VOLUME_ONLY` to `export_mesh_results_to_exodus` instead of
   `"unstr"` for that one call — a new `Exo_format.MeshType` value added
   specifically for this (behaves exactly like `"imp"`/`"str"` there:
   `volume_blocks = all_blocks`, no slicing), rather than reusing `"imp"` and
   mislabeling an unstructured-origin mesh as implicit just to get that code
   path — `MeshResults.mesh_type` (the mesh's real, separate provenance
   field, see below) correctly still reads `"unstructured"` throughout; only
   the Exodus exporter's own internal code-path selector changes for this
   one call. Boundary conditions are unaffected by this switch:
   `Gamma_Top`/`Gamma_Bottom` are built from `mesh_results.point_sets`
   directly in `_region_lines()`, never from Exodus side-sets.

**Verified end-to-end on model2** (1 fault): builder construction,
`compute_cell_materials()`, and a full `run_simulation_sfepy()` solve all
succeed, under `include_flow=False` and `include_flow=True` (exercising
`compute_darcy_velocity`'s fault-zone cell-splitting), and with the fault zone
inactive too (regression). Also regression-verified on model1's implicit mesh
(unaffected code path) and unstructured mesh without a fault.

**Residual risk, not fixed**: `Exo_format.py`'s `NUM_SIDE_BLOCKS = 6` constant
is still hardcoded and still wrong in general (e.g. multiple faults, wells, or
sources push the real non-volume count past 6) — this fix works around it
specifically for the path this builder uses (by pre-filtering and switching
export type before Exodus ever sees the extra blocks), but doesn't fix
`Exo_format.py` itself for any other caller relying on its unstructured
export path with a similarly non-standard block count. Worth flagging to
whoever owns that file.

## Bugs found and fixed vs. the previous implementation

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
  iterations. A silently non-converged "solution" (SfePy exits 0 either way) used
  to look identical to a real one. Fixed: `'report_status': True` was added to
  the shared Newton solver config
  (`HydrothermalProblemBuilder._newton_ts_options_blocks`, used by both the
  pressure and heat stages), which makes SfePy print a definitive
  `cond: N, iter: ..., err0: ..., err: ...` line per nonlinear solve (`N`: 0 =
  converged, 1 = max iterations reached, 2 = linesearch gave up — SfePy's own
  codes, from `conv_test()` in `sfepy/solvers/nls.py`). `_run_sfepy_input_file`
  now scans stdout for every such line (a transient run does one nonlinear solve
  per time step, so an earlier step failing while a later one happens to succeed
  must not go unnoticed) and raises with the specific failing line(s) if any
  `cond != 0`.

## Verification approach

No dedicated automated test suite yet for this pipeline (see Known gaps). Verified
throughout development by: running each example end-to-end and checking the
result is physically sensible (temperature gradients relaxing toward expected
steady states, boundary conditions matching configured values); a specific
correctness check for the fault-zone feature (setting `fault_zone_properties`
identical to the surrounding homogeneous rock properties and confirming the result
is unchanged to floating-point/solver-tolerance noise — isolates the region-
splitting and Darcy-velocity-handoff plumbing from whether the physics itself is
right); and the lithology-mapping accuracy comparison described above (ground-truth
cross-check via independent nearest-neighbor lookup, not just "did it run without
error").

The convergence check itself (`_check_convergence` in `sfepy_hydrothermal_run.py`)
was verified two ways: synthetic stdout strings covering a converged case, a
single non-converged case, and a mixed multi-step case where only one of several
nonlinear solves fails (confirms it scans every match, not just the last); and
against a real reproduced failure, not just a synthetic one — re-running model1
with `include_flow=True` and the historically-bad `t1=700` (see "Physics model"
above) genuinely diverges on the second time step (residual growing from 0.008 to
0.065 over 5 Newton iterations, `cond: 1`) and the check correctly raises with
that exact line. Note this reproduction needed `include_flow=True`: with
`include_flow=False`, the heat stage's initial condition is already the exact
pure-conduction steady state (see `ic_temp`'s docstring in the builder), so
there's nothing to diffuse and the solve trivially converges in one Newton
iteration regardless of `t1` — the scale-mismatch problem only shows up once
advection perturbs the field away from that steady state.

## Known gaps

- **`mesh_type="structured"` is not recommended for faulted models.** Cross-checked
  all three mesh types against each other on model2 (1 fault) at its real settings:
  `implicit` and `unstructured` agree well (domain-mean T within ~1%, identical
  min/max, close std), but `structured` genuinely fails to converge (`cond: 1`,
  residual dropping from ~1569 to ~8.86 over 5 Newton iterations but never reaching
  tolerance — the convergence check catching a real numerical issue, not a false
  positive). Confirmed it's specific to the fault+structured-mesh combination, not
  structured meshing generally: the identical builder settings on model1 (same
  physics, no fault) converge cleanly on a structured mesh. Not yet root-caused
  (suspected: structured mesh's cell geometry/connectivity interacting badly with
  the fault-zone region split) or fixed — implicit/unstructured are unaffected and
  recommended for faulted models until this is investigated further.
- **The convergence check only catches SfePy self-reporting non-convergence**
  (`cond` 1 or 2). It cannot catch a solve that satisfies Newton's tolerance
  trivially against a physically-wrong state — e.g. a badly-scaled system where
  the initial guess already nearly zeroes the residual for the wrong reason (this
  is what happened in the `include_flow=False` reproduction above: no crash, no
  `cond` failure, just an uninteresting trivial solve). Judging physical
  correctness, not just numerical convergence, still requires the
  sanity-checking approach described in "Verification approach" above.
- **No automated test suite.** The previous implementation's tests
  (`tests/tests_simulation_components/`) tested the removed `run_sfepy`/
  `load_exodus_results` functions directly and were removed rather than adapted,
  since the functions they tested no longer exist. Writing real coverage for this
  pipeline (`HydrothermalProblemBuilder`, `run_simulation_sfepy`, the fault-zone
  region-splitting logic) is real, not-yet-done follow-up work.
- **No Exodus export** for `SimulationResults` (`export_simulation_results` only
  supports `format="vtk"`).
- **Two-stage physics only.** The pressure→heat segregation with a Darcy-velocity
  handoff between stages is specific to this builder. A user wanting genuinely
  different or custom physics currently has no supported path other than editing
  this builder directly — a "bring your own SfePy input file" component was
  researched (see `project_sfepy_hydrothermal_builder` memory from
  development) but not built; it would need a shared `Stage`/stage-sequencing
  interface both this builder and a custom-file wrapper implement, so
  `run_simulation_sfepy` can stay a single component regardless of which produced
  the input.
- **`Exo_format.py`'s `NUM_SIDE_BLOCKS = 6`** (unstructured Exodus export) is
  still hardcoded and still wrong in general beyond what this builder now
  works around — see "GMSH + Windows `CreateProcess` conflict" above.
- **Fault support** is intentionally partial (see above) — one shared material
  for all faults, no exact hanging-wall/footwall geometry.
