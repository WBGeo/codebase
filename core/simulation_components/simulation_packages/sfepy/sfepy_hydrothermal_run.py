"""
Runs the SfePy input files HydrothermalProblemBuilder generates
(sfepy_hydrothermal_builder.py) and loads the results back into a
SimulationResults object -- split out from the builder to keep "build the
input data" and "run the simulation" as separate concerns, matching the
Workbench's component/connector model: HydrothermalProblemBuilder is the
input-data-generator component, run_simulation_sfepy() here is the run
component that takes it as input.

Depends on export_mesh_results_to_exodus (mesh export/format-conversion
logic, owned by meshing, not simulation).
"""
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Optional, Union

import meshio
import numpy as np

from core.object_components import MeshResults, SimulationResults
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results, SfepyOutputType
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder,
    CustomSfepyBuilder,
    MESH_TYPE_CODES,
    is_custom_sfepy_builder,
)
from core.simulation_components.simulation_visualization.simulation_visualization import build_grid_from_class
from py_api_wbgeo.nodesapi import BasicallyABufferedFile

logger = logging.getLogger(__name__)


#: SfePy's own nonlinear-solver convergence codes (see conv_test() in
#: sfepy/solvers/nls.py's docstring): 0 = converged (tolerances met), any
#: other value = not converged. Every backend agrees on 0 meaning
#: converged (nls.newton's 1/2 vs. nls.petsc's -1 on failure -- see the
#: same nls.py: `condition = 0 if converged else -1`), but not on what a
#: given non-zero code specifically means. CustomSfepyBuilder lets a user
#: configure any SfePy nonlinear solver, so guessing a specific reason per
#: code (e.g. "linesearch gave up" for 2) risks attaching a wrong,
#: solver-specific explanation to a code some other, unanticipated solver
#: backend reuses for something else -- so only pass/fail plus the raw
#: code is reported here; the full stdout below carries whatever
#: solver-specific detail actually exists.
_SFEPY_CONVERGED_CONDITION = 0


def _check_convergence(stdout: str, input_file: str) -> None:
    """
    Raise if any nonlinear solve reported in `stdout` didn't converge.

    Requires 'report_status': True on the generated input file's 'newton'
    solver config (HydrothermalProblemBuilder._newton_ts_options_blocks) --
    without it, SfePy only prints per-iteration residuals, with no
    definitive final "did this converge" signal to check. A transient
    solve with num_steps > 1 runs one nonlinear solve per time step, each
    reporting its own "cond: N, iter: ..." line -- check every one, not
    just the last, since an earlier step failing while a later one
    happens to succeed would otherwise go unnoticed.

    This only catches genuine non-convergence (SfePy itself reporting it
    exhausted i_max or gave up) -- a crashed/errored run is already caught
    separately by _run_sfepy_input_file's returncode check above this.

    The full stdout is already logged unconditionally by the caller, line
    by line as SfePy runs (logger.info, regardless of eventual outcome)
    -- but log output isn't necessarily what a caller actually sees as
    "the error" (e.g. inside the Workbench/py_runner, where the raised
    exception is what surfaces to the user, not the container's logs).
    So the raised exception below
    includes the full stdout too, not just the matched "cond: ..." lines,
    letting a user directly inspect the solver's per-iteration output to
    judge solution quality without having to go find the logs separately.
    """
    failures = [
        match.group(0).strip()
        for match in re.finditer(r"cond:\s*(-?\d+),\s*iter:\s*(\d+).*", stdout)
        if int(match.group(1)) != _SFEPY_CONVERGED_CONDITION
    ]

    if failures:
        raise RuntimeError(
            f"sfepy-run on {input_file!r} exited successfully but did not converge "
            f"({len(failures)} of its nonlinear solve(s) reported a non-zero "
            f"condition code, i.e. did not converge):\n"
            + "\n".join(failures)
            + "\nConsider raising linear_solver_i_max, tightening/loosening "
              "linear_solver_eps_r, or checking t1/num_steps are appropriate for "
              "this model's diffusion timescale (see "
              "HydrothermalProblemBuilder._default_diffusion_timescale)."
            + "\n\nFull sfepy-run output:\n"
            + stdout
        )


def _run_sfepy_input_file(
    input_file: str,
    mesh_results: MeshResults,
    mesh_type_code: str,
    output_dir: Optional[str] = None,
    fault_zone_cell_mask: Optional[np.ndarray] = None,
    fault_group_id: Optional[int] = None,
) -> SfepyOutputType:
    """
    Run a generated SfePy input file against mesh_results and return
    a SfepyOutputType(output_dir, is_temp, stdout) -- output_dir/is_temp are
    the shape load_vtk_results() already expects (a drop-in replacement at
    the call site); stdout is carried separately by callers that want it
    (see run_simulation_sfepy()'s SimulationResults.sfepy_stdout), since
    load_vtk_results() itself only cares about the VTK files on disk.

    fault_zone_cell_mask/fault_group_id (both from
    HydrothermalProblemBuilder, both None if no fault zone is active):
    boolean array (one entry per mesh cell, concatenated in mesh_results
    .elements block order) and the extra material group id it maps to.
    When given, cells where the mask is True get their `mat_id` overridden
    from their host block's index to fault_group_id, pulling them into
    their own SfePy region (Omega_fault in the generated input files)
    instead of their host lithology's -- this is the only place mat_id
    actually gets assigned, so it's the one place that override can happen.

    Converts mesh_results to the format SfePy actually reads (MeshResults
    -> Exodus buffer -> meshio round-trip, tagging each cell block with a
    `mat_id` matching its positional index -- this is exactly what makes
    Omega{mat_id} in the generated input files line up with
    mesh_results.elements' block order -> Medit .mesh file), then launches
    `sfepy-run` as a subprocess. The mesh path and output dir are passed to
    that subprocess via its own environment (sfepy-run executes the input
    file fresh in a separate process; unlike most of this builder's
    config, the mesh path is only known after this conversion happens, not
    at input-file-generation time, so it can't be embedded as a literal).

    Builds a local `env` dict for the subprocess rather than mutating the
    real process-global `os.environ`, so nothing needs to be restored on
    the parent side afterward. Streams the subprocess's combined
    stdout/stderr to `logger.info` line by line as it runs (via
    `sys.executable -u`, which forces the subprocess's own Python to be
    unbuffered -- without it, output piped to a non-tty is typically
    block-buffered internally and would arrive in one lump near the end
    anyway, defeating the point of streaming), instead of capturing
    everything and logging it as a single blob after the process exits.
    This gives a live progress signal while a run is in flight (useful to
    tell an active solve apart from a hung one, e.g. from an OOM) and, as
    a side effect, means the full transcript is already logged by the
    time any failure -- crash or non-convergence -- is detected below, so
    a failed run doesn't lose the record a successful one gets. Raises
    with the captured output on a non-zero exit -- a crashed/errored
    SfePy run otherwise exits non-zero with no indication of why. Also
    checks solver *convergence*, not just the return code (see
    _check_convergence()) -- a returncode of 0 only means SfePy didn't
    crash, not that the nonlinear solve actually reached tolerance within
    `i_max` iterations.

    `sfepy-run` is resolved to its absolute path via `shutil.which()`
    rather than passed as the bare command name, and that resolved path
    is passed as an argument to `sys.executable` (the actual child
    process) rather than launched directly. `shutil.which()` is what
    `sys.executable -u`-based streaming above actually needs: it must
    hand a real file path to run as a script, independent of the
    subprocess's own cwd/PATH. This also happens to sidestep a Windows-
    only bug the resolution originally existed for: after GMSH's
    `initialize()`/`finalize()` has run anywhere earlier in the same
    process (i.e. after any `mesh_type="unstructured"` mesh generation),
    Win32 `CreateProcess`'s own executable search stops finding bare
    command names, raising `FileNotFoundError: [WinError 2]`, even though
    `PATH` itself is untouched and `shutil.which()` still resolves it
    correctly -- moot now that `sys.executable` (always absolute) is the
    process actually launched, but resolving `sfepy-run` to an absolute
    path remains correct regardless.
    """
    is_temp = output_dir is None
    if is_temp:
        output_dir = tempfile.mkdtemp(prefix="sfepy_output_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    if mesh_type_code == "unstr":
        # Drop non-volume blocks (dim != 3: fault surfaces, boundary/
        # "extended" surfaces, wells, sources -- see
        # HydrothermalProblemBuilder._map_mat_id_to_lithology's docstring)
        # ourselves, rather than relying on Exo_format.py's own unstructured
        # export path, which strips a hardcoded NUM_SIDE_BLOCKS=6 trailing
        # blocks -- correct only when there are exactly 6 non-volume blocks
        # total. A faulted model has 7+ (one extra triangle block per
        # fault), so that slice would wrongly leave a fault-surface
        # triangle block mixed in with real tetrahedra, which SfePy cannot
        # handle (mixed 3-node/4-node elements in one region). Once
        # pre-filtered to tetra-only, use Exo_format.MeshType.VOLUME_ONLY
        # for the export call -- it behaves exactly like "imp"/"str" there
        # (volume_blocks = all_blocks, no slicing), without mislabeling the
        # mesh's true origin as "imp" just to get that behavior (this
        # mesh's real provenance stays "unstructured" -- see
        # MeshResults.mesh_type -- this only affects which code path
        # Exo_format.py's exporter takes). Boundary conditions
        # (Gamma_Top/Gamma_Bottom) don't depend on this: they're built from
        # mesh_results.point_sets directly in
        # HydrothermalProblemBuilder._region_lines(), never from Exodus
        # side sets, so switching export type here doesn't affect them.
        volume_elements = [block for block in mesh_results.elements if block.dim == 3]
        if len(volume_elements) != len(mesh_results.elements):
            mesh_results = mesh_results.model_copy(update={"elements": volume_elements})
        export_type_code = "volume_only"
    else:
        export_type_code = mesh_type_code

    exo_buffer = export_mesh_results_to_exodus(mesh_results, type=export_type_code)

    tmp_exo_fd, tmp_exo_path = tempfile.mkstemp(suffix=".exo")
    os.close(tmp_exo_fd)
    tmp_mesh_fd, tmp_mesh_path = tempfile.mkstemp(suffix=".mesh")
    os.close(tmp_mesh_fd)

    try:
        with open(tmp_exo_path, "wb") as f:
            f.write(exo_buffer.getbuffer())

        mesh = meshio.read(tmp_exo_path, file_format="exodus")
        mat_ids = []
        offset = 0
        for i, cell_block in enumerate(mesh.cells):
            n = len(cell_block.data)
            ids = np.full(n, i, dtype=int)
            if fault_zone_cell_mask is not None:
                # Relabels in place -- does NOT reorder cells, so this stays
                # consistent with HydrothermalProblemBuilder.compute_darcy_velocity()'s
                # matching per-block mask split (both preserve each cell's
                # relative/global order, just partition it by group).
                block_mask = fault_zone_cell_mask[offset:offset + n]
                ids[block_mask] = fault_group_id
            mat_ids.append(ids)
            offset += n
        mesh.cell_data = {"mat_id": mat_ids}
        mesh.write(tmp_mesh_path, file_format="medit")

        env = dict(os.environ)
        env["SFEpy_OUTPUT_DIR"] = output_dir
        env["TEMP_MESH_FILE"] = tmp_mesh_path

        # Resolved to an absolute path since it's passed as an argument to
        # sys.executable below (a real file path, not a bare command name
        # sys.executable would need to search PATH for) -- see this
        # function's docstring for why sys.executable runs it rather than
        # launching sfepy-run directly.
        sfepy_run_exe = shutil.which("sfepy-run")
        if sfepy_run_exe is None:
            raise RuntimeError(
                "'sfepy-run' not found on PATH -- is the sfepy package installed "
                "in this environment?"
            )

        logger.info("Running SfePy on %s...", input_file)
        process = subprocess.Popen(
            [sys.executable, "-u", sfepy_run_exe, input_file],
            env=env,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        stdout_lines = []
        for line in process.stdout:
            logger.info(line.rstrip())
            stdout_lines.append(line)
        process.wait()
        stdout = "".join(stdout_lines)

        if process.returncode != 0:
            raise RuntimeError(
                f"sfepy-run failed (exit code {process.returncode}) on {input_file!r}:\n"
                f"{stdout or '(no output captured)'}"
            )
        _check_convergence(stdout, input_file)
        logger.info("SfePy finished.")
    finally:
        for p in (tmp_exo_path, tmp_mesh_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return SfepyOutputType(output_dir=output_dir, is_temp=is_temp, stdout=stdout)


def _run_hydrothermal_problem(
    builder: HydrothermalProblemBuilder, keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> SimulationResults:
    """
    Run the full two-stage solve (pressure, then temperature+advection) for
    the given builder and return a single merged SimulationResults object --
    matching the meshing pattern, where a typed result object (MeshResults)
    flows directly out of create_unstructured_mesh_data() etc. and into
    downstream components, rather than a file path callers have to
    separately load.

    The HydrothermalProblemBuilder-specific half of the public
    run_simulation_sfepy() dispatcher below (see its docstring) -- kept as
    its own function, not inlined, so this already-tested two-stage logic
    stays untouched by the CustomSfepyBuilder branch.

    A standalone function taking the builder as input (not a method on it)
    to match the Workbench's @wbgeo_component style, where components are
    plain functions with typed inputs/outputs rather than methods on a
    stateful class -- HydrothermalProblemBuilder itself is the input-data
    generator (its constructor arguments are the component's configurable
    inputs); this is the actual "run" component.

    Solved as two sequential SfePy runs rather than one combined system:
    mixing pressure (~1e6 Pa) and temperature (~10s of degC) in the same
    matrix is badly conditioned for iterative solvers, regardless of
    preconditioning. Segregating matches the physics anyway (T depends on
    the flow field, but p never depends on T), and each stage is a
    well-conditioned single-physics problem on its own.

    No files are left behind by default: SfePy's own output directories
    are always fresh temporary ones, and load_vtk_results() deletes them
    automatically once each stage's result is loaded in memory (see
    _run_sfepy_input_file() / load_vtk_results()) -- unless output_dir
    below is given. To get a persisted file out of a SimulationResults
    object regardless (e.g. from the Workbench, which has no on-disk-path
    affordance), use export_simulation_results() below -- analogous to
    Download Mesh / export_mesh_results() for MeshResults, as its own
    separate Workbench component taking SimulationResults as input.

    The returned SimulationResults.sfepy_stdout carries each stage's raw
    solver output ("pressure"/"heat"), regardless of whether the solve
    converged -- so solver quality can be inspected on a normal, successful
    run too, not only via the exception raised on non-convergence.

    Uses builder.mesh_type (set at construction, validated against how
    builder.mesh_results was actually produced) -- not re-specified here,
    to avoid two separate places that could silently disagree.

    Time-key semantics (easy to get wrong, so noted explicitly here): the
    returned SimulationResults' nodes_by_time/node_data_by_time are keyed by
    each saved VTK output file's 0-based *step index*, not physical time and
    not including a separate initial-condition snapshot -- SfePy only saves
    one file per actually-*solved* step. So for num_steps=N, valid keys are
    0..N-1, and the fully evolved final state is always at
    max(result.nodes_by_time.keys()), not at key 0 or at t1.

    keep_files_dir: optional directory to write the generated input
    files and the intermediate velocity .npz into, for debugging -- if
    given, these are NOT deleted afterward. If omitted (the default),
    temporary files are used and cleaned up, keeping the whole call
    file-free from the caller's perspective.

    output_dir: optional directory to persist SfePy's own raw VTK output
    into, for post-processing outside this pipeline (e.g. opening directly
    in ParaView) -- if given, NOT deleted afterward (see
    _run_sfepy_input_file's output_dir / load_vtk_results' is_temp
    handling). Pressure and heat are separate SfePy runs, each producing
    its own raw VTK files (pressure.*.vtk / heat.*.vtk) that never get
    merged on disk -- only in the SimulationResults this function returns
    -- so each stage gets its own subdirectory here to keep them from
    being walked together by load_vtk_results(). If omitted (the default,
    unchanged behavior), each stage uses a fresh temp directory that's
    deleted once loaded into memory.
    """
    def _path(name):
        """Resolve a generated-file name to keep_files_dir if given, else a fresh temp path."""
        if keep_files_dir is not None:
            return os.path.join(keep_files_dir, name)
        fd, p = tempfile.mkstemp(suffix=f"_{name}")
        os.close(fd)
        return p

    pressure_file = _path("pressure.py") if builder.include_flow else None
    heat_file = _path("heat.py")
    velocity_npz = _path("velocity.npz") if builder.include_flow else None
    # _run_sfepy_input_file()/export_mesh_results_to_exodus() need the short
    # MeshType code ("imp"/"str"/"unstr"), not builder.mesh_type's readable name.
    mesh_type_code = MESH_TYPE_CODES[builder.mesh_type]
    pressure_output_dir = os.path.join(output_dir, "pressure") if output_dir is not None else None
    heat_output_dir = os.path.join(output_dir, "heat") if output_dir is not None else None

    try:
        pressure_sim = None
        time = None
        if builder.include_flow:
            # Stage 1: pressure
            builder.build_pressure_input_file(pressure_file)
            p_out = _run_sfepy_input_file(
                pressure_file, builder.mesh_results, mesh_type_code, output_dir=pressure_output_dir,
                fault_zone_cell_mask=builder.fault_zone_cell_mask, fault_group_id=builder.fault_group_id,
            )
            pressure_sim = load_vtk_results(p_out)
            if not pressure_sim.nodes_by_time:
                # _run_sfepy_input_file() already checks sfepy-run's return
                # code, but load_vtk_results() can still come back empty if
                # sfepy "succeeded" without actually writing output (e.g. an
                # input file with no equations/solver would exit 0 and do
                # nothing) -- fail clearly here instead of an opaque
                # "not enough values to unpack" a few lines down.
                raise RuntimeError("Pressure stage produced no SfePy output (0 VTK files).")
            (time,) = pressure_sim.nodes_by_time.keys()  # single steady-state solution

            # Velocity hand-off
            velocities = builder.compute_darcy_velocity(pressure_sim, time)
            np.savez(velocity_npz, **{f"v{mat_id}": arr for mat_id, arr in velocities.items()})

        # Stage 2: temperature (pure conduction if not builder.include_flow)
        builder.build_heat_input_file(heat_file, velocity_npz)
        t_out = _run_sfepy_input_file(
            heat_file, builder.mesh_results, mesh_type_code, output_dir=heat_output_dir,
            fault_zone_cell_mask=builder.fault_zone_cell_mask, fault_group_id=builder.fault_group_id,
        )
        heat_sim = load_vtk_results(t_out)
        if not heat_sim.nodes_by_time:
            raise RuntimeError("Heat stage produced no SfePy output (0 VTK files).")

        # Merge: same mesh/time for both stages, just different fields.
        merged = SimulationResults()
        for t in heat_sim.nodes_by_time:
            merged.nodes_by_time[t] = heat_sim.nodes_by_time[t]
            merged.cells_by_time[t] = heat_sim.cells_by_time[t]
            merged.celltypes_by_time[t] = heat_sim.celltypes_by_time[t]
            merged.node_data_by_time[t] = dict(heat_sim.node_data_by_time[t])
            merged.cell_data_by_time[t] = dict(heat_sim.cell_data_by_time.get(t, {}))
            if builder.include_flow and time in pressure_sim.node_data_by_time:
                merged.node_data_by_time[t]["p"] = pressure_sim.node_data_by_time[time]["p"]
        merged.sfepy_stdout = {"heat": t_out.stdout}
        if builder.include_flow:
            merged.sfepy_stdout["pressure"] = p_out.stdout
        return merged
    finally:
        if keep_files_dir is None:
            for p in (pressure_file, heat_file, velocity_npz):
                if p is None:
                    continue
                try:
                    os.remove(p)
                except OSError:
                    pass


def _run_custom_sfepy_problem(
    problem: CustomSfepyBuilder, keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> SimulationResults:
    """
    Run a CustomSfepyBuilder's file once and return the result -- no
    velocity hand-off, no merge step, unlike _run_hydrothermal_problem's
    two-stage sequence. The file is assumed to already contain all
    staging/sequencing logic and to write out everything it needs in one
    sfepy-run invocation. problem.fault_zone_cell_mask/fault_group_id
    (both None unless CustomSfepyBuilder.fault_zone_n_voxels was set) are
    passed through to _run_sfepy_input_file the same way
    _run_hydrothermal_problem already does -- the mat_id relabeling itself
    is entirely generic, not HydrothermalProblemBuilder-specific.

    The returned SimulationResults.sfepy_stdout carries the raw solver
    output under the "custom" key, same purpose as
    _run_hydrothermal_problem's "pressure"/"heat" keys.

    problem.input_file_contents is written to a fresh temp file here (not
    at CustomSfepyBuilder construction time) for the same reason
    _run_hydrothermal_problem writes its generated input files at run
    time: the content is what survives @wbgeo_type serialization, a temp
    path would not.

    output_dir: optional directory to persist SfePy's own raw VTK output
    into -- see _run_hydrothermal_problem's output_dir docstring. Only one
    sfepy-run invocation here (unlike the two-stage hydrothermal case), so
    no subdirectory split is needed -- output_dir is used as-is.
    """
    def _path(name):
        """Resolve a generated-file name to keep_files_dir if given, else a fresh temp path."""
        if keep_files_dir is not None:
            return os.path.join(keep_files_dir, name)
        fd, p = tempfile.mkstemp(suffix=f"_{name}")
        os.close(fd)
        return p

    input_file = _path("custom.py")
    mesh_type_code = MESH_TYPE_CODES[problem.mesh_type]
    try:
        with open(input_file, "w") as f:
            f.write(problem.input_file_contents)
        out = _run_sfepy_input_file(
            input_file, problem.mesh_results, mesh_type_code, output_dir=output_dir,
            fault_zone_cell_mask=problem.fault_zone_cell_mask, fault_group_id=problem.fault_group_id,
        )
        sim = load_vtk_results(out)
        if not sim.nodes_by_time:
            raise RuntimeError("Custom SfePy input file produced no SfePy output (0 VTK files).")
        sim.sfepy_stdout = {"custom": out.stdout}
        return sim
    finally:
        if keep_files_dir is None:
            try:
                os.remove(input_file)
            except OSError:
                pass


def run_simulation_sfepy(
    problem: Union[HydrothermalProblemBuilder, CustomSfepyBuilder], keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> SimulationResults:
    """
    Run either kind of SfePy problem and return a SimulationResults in the
    same format regardless of which one was given -- a HydrothermalProblemBuilder
    (auto-generated two-stage pressure/heat problem, see
    _run_hydrothermal_problem) or a CustomSfepyBuilder (a user-supplied,
    already-complete SfePy input file with its own internal
    staging/sequencing logic, see _run_custom_sfepy_problem). One runner
    for both, dispatching on type, so callers (and the Workbench) don't
    need two separate "run" components with diverging behavior/result
    shapes to keep in sync.

    output_dir: optional directory to persist SfePy's own raw VTK output
    into instead of the default temp-and-delete behavior -- see
    _run_hydrothermal_problem's output_dir docstring for the two-stage
    subdirectory layout used when problem is a HydrothermalProblemBuilder.
    For script/library callers only: the Workbench has no on-disk-path
    affordance, so its "Run Simulation" component always leaves this at
    the default (None) -- use the "Export Simulation Results" component
    (export_simulation_results()) to get a file out of a Workbench run.
    """
    if is_custom_sfepy_builder(problem):
        return _run_custom_sfepy_problem(problem, keep_files_dir, output_dir)
    return _run_hydrothermal_problem(problem, keep_files_dir, output_dir)


def export_simulation_results(sim: SimulationResults, format: str = "vtk") -> BasicallyABufferedFile:
    """
    "Download Simulation Results" component, mirroring export_mesh_results_to_X()
    / "Download Mesh" for MeshResults. run_simulation_sfepy() doesn't keep
    SfePy's raw output files around (see its docstring), so this serializes
    SimulationResults's in-memory dicts (nodes_by_time, cells_by_time,
    node_data_by_time, ...) back out to a file, rather than reusing any
    on-disk SfePy output.

    format="vtk" is the only one implemented so far -- see
    _export_simulation_results_to_vtk() for what it actually produces.
    Exodus (mirroring export_mesh_results_to_exodus, already used
    internally by _run_sfepy_input_file() for the mesh side) would be the
    natural next format if needed.
    """
    if format != "vtk":
        raise NotImplementedError(
            f"export_simulation_results only supports format='vtk' so far, got {format!r}."
        )
    return _export_simulation_results_to_vtk(sim)


def _export_simulation_results_to_vtk(sim: SimulationResults) -> BasicallyABufferedFile:
    """
    Writes one legacy .vtk file per solved time step (via
    simulation_visualization.build_grid_from_class() + PyVista's own writer --
    same legacy .vtk format core.meshing_components.mesh_format.vtk.VTK_format
    uses for MeshResults, for consistency) plus a ParaView .pvd collection
    file tying them together with their actual time values, so the result
    opens directly in ParaView with time-stepping/animation across the whole
    transient result -- not just a single snapshot. All bundled into one
    .zip, since BasicallyABufferedFile is a single in-memory file and a
    multi-file time series can't be one raw VTK file.
    """
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("SimulationResults has no time steps to export.")

    tmp_dir = tempfile.mkdtemp(prefix="sim_export_")
    try:
        vtk_names = [f"result_{i:04d}.vtk" for i in range(len(times))]
        for name, t in zip(vtk_names, times):
            grid = build_grid_from_class(sim, t)
            grid.save(os.path.join(tmp_dir, name))

        dataset_lines = "\n".join(
            f'    <DataSet timestep="{t}" group="" part="0" file="{name}"/>'
            for name, t in zip(vtk_names, times)
        )
        pvd_content = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            "  <Collection>\n"
            f"{dataset_lines}\n"
            "  </Collection>\n"
            "</VTKFile>\n"
        )
        pvd_path = os.path.join(tmp_dir, "results.pvd")
        with open(pvd_path, "w") as f:
            f.write(pvd_content)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pvd_path, "results.pvd")
            for name in vtk_names:
                zf.write(os.path.join(tmp_dir, name), name)
        buf.seek(0)
        buf.filename = "simulation_results.zip"
        return buf
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
