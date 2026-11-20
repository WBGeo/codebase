"""
Runs the SfePy input files SfepyHydrothermalProblemBuilder generates
(sfepy_model_builder.py) and loads the results back into a
SimulationResults object -- split out from the builder to keep "build the
input data" and "run the simulation" as separate concerns, matching the
Workbench's component/connector model: SfepyHydrothermalProblemBuilder is the
input-data-generator component, run_simulation_sfepy() here is the run
component that takes it as input.

Depends on export_mesh_results_to_exodus (mesh export/format-conversion
logic, owned by meshing, not simulation).
"""
import io
import types
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Optional, Union
from meshio import CellBlock

import meshio
import numpy as np

from core.object_components import MeshResults, SimulationResults
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results, SfepyOutputType
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    SfepyHydrothermalProblemBuilder,
    SfepyCustomBuilder,
    MESH_TYPE_CODES,
    enumerate_rock_units,
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
#: given non-zero code specifically means. SfepyCustomBuilder lets a user
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
    solver config (SfepyHydrothermalProblemBuilder._newton_ts_options_blocks) --
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
              "linear_solver_eps_r, or checking end_time/num_steps are appropriate for "
              "this model's diffusion timescale (see "
              "default_diffusion_timescale)."
            + "\n\nFull sfepy-run output:\n"
            + stdout
        )


def _merge_consecutive_cell_blocks(
    cells: list[CellBlock], mat_ids: list[np.ndarray]
) -> tuple[list[CellBlock], list[np.ndarray]]:
    """Merge consecutive cell blocks of the same cell type, together with their mat_ids.

    SfePy keeps one element group per input block, and its VTK writer then
    writes all cells of a cell type once per group -- with N lithology blocks
    of hexahedra the result held every cell N times (GitHub issue #4). After
    merging, the lithology of each cell lives only in its mat_id, which is
    what the generated input files use anyway (``cells of group N``).

    Only consecutive blocks are merged, so the global cell order -- which the
    fault-zone mask and SfepyHydrothermalProblemBuilder.compute_darcy_velocity()
    rely on -- stays exactly as it was.
    """
    merged_cells: list[CellBlock] = []
    merged_ids: list[np.ndarray] = []
    for block, ids in zip(cells, mat_ids):
        if merged_cells and merged_cells[-1].type == block.type:
            merged_cells[-1] = CellBlock(
                block.type, np.concatenate([merged_cells[-1].data, block.data])
            )
            merged_ids[-1] = np.concatenate([merged_ids[-1], ids])
        else:
            merged_cells.append(CellBlock(block.type, block.data))
            merged_ids.append(ids)
    return merged_cells, merged_ids


def _run_sfepy_input_file(
    input_file: str,
    mesh_results: MeshResults,
    mesh_type_code: str,
    output_dir: Optional[str] = None,
    fault_zone_cell_mask: Optional[np.ndarray] = None,
    fault_group_id: Optional[int] = None,
    stream_output: bool = False,
) -> SfepyOutputType:
    """
    Run a generated SfePy input file against mesh_results and return
    a SfepyOutputType(output_dir, is_temp, stdout) -- output_dir/is_temp are
    the shape load_vtk_results() already expects (a drop-in replacement at
    the call site); stdout is carried separately by callers that want it
    (see run_simulation_sfepy()'s SimulationResults.solver_output), since
    load_vtk_results() itself only cares about the VTK files on disk.

    fault_zone_cell_mask/fault_group_id (both from
    SfepyHydrothermalProblemBuilder, both None if no fault zone is active):
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

    boundary_mode = "fixed_trailing"
    if mesh_type_code == "unstr":
        # Keep only the volume blocks (dim == 3); fault surfaces, boundary/
        # "extended" surfaces, wells and sources are not part of the SfePy
        # domain (see SfepyHydrothermalProblemBuilder._map_mat_id_to_lithology).
        # Their position in mesh_results.elements varies (a faulted model has
        # one extra triangle block per fault), so the exporter's default
        # "fixed_trailing" mode -- which treats exactly the last six blocks as
        # boundaries -- must not be used here. boundary_mode="dim" writes the
        # 3D blocks in their original order, whatever comes after them.
        # Boundary conditions (Gamma_Top/Gamma_Bottom) are built from
        # mesh_results.point_sets in SfepyHydrothermalProblemBuilder.
        # _region_lines(), never from Exodus side sets, so they don't depend
        # on this.
        mesh_results = mesh_results.model_copy(
            update={"elements": [block for block in mesh_results.elements if block.dim == 3]}
        )
        boundary_mode = "dim"

    exo_buffer = export_mesh_results_to_exodus(
        mesh_results, type=mesh_type_code, boundary_mode=boundary_mode
    )

    tmp_exo_fd, tmp_exo_path = tempfile.mkstemp(suffix=".exo")
    os.close(tmp_exo_fd)
    tmp_mesh_fd, tmp_mesh_path = tempfile.mkstemp(suffix=".mesh")
    os.close(tmp_mesh_fd)

    try:
        with open(tmp_exo_path, "wb") as f:
            f.write(exo_buffer.getbuffer())
        from meshio.exodus import _exodus

        # Use a private copy of meshio's Exodus element mapping for the
        # TETRA4/TET4 aliases emitted by our exporter. Do not mutate meshio's
        # process-global dictionary (which other readers may also use).
        reader_globals = dict(_exodus.read.__globals__)
        element_types = dict(_exodus.exodus_to_meshio_type)
        element_types.update({"TETRA4": "tetra", "TET4": "tetra"})
        reader_globals["exodus_to_meshio_type"] = element_types
        local_exodus_reader = types.FunctionType(
            _exodus.read.__code__, reader_globals,
            _exodus.read.__name__, _exodus.read.__defaults__,
            _exodus.read.__closure__,
        )
        mesh = local_exodus_reader(tmp_exo_path)
        mat_ids = []
        offset = 0
        for i, cell_block in enumerate(mesh.cells):
            n = len(cell_block.data)
            ids = np.full(n, i, dtype=int)
            if fault_zone_cell_mask is not None:
                # Relabels in place -- does NOT reorder cells, so this stays
                # consistent with SfepyHydrothermalProblemBuilder.compute_darcy_velocity()'s
                # matching per-block mask split (both preserve each cell's
                # relative/global order, just partition it by group).
                block_mask = fault_zone_cell_mask[offset:offset + n]
                ids[block_mask] = fault_group_id
            mat_ids.append(ids)
            offset += n
        mesh.cells, mat_ids = _merge_consecutive_cell_blocks(mesh.cells, mat_ids)
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
        if stream_output:
            print(f"SfePy: starting {os.path.basename(input_file)}", flush=True)
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
            if stream_output:
                # Do not rely on the application's logging level: an INFO
                # logger is commonly hidden in WBGeo scripts.  The child is
                # launched unbuffered and each line is flushed immediately.
                print(line, end="", flush=True)
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
        if stream_output:
            print(f"SfePy: finished {os.path.basename(input_file)}", flush=True)
    finally:
        for p in (tmp_exo_path, tmp_mesh_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return SfepyOutputType(output_dir=output_dir, is_temp=is_temp, stdout=stdout)


def _run_hydrothermal_problem(
    builder: SfepyHydrothermalProblemBuilder, keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None, stream_output: bool = False,
) -> SimulationResults:
    """
    Run the full two-stage solve (pressure, then temperature+advection) for
    the given builder and return a single merged SimulationResults object --
    matching the meshing pattern, where a typed result object (MeshResults)
    flows directly out of create_unstructured_mesh_data() etc. and into
    downstream components, rather than a file path callers have to
    separately load.

    The SfepyHydrothermalProblemBuilder-specific half of the public
    run_simulation_sfepy() dispatcher below (see its docstring) -- kept as
    its own function, not inlined, so this already-tested two-stage logic
    stays untouched by the SfepyCustomBuilder branch.

    A standalone function taking the builder as input (not a method on it)
    to match the Workbench's @wbgeo_component style, where components are
    plain functions with typed inputs/outputs rather than methods on a
    stateful class -- SfepyHydrothermalProblemBuilder itself is the input-data
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

    The returned SimulationResults.solver_output carries each stage's raw
    solver output ("pressure"/"heat"), regardless of whether the solve
    converged -- so solver quality can be inspected on a normal, successful
    run too, not only via the exception raised on non-convergence.

    Uses builder.mesh_type (set at construction, validated against how
    builder.mesh_results was actually produced) -- not re-specified here,
    to avoid two separate places that could silently disagree.

    Time-key semantics (easy to get wrong, so noted explicitly here): the
    returned SimulationResults' nodes_by_time/node_data_by_time are keyed by
    physical time. For num_steps=N there are N+1 keys, evenly spaced from
    start_time (the initial state) to end_time, the same times GOLEM writes;
    the fully evolved final state is at max(result.nodes_by_time.keys()).

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
                stream_output=stream_output,
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
            stream_output=stream_output,
        )
        heat_sim = load_vtk_results(t_out)
        if not heat_sim.nodes_by_time:
            raise RuntimeError("Heat stage produced no SfePy output (0 VTK files).")

        # SfePy's legacy VTK writer commonly stores the output sequence as
        # 0, 1, ..., num_steps instead of preserving physical time.  Map the
        # complete saved sequence back to the exact times used by GOLEM so
        # downstream plots and comparisons address identical time points.
        stored_heat_times = sorted(heat_sim.nodes_by_time)
        expected_count = builder.num_steps + 1
        if len(stored_heat_times) == expected_count:
            physical_heat_times = np.linspace(
                builder.start_time, builder.end_time, expected_count,
            ).tolist()
        else:
            # Retain reader-provided times for unusual custom output
            # schedules instead of guessing an incorrect mapping.
            physical_heat_times = stored_heat_times

        # Merge: same mesh/time for both stages, just different fields.
        merged = SimulationResults()
        for stored_t, physical_t in zip(stored_heat_times, physical_heat_times):
            merged.nodes_by_time[physical_t] = heat_sim.nodes_by_time[stored_t]
            merged.cells_by_time[physical_t] = heat_sim.cells_by_time[stored_t]
            merged.celltypes_by_time[physical_t] = heat_sim.celltypes_by_time[stored_t]
            merged.node_data_by_time[physical_t] = dict(heat_sim.node_data_by_time[stored_t])
            merged.cell_data_by_time[physical_t] = dict(
                heat_sim.cell_data_by_time.get(stored_t, {})
            )
            if builder.include_flow and time in pressure_sim.node_data_by_time:
                merged.node_data_by_time[physical_t]["p"] = pressure_sim.node_data_by_time[time]["p"]
        merged.solver = "sfepy"
        merged.solver_output = {"heat": t_out.stdout}
        merged.input_files = {"heat.py": _read_text(heat_file)}
        if builder.include_flow:
            merged.solver_output = {"pressure": p_out.stdout, **merged.solver_output}
            merged.input_files = {"pressure.py": _read_text(pressure_file), **merged.input_files}
        merged.block_names = {
            mat_id: builder._lithology_name(lith_id) for mat_id, lith_id in builder.mat_id_to_lithology.items()
        }
        if builder.fault_group_id is not None:
            merged.block_names[builder.fault_group_id] = builder.fault_zone_properties.name
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
    problem: SfepyCustomBuilder, keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None, stream_output: bool = False,
) -> SimulationResults:
    """
    Run a SfepyCustomBuilder's file once and return the result -- no
    velocity hand-off, no merge step, unlike _run_hydrothermal_problem's
    two-stage sequence. The file is assumed to already contain all
    staging/sequencing logic and to write out everything it needs in one
    sfepy-run invocation. problem.fault_zone_cell_mask/fault_group_id
    (both None unless SfepyCustomBuilder.fault_zone_n_voxels was set) are
    passed through to _run_sfepy_input_file the same way
    _run_hydrothermal_problem already does -- the mat_id relabeling itself
    is entirely generic, not SfepyHydrothermalProblemBuilder-specific.

    The returned SimulationResults.solver_output carries the raw solver
    output under the "custom" key, same purpose as
    _run_hydrothermal_problem's "pressure"/"heat" keys.

    problem.input_file_contents is written to a fresh temp file here (not
    at SfepyCustomBuilder construction time) for the same reason
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
            stream_output=stream_output,
        )
        sim = load_vtk_results(out)
        if not sim.nodes_by_time:
            raise RuntimeError("Custom SfePy input file produced no SfePy output (0 VTK files).")
        sim.solver = "sfepy"
        sim.solver_output = {"custom": out.stdout}
        sim.input_files = {"custom.py": problem.input_file_contents}
        if problem.geomodel_result is not None:
            lith_names = dict(enumerate_rock_units(problem.geomodel_result))
            sim.block_names = {mat_id: lith_names[lith_id]
                               for mat_id, lith_id in problem.mat_id_to_lithology.items() if lith_id in lith_names}
        if problem.fault_group_id is not None:
            sim.block_names[problem.fault_group_id] = "fault_zone"
        return sim
    finally:
        if keep_files_dir is None:
            try:
                os.remove(input_file)
            except OSError:
                pass


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as stream:
        return stream.read()


def run_simulation_sfepy(
    problem: Union[SfepyHydrothermalProblemBuilder, SfepyCustomBuilder], keep_files_dir: Optional[str] = None,
    output_dir: Optional[str] = None, stream_output: bool = False,
) -> SimulationResults:
    """
    Run either kind of SfePy problem and return a SimulationResults in the
    same format regardless of which one was given -- a SfepyHydrothermalProblemBuilder
    (auto-generated two-stage pressure/heat problem, see
    _run_hydrothermal_problem) or a SfepyCustomBuilder (a user-supplied,
    already-complete SfePy input file with its own internal
    staging/sequencing logic, see _run_custom_sfepy_problem). One runner
    for both, dispatching on type, so callers (and the Workbench) don't
    need two separate "run" components with diverging behavior/result
    shapes to keep in sync.

    output_dir: optional directory to persist SfePy's own raw VTK output
    into instead of the default temp-and-delete behavior -- see
    _run_hydrothermal_problem's output_dir docstring for the two-stage
    subdirectory layout used when problem is a SfepyHydrothermalProblemBuilder.
    For script/library callers only: the Workbench has no on-disk-path
    affordance, so its "Run SfePy Simulation" component always leaves this at
    the default (None) -- use the "Export Simulation Results" component
    (export_simulation_results()) to get a file out of a Workbench run.

    stream_output: print sfepy-run's output live to the console while it
    runs (scripts only, as in run_simulation_golem). The output is captured
    into the result either way; the builder's print_linear_residuals only
    decides whether it contains the linear solver residuals.
    """
    if is_custom_sfepy_builder(problem):
        return _run_custom_sfepy_problem(problem, keep_files_dir, output_dir, stream_output)
    return _run_hydrothermal_problem(problem, keep_files_dir, output_dir, stream_output)



# Backward-compatible import. The generic exporter is solver-independent and
# now lives in simulation_packages/simulation_export.py.
from core.simulation_components.simulation_packages.simulation_export import (
    export_simulation_results,
)
