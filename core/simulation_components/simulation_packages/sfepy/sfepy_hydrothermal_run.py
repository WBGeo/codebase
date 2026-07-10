"""
Runs the SfePy input files HydrothermalProblemBuilder generates
(sfepy_hydrothermal_builder.py) and loads the results back into a
SimulationResults object -- split out from the builder to keep "build the
input data" and "run the simulation" as separate concerns, matching the
Workbench's component/connector model: HydrothermalProblemBuilder is the
input-data-generator component, run_simulation_sfepy() here is the run
component that takes it as input.

This sandbox owns its own "run sfepy" step (_run_sfepy_input_file below)
rather than depending on the colleague's run_sfepy()
(simulation_packages/sfepy/simulation_run.py) -- that function never checks
the sfepy-run subprocess's return code, so a crashed/errored solve silently
looks like success (see _run_sfepy_input_file's docstring for the specifics
and the fixes). This sandbox is intended to eventually replace that module
entirely, so the fix belongs here rather than as a workaround layered on
top of it. Still depends on export_mesh_results_to_exodus (mesh
export/format-conversion logic, owned by meshing, not simulation).
"""
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Dict, Optional

import meshio
import numpy as np

from core.object_components import MeshResults, SimulationResults
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder,
    MESH_TYPE_CODES,
)
from core.simulation_components.simulation_visualization.simulation_visualization import build_grid_from_class
from py_api_wbgeo.nodesapi import BasicallyABufferedFile

logger = logging.getLogger(__name__)


#: SfePy's own nonlinear-solver convergence codes (see conv_test() in
#: sfepy/solvers/nls.py's docstring): 0 = converged (tolerances met),
#: 1 = max iterations reached without converging, 2 = linesearch gave up.
#: Only 0 counts as success here.
_SFEPY_CONVERGED_CONDITION = 0
_SFEPY_COND_MEANINGS = {
    1: "max iterations reached without converging",
    2: "linesearch gave up",
}


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
    Before this check existed, a run that silently failed to converge
    (exit code 0, but the solution wasn't actually valid) would have
    looked identical to a real success -- the builder's own docstring
    even said as much ("as with any iterative method, verify convergence
    -- check the printed residual"), i.e. it used to be the caller's job.
    """
    failures = []
    for match in re.finditer(r"cond:\s*(-?\d+),\s*iter:\s*(\d+).*", stdout):
        condition = int(match.group(1))
        if condition != _SFEPY_CONVERGED_CONDITION:
            reason = _SFEPY_COND_MEANINGS.get(condition, f"unknown condition {condition}")
            failures.append(f"{match.group(0).strip()} ({reason})")

    if failures:
        raise RuntimeError(
            f"sfepy-run on {input_file!r} exited successfully but did not converge "
            f"({len(failures)} of its nonlinear solve(s) failed to reach tolerance):\n"
            + "\n".join(failures)
            + "\nConsider raising linear_solver_i_max, tightening/loosening "
              "linear_solver_eps_r, or checking t1/num_steps are appropriate for "
              "this model's diffusion timescale (see "
              "HydrothermalProblemBuilder._default_diffusion_timescale)."
        )


def _run_sfepy_input_file(
    input_file: str,
    mesh_results: MeshResults,
    mesh_type_code: str,
    output_dir: Optional[str] = None,
    fault_zone_cell_mask: Optional[np.ndarray] = None,
    fault_group_id: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run a generated SfePy input file against mesh_results and return
    {"output_dir": ..., "is_temp": ...} (same shape load_vtk_results()
    already expects, so it's a drop-in replacement at the call site).

    fault_zone_cell_mask/fault_group_id (both from
    HydrothermalProblemBuilder, both None if no fault zone is active):
    boolean array (one entry per mesh cell, concatenated in mesh_results
    .elements block order) and the extra material group id it maps to.
    When given, cells where the mask is True get their `mat_id` overridden
    from their host block's index to fault_group_id, pulling them into
    their own SfePy region (Omega_fault in the generated input files)
    instead of their host lithology's -- this is the only place mat_id
    actually gets assigned, so it's the one place that override can happen.

    Own replacement for the colleague's run_sfepy() (simulation_run.py) --
    this sandbox is intended to eventually replace that module entirely, so
    the actual "run sfepy" mechanics live here rather than depending on it.

    Same core approach: MeshResults -> Exodus buffer -> meshio round-trip
    (tags each cell block with a `mat_id` matching its positional index --
    this is exactly what makes Omega{mat_id} in the generated input files
    line up with mesh_results.elements' block order) -> Medit .mesh file ->
    sfepy-run subprocess. The mesh path and output dir are passed to that
    subprocess via its own environment (sfepy-run executes the input file
    fresh in a separate process; unlike most of this builder's config, the
    mesh path is only known after this conversion happens, not at
    input-file-generation time, so it can't be embedded as a literal).

    Five real gaps fixed vs. the original:
    - Return code was never checked at all -- a crashed/errored SfePy run
      silently looked like success, and load_vtk_results() finding zero
      .vtk files (several call frames away, logged at INFO level) was the
      only symptom. Fixed: checks returncode, raises with the captured
      output on failure.
    - stdout/stderr were not captured (streamed directly to the parent's
      console with no way to recover them programmatically). Fixed:
      captured via subprocess.run(capture_output=True), so the actual SfePy
      error ends up in the raised exception instead of just scrolling past.
    - The original mutated the real process-global os.environ
      (os.environ["SFEpy_OUTPUT_DIR"] = ..., etc.) and never cleaned it up
      afterward. Fixed: builds a local env dict instead --
      subprocess.run's env= only affects the child process, so there's
      nothing to mutate or restore on the parent side at all.
    - Solver *convergence* was never checked either -- a returncode of 0
      only means SfePy didn't crash, not that the nonlinear solve actually
      reached tolerance within i_max iterations; a silently non-converged
      "solution" used to look identical to a real one. Fixed: see
      _check_convergence() below.
    - On Windows, mesh_type="unstructured" always broke this subprocess call
      with FileNotFoundError: [WinError 2] -- GMSH's initialize()/finalize()
      (run during unstructured mesh generation, earlier in the same process)
      leaves Win32 CreateProcess's own executable search unable to find bare
      command names, even though PATH itself is untouched and
      shutil.which() still resolves them fine. Fixed: resolve "sfepy-run" to
      its absolute path via shutil.which() before calling subprocess.run,
      sidestepping CreateProcess's own search entirely.
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
        # total. With any fault present there are 7+ (one extra triangle
        # block per fault), so that slice wrongly leaves a fault-surface
        # triangle block mixed in with real tetrahedra, which SfePy then
        # fails on (mixed 3-node/4-node elements in one region -- confirmed
        # by reproducing on model2: "ValueError: Size of label 'j' for
        # operand 1 (4) does not match previous terms (3)" during the heat
        # solve). Once pre-filtered to tetra-only, use
        # Exo_format.MeshType.VOLUME_ONLY for the export call -- it behaves
        # exactly like "imp"/"str" there (volume_blocks = all_blocks, no
        # slicing), added specifically so this doesn't have to mislabel the
        # mesh's true origin as "imp" just to get that behavior (this mesh's
        # real provenance stays "unstructured" -- see MeshResults.mesh_type
        # -- this only affects which code path Exo_format.py's exporter
        # takes). Boundary conditions (Gamma_Top/Gamma_Bottom) don't depend
        # on this: they're built from mesh_results.point_sets directly in
        # HydrothermalProblemBuilder._region_lines(), never from Exodus side
        # sets, so switching export type here doesn't affect them.
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

        # Resolved to an absolute path rather than passed as the bare name
        # "sfepy-run" -- on Windows, after GMSH's initialize()/finalize() has
        # run anywhere earlier in this process (i.e. after any
        # mesh_type="unstructured" mesh generation), Win32 CreateProcess's own
        # executable search stops finding bare command names, raising
        # FileNotFoundError: [WinError 2], even though PATH itself is
        # untouched and shutil.which() still resolves it correctly. Passing
        # the resolved absolute path sidesteps CreateProcess's search
        # entirely, which avoids the problem.
        sfepy_run_exe = shutil.which("sfepy-run")
        if sfepy_run_exe is None:
            raise RuntimeError(
                "'sfepy-run' not found on PATH -- is the sfepy package installed "
                "in this environment?"
            )

        logger.info("Running SfePy on %s...", input_file)
        result = subprocess.run(
            [sfepy_run_exe, input_file],
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.info(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"sfepy-run failed (exit code {result.returncode}) on {input_file!r}:\n"
                f"{result.stderr or result.stdout or '(no output captured)'}"
            )
        _check_convergence(result.stdout, input_file)
        logger.info("SfePy finished.")
    finally:
        for p in (tmp_exo_path, tmp_mesh_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return {"output_dir": output_dir, "is_temp": is_temp}


def run_simulation_sfepy(
    builder: HydrothermalProblemBuilder, keep_files_dir: Optional[str] = None
) -> SimulationResults:
    """
    Run the full two-stage solve (pressure, then temperature+advection) for
    the given builder and return a single merged SimulationResults object --
    matching the meshing pattern, where a typed result object (MeshResults)
    flows directly out of create_unstructured_mesh_data() etc. and into
    downstream components, rather than a file path callers have to
    separately load.

    A standalone function taking the builder as input (not a method on it)
    to match the Workbench's @wbgeo_component style, where components are
    plain functions with typed inputs/outputs rather than methods on a
    stateful class -- HydrothermalProblemBuilder itself is the input-data
    generator (its constructor arguments are the component's configurable
    inputs); this is the actual "run" component.

    Solved as two sequential SfePy runs rather than one combined system:
    an earlier monolithic version (pressure and temperature in one
    equations dict, solved together) turned out to be badly conditioned
    for iterative solvers -- pressure (~1e6 Pa) and temperature (~10s of
    degC) in the same matrix, which no amount of preconditioning fixed
    (validated: relative residual stuck near 1.0 even with ILU +
    diagonal scaling). Segregating matches the physics anyway (T depends
    on the flow field, but p never depends on T), and each stage is a
    well-conditioned single-physics problem on its own.

    No files are left behind by default: SfePy's own output directories
    are always fresh temporary ones, and load_vtk_results() deletes them
    automatically once each stage's result is loaded in memory (see
    _run_sfepy_input_file() / load_vtk_results()). To get a persisted file out of a
    SimulationResults object, use export_simulation_results() below --
    analogous to Download Mesh / export_mesh_results() for MeshResults, as
    its own separate Workbench component taking SimulationResults as input.

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

    try:
        pressure_sim = None
        time = None
        if builder.include_flow:
            # Stage 1: pressure
            builder.build_pressure_input_file(pressure_file)
            p_out = _run_sfepy_input_file(
                pressure_file, builder.mesh_results, mesh_type_code, output_dir=None,
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
            heat_file, builder.mesh_results, mesh_type_code, output_dir=None,
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
