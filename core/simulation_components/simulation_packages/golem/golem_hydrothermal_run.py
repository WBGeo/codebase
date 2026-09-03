"""Run a generated GOLEM problem in Docker and return its output paths."""
from __future__ import annotations

import logging
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Union

from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.object_components import MeshType, SimulationResults
from core.simulation_components.output_format.exodus.unified_format_exodus import (
    load_exodus_results,
)

from core.simulation_components.simulation_packages.golem.golem_hydrothermal_builder import (
    CustomGolemBuilder,
    GolemHydrothermalProblemBuilder,
)

logger = logging.getLogger(__name__)

MESH_TYPE_CODES = {
    MeshType.IMPLICIT: "imp",
    MeshType.STRUCTURED: "str",
    MeshType.UNSTRUCTURED: "unstr",
}


def _stream_process(command, heartbeat_seconds: float = 10.0) -> str:
    """Run GOLEM while printing output and silence-period heartbeats live."""
    print("\n" + "=" * 72, flush=True)
    print("Starting GOLEM Docker/MPI simulation", flush=True)
    print("Command:", shlex.join(command), flush=True)
    print("=" * 72, flush=True)

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    if process.stdout is None:
        raise RuntimeError("Could not capture GOLEM output")

    output_queue = queue.Queue()

    def read_output() -> None:
        try:
            for output_line in iter(process.stdout.readline, ""):
                output_queue.put(output_line)
        finally:
            process.stdout.close()
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    lines = []
    start_time = time.monotonic()
    last_activity = start_time
    while True:
        try:
            line = output_queue.get(timeout=heartbeat_seconds)
        except queue.Empty:
            elapsed = time.monotonic() - start_time
            silent = time.monotonic() - last_activity
            print(
                f"[GOLEM still running: elapsed {elapsed:.0f} s; "
                f"no new output for {silent:.0f} s]",
                flush=True,
            )
            continue

        if line is None:
            break
        print(line, end="", flush=True)
        logger.debug("GOLEM: %s", line.rstrip())
        lines.append(line)
        last_activity = time.monotonic()

    reader.join()
    return_code = process.wait()
    output = "".join(lines)
    if return_code:
        raise RuntimeError(f"GOLEM exited with status {return_code}:\n{output}")
    elapsed = time.monotonic() - start_time
    print("=" * 72, flush=True)
    print(f"GOLEM completed successfully in {elapsed:.1f} seconds", flush=True)
    print("=" * 72 + "\n", flush=True)
    return output


def run_simulation_golem(
    problem: Union[GolemHydrothermalProblemBuilder, CustomGolemBuilder],
    image_name: str,
    project_dir: str,
    docker_dir: str = "/project",
    output_dir: Optional[str] = None,
    nproc: int = 4,
    keep_files_dir: Optional[str] = None,
) -> SimulationResults:
    """Build or copy, validate and execute a GOLEM input against a WBGeo mesh.

    ``project_dir`` is bind-mounted only to expose the GOLEM executable at
    ``docker_dir/golem/golem-opt``. Input and mesh files are copied into a
    private directory in a named container, results are copied back after the
    run, and the container (including its temporary directory) is removed.
    ``problem`` may be an automatically generated hydrothermal problem or a
    ``CustomGolemBuilder`` containing complete caller-supplied GOLEM input.
    """
    if nproc < 1:
        raise ValueError("nproc must be >= 1")
    project = Path(project_dir).resolve()
    project.mkdir(parents=True, exist_ok=True)
    output = Path(output_dir).resolve() if output_dir else project / "golem_output"
    output.mkdir(parents=True, exist_ok=True)

    if keep_files_dir:
        run_dir = Path(keep_files_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary = False
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="golem_run_", dir=project))
        temporary = True

    container_name = f"wbgeo_golem_{uuid.uuid4().hex[:12]}"
    container_created = False
    try:
        mesh_path = run_dir / "wbgeo_mesh.exo"
        input_path = run_dir / "wbgeo.i"
        mesh_type_code = MESH_TYPE_CODES[problem.mesh_results.mesh_type]
        exo = export_mesh_results_to_exodus(problem.mesh_results, type=mesh_type_code)
        mesh_path.write_bytes(exo.getbuffer())
        problem.build_input_file(str(input_path))

        container_run = "/tmp/wbgeo_golem_run"
        container_mesh = f"{container_run}/wbgeo_mesh.exo"
        container_input = f"{container_run}/wbgeo.i"

        generated = input_path.read_text(encoding="utf-8")
        if "${GOLEM_MESH_FILE}" not in generated:
            raise ValueError("Generated GOLEM input has no ${GOLEM_MESH_FILE} placeholder")
        input_path.write_text(
            generated.replace("${GOLEM_MESH_FILE}", container_mesh), encoding="utf-8"
        )

        print(f"GOLEM host staging directory: {run_dir}", flush=True)
        print(f"Input to copy: {input_path}", flush=True)
        print(f"Mesh to copy: {mesh_path}", flush=True)
        print(f"Docker container: {container_name}", flush=True)
        print(f"MPI processes: {nproc}", flush=True)

        golem_command = (
            f'set -e; mkdir -p "{container_run}/output"; '
            f'cd "{container_run}"; '
            f'mpiexec --allow-run-as-root -n {nproc} '
            f'{docker_dir.rstrip("/")}/golem/golem-opt -i "{container_input}"'
        )
        create_command = [
            "docker", "create", "--name", container_name, "-i",
            "-v", f"{project}:{docker_dir}",
            image_name,
            "-c", golem_command,
        ]
        created = subprocess.run(
            create_command, text=True, capture_output=True, check=False
        )
        if created.returncode:
            raise RuntimeError(
                "Could not create the GOLEM Docker container:\n"
                + created.stdout + created.stderr
            )
        container_created = True

        # Copy the complete staging directory so the input and mesh exist in
        # the container independently of the host bind mount.
        copied = subprocess.run(
            ["docker", "cp", str(run_dir) + "/.",
             f"{container_name}:{container_run}"],
            text=True, capture_output=True, check=False,
        )
        if copied.returncode:
            raise RuntimeError(
                "Could not copy GOLEM input files into the container:\n"
                + copied.stdout + copied.stderr
            )

        transcript = _stream_process(["docker", "start", "-a", container_name])

        raw_output = run_dir / "container_output"
        raw_output.mkdir(exist_ok=True)
        copied_back = subprocess.run(
            ["docker", "cp", f"{container_name}:{container_run}/output/.",
             str(raw_output)],
            text=True, capture_output=True, check=False,
        )
        if copied_back.returncode:
            raise RuntimeError(
                "GOLEM completed but its output could not be copied from Docker:\n"
                + copied_back.stdout + copied_back.stderr
            )
        if not raw_output.is_dir():
            raise RuntimeError(f"GOLEM completed but did not create {raw_output}")
        shutil.copytree(raw_output, output, dirs_exist_ok=True)
        candidates = list(output.rglob("*.e")) + list(output.rglob("*.exo"))
        if not candidates:
            raise RuntimeError(f"GOLEM completed but no Exodus result exists in {output}")
        result_file = max(candidates, key=lambda path: path.stat().st_mtime)

        transcript_path = output / "golem_stdout.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        if temporary:
            # Preserve the generated inputs beside results before cleanup.
            saved_input = output / "wbgeo_generated.i"
            saved_mesh = output / "wbgeo_mesh.exo"
            shutil.copy2(input_path, saved_input)
            shutil.copy2(mesh_path, saved_mesh)
        else:
            saved_input, saved_mesh = input_path, mesh_path

        run_info = {
            "input_file": str(saved_input),
            "mesh_file": str(saved_mesh),
            "output_file": str(result_file),
            "output_dir": str(output),
            "stdout_file": str(transcript_path),
        }
        print(f"GOLEM Exodus result: {result_file}", flush=True)
        print("Loading GOLEM Exodus result into SimulationResults...", flush=True)
        simulation_results = load_exodus_results(run_info)
        if not simulation_results.nodes_by_time:
            raise RuntimeError(
                f"The GOLEM Exodus file at {result_file} contained no readable time steps."
            )
        print(
            f"Loaded {len(simulation_results.nodes_by_time)} GOLEM time step(s).",
            flush=True,
        )
        return simulation_results
    finally:
        if container_created:
            removed = subprocess.run(
                ["docker", "rm", "-f", container_name],
                text=True, capture_output=True, check=False,
            )
            if removed.returncode:
                logger.warning(
                    "Could not remove temporary GOLEM container %s: %s",
                    container_name,
                    (removed.stdout + removed.stderr).strip(),
                )
        if temporary:
            shutil.rmtree(run_dir, ignore_errors=True)
