"""Run a generated GOLEM problem in Docker and return its output paths."""
from __future__ import annotations

import logging
import queue
import re
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

from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
    GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
)

logger = logging.getLogger(__name__)

MESH_TYPE_CODES = {
    MeshType.IMPLICIT: "imp",
    MeshType.STRUCTURED: "str",
    MeshType.UNSTRUCTURED: "unstr",
}


def validate_golem_input(content: str, mesh_results, mesh_type: MeshType) -> None:
    """Validate a GOLEM/MOOSE input before starting Docker.

    This mirrors the structural checks used by ``simulation_run`` while
    accepting both modern ``[name]`` and legacy ``[./name]`` MOOSE subsection
    syntax, which is emitted by the GOLEM builders in this module family.
    """
    required_blocks = [
        "Mesh",
        "Variables",
        "Kernels",
        "Executioner",
        "Outputs",
        "BCs",
        "Materials",
    ]
    missing = [
        block
        for block in required_blocks
        if not re.search(rf"^\s*\[{re.escape(block)}\]\s*$", content, re.MULTILINE)
    ]
    if missing:
        raise ValueError("Missing required blocks: " + ", ".join(missing))

    mesh_pattern = re.compile(
        r"""
        ^\s*\[Mesh\]\s*$
        .*?
        ^\s*type\s*=\s*FileMeshGenerator\s*$
        .*?
        ^\s*file\s*=\s*\$\{GOLEM_MESH(?:1|2)?_FILE\}\s*$
        """,
        re.DOTALL | re.MULTILINE | re.VERBOSE,
    )
    if not mesh_pattern.search(content):
        raise ValueError(
            "Mesh block must contain type = FileMeshGenerator and "
            "file = ${GOLEM_MESH_FILE}, ${GOLEM_MESH1_FILE}, or ${GOLEM_MESH2_FILE}"
        )

    def extract_parent_block(text: str, parent_block: str) -> str:
        lines = text.splitlines()
        inside_parent = False
        depth = 0
        collected = []
        parent_start = f"[{parent_block}]"

        for line in lines:
            stripped = line.strip()
            if not inside_parent:
                if stripped == parent_start:
                    inside_parent = True
                continue

            if stripped == "[]":
                if depth == 0:
                    break
                depth -= 1
                collected.append(line)
                continue

            if stripped == "[../]":
                if depth > 0:
                    depth -= 1
                collected.append(line)
                continue

            if re.match(r"^\[(?:\./)?[A-Za-z_][A-Za-z0-9_]*\]$", stripped):
                depth += 1
            collected.append(line)

        return "\n".join(collected)

    def get_direct_child_blocks(text: str, parent_block: str):
        lines = text.splitlines()
        inside_parent = False
        depth = 0
        children = []
        parent_start = f"[{parent_block}]"

        for line in lines:
            stripped = line.strip()
            if not inside_parent:
                if stripped == parent_start:
                    inside_parent = True
                continue

            if stripped == "[]":
                if depth == 0:
                    break
                depth -= 1
                continue

            if stripped == "[../]":
                if depth > 0:
                    depth -= 1
                continue

            # Legacy [./name] is always a subsection.  Modern [name] is a
            # direct child only while we are inside the parent; once a direct
            # child has closed and another known top-level block starts, the
            # parent has ended (some custom MOOSE inputs omit the parent's []).
            legacy_match = re.match(
                r"^\[\./([A-Za-z_][A-Za-z0-9_]*)\]$", stripped
            )
            if legacy_match:
                if depth == 0:
                    children.append(legacy_match.group(1))
                depth += 1
                continue

            match = re.match(r"^\[([A-Za-z_][A-Za-z0-9_]*)\]$", stripped)
            if match:
                name = match.group(1)
                if depth == 0 and name in {
                    "Mesh", "Variables", "AuxVariables", "Problem", "Kernels",
                    "AuxKernels", "Materials", "BCs", "Executioner", "Outputs",
                    "UserObjects", "Functions", "Postprocessors", "VectorPostprocessors",
                    "Preconditioning", "Controls", "Debug",
                }:
                    break
                if depth == 0:
                    children.append(name)
                depth += 1

        return children

    materials_content = extract_parent_block(content, "Materials")
    if not materials_content.strip():
        raise ValueError("Materials block is empty")

    material_sections = get_direct_child_blocks(content, "Materials")
    if not material_sections:
        raise ValueError("No material definitions found inside the [Materials] block")

    if mesh_type == MeshType.UNSTRUCTURED:
        n_mesh_blocks = len(mesh_results.elements) - 6
    else:
        n_mesh_blocks = len(mesh_results.elements)

    if n_mesh_blocks <= 0:
        raise ValueError("The mesh has no exportable material blocks")

    # A material section does not have to correspond one-to-one with a mesh
    # block.  GolemSimpleReservoirProblemBuilder intentionally groups several
    # mesh blocks into sections such as [uniform_rock], [fractures], and [well].
    # Validate the actual block assignments instead of counting subsections.
    # Material assignments may use either numeric Exodus block IDs or names
    # introduced by a MOOSE RenameBlockGenerator.  Resolve renamed blocks back
    # to their original numeric Exodus IDs so the coverage check below remains
    # valid for both forms.
    rename_name_to_id = {}
    rename_sections = re.finditer(
        r"type\s*=\s*RenameBlockGenerator(?P<body>.*?)(?=^\s*\[\]\s*$)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    for section in rename_sections:
        body = section.group("body")
        old_match = re.search(
            r"old_block\s*=\s*(['\"])(.*?)\1", body, re.DOTALL
        )
        new_match = re.search(
            r"new_block\s*=\s*(['\"])(.*?)\1", body, re.DOTALL
        )
        if not old_match or not new_match:
            continue
        old_tokens = old_match.group(2).split()
        new_tokens = new_match.group(2).split()
        if len(old_tokens) != len(new_tokens):
            raise ValueError(
                "RenameBlockGenerator must contain the same number of "
                f"old_block and new_block entries ({len(old_tokens)} != "
                f"{len(new_tokens)})"
            )
        for old_token, new_name in zip(old_tokens, new_tokens):
            try:
                old_id = int(old_token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid old_block id {old_token!r} in RenameBlockGenerator"
                ) from exc
            if new_name in rename_name_to_id and rename_name_to_id[new_name] != old_id:
                raise ValueError(
                    f"RenameBlockGenerator assigns duplicate new block name "
                    f"{new_name!r} to multiple block IDs"
                )
            rename_name_to_id[new_name] = old_id

    material_block_ids = set()
    for match in re.finditer(
        r"^\s*block\s*=\s*['\"]?([^'\"#\n]+)['\"]?\s*(?:#.*)?$",
        materials_content,
        re.MULTILINE,
    ):
        for token in match.group(1).split():
            try:
                material_block_ids.add(int(token))
            except ValueError:
                if token not in rename_name_to_id:
                    raise ValueError(
                        f"Unknown material block name {token!r} in [Materials]; "
                        "it is neither a numeric Exodus block ID nor a name "
                        "defined by RenameBlockGenerator"
                    )
                material_block_ids.add(rename_name_to_id[token])

    if not material_block_ids:
        raise ValueError(
            "No material 'block = ...' assignments found inside [Materials]"
        )

    expected_block_ids = set(range(1, n_mesh_blocks + 1))
    missing_material_ids = sorted(expected_block_ids - material_block_ids)
    extra_material_ids = sorted(material_block_ids - expected_block_ids)

    if missing_material_ids:
        raise ValueError(
            "The following mesh blocks have no material assignment: "
            + ", ".join(map(str, missing_material_ids))
        )
    if extra_material_ids:
        raise ValueError(
            "The [Materials] block references non-exportable mesh block ids: "
            + ", ".join(map(str, extra_material_ids))
        )

    variable_names = get_direct_child_blocks(content, "Variables")
    if not variable_names:
        raise ValueError("No variables found inside the [Variables] block")

    kernels_content = extract_parent_block(content, "Kernels")
    if not kernels_content.strip():
        raise ValueError("Kernels block is empty")

    kernel_variables = set(
        re.findall(
            r"^\s*variable\s*=\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$",
            kernels_content,
            re.MULTILINE,
        )
    )
    variables_without_kernel = [
        variable for variable in variable_names if variable not in kernel_variables
    ]
    if variables_without_kernel:
        raise ValueError(
            "The following variables do not have at least one kernel: "
            + ", ".join(variables_without_kernel)
        )

    logger.info(
        "GOLEM input validation successful: %d mesh blocks covered by %d material sections; variables=%s",
        n_mesh_blocks,
        len(material_sections),
        ", ".join(variable_names),
    )


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
    problem: Union[
        GolemHydrothermalProblemBuilder,
        GolemSimpleReservoirProblemBuilder,
        CustomGolemBuilder,
    ],
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
    ``problem`` may be an automatically generated hydrothermal/simple-reservoir
    problem or a ``CustomGolemBuilder`` containing complete caller-supplied
    GOLEM input.
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
        input_path = run_dir / "wbgeo.i"
        problem.build_input_file(str(input_path))

        container_run = "/tmp/wbgeo_golem_run"
        container_input = f"{container_run}/wbgeo.i"
        generated = input_path.read_text(encoding="utf-8")

        if isinstance(problem, CustomGolemBuilder):
            referenced_meshes = problem.referenced_meshes()
        else:
            referenced_meshes = {"${GOLEM_MESH_FILE}": problem.mesh_results}

        mesh_specs = {
            "${GOLEM_MESH_FILE}": ("wbgeo_mesh.exo", f"{container_run}/wbgeo_mesh.exo"),
            "${GOLEM_MESH1_FILE}": ("wbgeo_mesh1.exo", f"{container_run}/wbgeo_mesh1.exo"),
            "${GOLEM_MESH2_FILE}": ("wbgeo_mesh2.exo", f"{container_run}/wbgeo_mesh2.exo"),
        }
        mesh_paths = {}
        for placeholder, mesh in referenced_meshes.items():
            filename, container_mesh = mesh_specs[placeholder]
            mesh_path = run_dir / filename
            mesh_type_code = MESH_TYPE_CODES[mesh.mesh_type]
            exo = export_mesh_results_to_exodus(mesh, type=mesh_type_code)
            mesh_path.write_bytes(exo.getbuffer())
            mesh_paths[placeholder] = mesh_path
            generated = generated.replace(placeholder, container_mesh)

        # validate against any referenced mesh; structural validation of the
        # Mesh block is independent of which hierarchy level it represents.
        validation_mesh = next(iter(referenced_meshes.values()))
        validate_golem_input(generated if False else input_path.read_text(encoding="utf-8"),
                             validation_mesh, validation_mesh.mesh_type)
        input_path.write_text(generated, encoding="utf-8")

        # Legacy variable retained for the rest of the runner and run_info.
        # For hierarchical-only custom inputs it points to mesh1 when present,
        # otherwise mesh2.
        mesh_path = (mesh_paths.get("${GOLEM_MESH_FILE}")
                     or mesh_paths.get("${GOLEM_MESH1_FILE}")
                     or mesh_paths.get("${GOLEM_MESH2_FILE}"))

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

        # Copy the ENTIRE GOLEM working directory back from the container.
        # Some MOOSE outputs (especially CSV when no file_base is given) are
        # written directly beside wbgeo.i rather than inside output/.
        container_snapshot = run_dir / "container_run"
        container_snapshot.mkdir(exist_ok=True)

        copied_back = subprocess.run(
            ["docker", "cp", f"{container_name}:{container_run}/.",
             str(container_snapshot)],
            text=True, capture_output=True, check=False,
        )
        if copied_back.returncode:
            raise RuntimeError(
                "GOLEM completed but its working directory could not be copied "
                "from Docker:\n" + copied_back.stdout + copied_back.stderr
            )

        # First preserve everything explicitly written under output/ at the
        # root of the caller's output_dir.
        staged_output = container_snapshot / "output"
        if staged_output.is_dir():
            shutil.copytree(staged_output, output, dirs_exist_ok=True)

        # Then preserve generated files/directories written directly in the
        # GOLEM working directory. Do not re-copy the staged input/mesh or the
        # output directory itself; those inputs are saved below with stable
        # WBGeo names.
        excluded_root_names = {
            "wbgeo.i", "wbgeo_mesh.exo", "wbgeo_mesh1.exo", "wbgeo_mesh2.exo", "output"
        }
        for item in container_snapshot.iterdir():
            if item.name in excluded_root_names:
                continue
            destination = output / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, destination)

        # GOLEM/MOOSE may create recovery/checkpoint directories such as
        # ``wbgeo_out_cp`` during long runs. These are internal restart data,
        # not user-facing WBGeo simulation results, so remove them from the
        # final output directory for all GOLEM problem types (hydrothermal,
        # simple reservoir, and custom).
        checkpoint_dirs = sorted(
            (
                path
                for path in output.rglob("*")
                if path.is_dir() and path.name.endswith("_cp")
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for checkpoint_dir in checkpoint_dirs:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        # The Docker working directory has now been fully copied into the
        # caller's output directory (with internal *_cp checkpoint folders
        # filtered out). Remove the temporary host-side snapshot immediately
        # instead of keeping a duplicate until the end of the run.
        shutil.rmtree(container_snapshot, ignore_errors=True)

        copied_files = sorted(
            (path for path in output.rglob("*") if path.is_file()),
            key=lambda path: str(path),
        )
        print(
            f"Copied {len(copied_files)} GOLEM output file(s) to {output}",
            flush=True,
        )
        for path in copied_files:
            print(f"  {path.relative_to(output)}", flush=True)

        # Exodus is still required to load the numerical result into WBGeo,
        # but the search below does not restrict which files are preserved.
        candidates = list(output.rglob("*.e")) + list(output.rglob("*.exo"))
        if not candidates:
            copied_listing = "\n".join(
                f"  {path.relative_to(output)}" for path in copied_files
            ) or "  <none>"
            raise RuntimeError(
                "GOLEM completed and its output files were copied, but no "
                f"Exodus result exists in {output}. Copied files:\n"
                + copied_listing
            )
        result_file = max(candidates, key=lambda path: path.stat().st_mtime)

        transcript_path = output / "golem_stdout.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        if temporary:
            # Preserve the generated inputs beside results before cleanup.
            saved_input = output / "wbgeo_generated.i"
            shutil.copy2(input_path, saved_input)
            saved_mesh_paths = {}
            for placeholder, source_path in mesh_paths.items():
                destination = output / source_path.name
                shutil.copy2(source_path, destination)
                saved_mesh_paths[placeholder] = destination
            saved_mesh = (saved_mesh_paths.get("${GOLEM_MESH_FILE}")
                          or saved_mesh_paths.get("${GOLEM_MESH1_FILE}")
                          or saved_mesh_paths.get("${GOLEM_MESH2_FILE}"))
        else:
            saved_input = input_path
            saved_mesh_paths = dict(mesh_paths)
            saved_mesh = mesh_path

        run_info = {
            "input_file": str(saved_input),
            "mesh_file": str(saved_mesh),
            "mesh1_file": str(saved_mesh_paths["${GOLEM_MESH1_FILE}"]) if "${GOLEM_MESH1_FILE}" in saved_mesh_paths else None,
            "mesh2_file": str(saved_mesh_paths["${GOLEM_MESH2_FILE}"]) if "${GOLEM_MESH2_FILE}" in saved_mesh_paths else None,
            "output_file": str(result_file),
            "output_dir": str(output),
            "stdout_file": str(transcript_path),
        }
        print(f"GOLEM Exodus result: {result_file}", flush=True)
        print("Loading GOLEM Exodus result into SimulationResults...", flush=True)
        simulation_results = load_exodus_results(run_info)
        # Keep the native GOLEM Exodus result path on the unified result object.
        # The generic simulation exporter uses this path for format="exodus" so
        # GOLEM's original multi-time-step wbgeo.e is returned unchanged instead
        # of reconstructing one Exodus file per time step.
        simulation_results.source_exodus_file = str(result_file)
        simulation_results.source_output_dir = str(output)
        simulation_results.source_solver = "golem"
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
