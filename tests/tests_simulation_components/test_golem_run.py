import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.simulation_components.simulation_packages.golem.golem_run as golem_run
from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
    GolemHydrothermalProblemBuilder,
)


def _generated_hydro_input(model1_implicit_mesh, model1_structural_result, tmp_path):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        num_steps=1,
    )
    path = tmp_path / "generated.i"
    builder.build_input_file(str(path))
    return builder, path.read_text()


# -----------------------------------------------------------------------------
# validate_golem_input -- pure text/mesh validation, no Docker
# -----------------------------------------------------------------------------

def test_validate_golem_input_accepts_current_builder_output(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder, content = _generated_hydro_input(
        model1_implicit_mesh, model1_structural_result, tmp_path
    )
    golem_run.validate_golem_input(
        content, builder.mesh_results, builder.mesh_results.mesh_type
    )


def test_validate_golem_input_rejects_missing_required_block(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder, content = _generated_hydro_input(
        model1_implicit_mesh, model1_structural_result, tmp_path
    )
    broken = content.replace("[Executioner]", "[NotExecutioner]", 1)

    with pytest.raises(ValueError, match="Missing required blocks: Executioner"):
        golem_run.validate_golem_input(
            broken, builder.mesh_results, builder.mesh_results.mesh_type
        )


def test_validate_golem_input_rejects_wrong_mesh_placeholder(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder, content = _generated_hydro_input(
        model1_implicit_mesh, model1_structural_result, tmp_path
    )
    broken = content.replace("${GOLEM_MESH_FILE}", "/tmp/hardcoded.exo", 1)

    with pytest.raises(ValueError, match="Mesh block must contain"):
        golem_run.validate_golem_input(
            broken, builder.mesh_results, builder.mesh_results.mesh_type
        )


def test_validate_golem_input_rejects_missing_material_coverage(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder, content = _generated_hydro_input(
        model1_implicit_mesh, model1_structural_result, tmp_path
    )

    # Remove one generated material's block assignment while leaving the
    # section itself intact. The validator must check coverage, not merely
    # count [Materials] subsections.
    broken = content.replace("    block = 1\n", "", 1)

    with pytest.raises(ValueError, match="no material assignment"):
        golem_run.validate_golem_input(
            broken, builder.mesh_results, builder.mesh_results.mesh_type
        )


def test_validate_golem_input_rejects_variable_without_kernel(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder, content = _generated_hydro_input(
        model1_implicit_mesh, model1_structural_result, tmp_path
    )
    broken = content.replace(
        "[Variables]",
        "[Variables]\n  [./unused_variable]\n    order = FIRST\n    family = LAGRANGE\n  [../]",
        1,
    )

    with pytest.raises(ValueError, match="unused_variable"):
        golem_run.validate_golem_input(
            broken, builder.mesh_results, builder.mesh_results.mesh_type
        )


# -----------------------------------------------------------------------------
# _stream_process -- subprocess behaviour mocked; no GOLEM executable
# -----------------------------------------------------------------------------

def test_stream_process_returns_combined_output(monkeypatch):
    class FakeStdout:
        def __init__(self):
            self._lines = iter(["first line\n", "second line\n"])

        def readline(self):
            return next(self._lines, "")

        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout()

        def wait(self):
            return 0

    monkeypatch.setattr(
        golem_run.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )

    output = golem_run._stream_process(["docker", "start", "-a", "fake"])
    assert output == "first line\nsecond line\n"


def test_stream_process_raises_with_transcript_on_failure(monkeypatch):
    class FakeStdout:
        def __init__(self):
            self._lines = iter(["GOLEM error\n"])

        def readline(self):
            return next(self._lines, "")

        def close(self):
            pass

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self):
            return 17

    monkeypatch.setattr(
        golem_run.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )

    with pytest.raises(RuntimeError, match=r"(?s)status 17.*GOLEM error"):
        golem_run._stream_process(["docker", "start", "-a", "fake"])


# -----------------------------------------------------------------------------
# run_simulation_golem -- orchestration with Docker/GOLEM/Exodus fully mocked
# -----------------------------------------------------------------------------

def test_run_simulation_golem_rejects_invalid_nproc(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    with pytest.raises(ValueError, match="nproc must be >= 1"):
        golem_run.run_simulation_golem(
            builder,
            image_name="unused:image",
            project_dir=str(tmp_path),
            nproc=0,
        )


def test_run_simulation_golem_stages_commands_and_loads_result_without_docker(
    monkeypatch, model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        num_steps=1,
    )

    # Exodus export is replaced by an in-memory fake file.
    fake_exodus = io.BytesIO(b"fake-exodus")
    monkeypatch.setattr(
        golem_run,
        "export_mesh_results_to_exodus",
        lambda mesh, type: fake_exodus,
    )

    # GOLEM itself is never launched.
    streamed = {}
    def fake_stream(command):
        streamed["command"] = command
        return "fake GOLEM stdout\n"
    monkeypatch.setattr(golem_run, "_stream_process", fake_stream)

    commands = []

    def fake_subprocess_run(command, text=True, capture_output=True, check=False):
        commands.append(command)

        # The second docker-cp copies the container working directory back.
        # Materialize a fake result there so the real runner's output discovery,
        # transcript writing, preservation, and loader call are exercised.
        if (
            len(command) >= 4
            and command[:2] == ["docker", "cp"]
            and ":" in command[2]
        ):
            destination = Path(command[3])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "output").mkdir(exist_ok=True)
            (destination / "output" / "wbgeo.e").write_bytes(b"result")
            # Simulate MOOSE recovery/checkpoint output produced by a long
            # hydrothermal run. The runner must not expose it to output_dir.
            cp_dir = destination / "output" / "wbgeo_out_cp"
            cp_dir.mkdir(exist_ok=True)
            (cp_dir / "checkpoint_mesh.cpr").write_bytes(b"checkpoint")
            (destination / "diagnostic.csv").write_text("x,y\n1,2\n")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(golem_run.subprocess, "run", fake_subprocess_run)

    loaded = {}
    result = SimulationResults()
    result.nodes_by_time[0.0] = model1_implicit_mesh.nodes

    def fake_load_exodus_results(run_info):
        loaded.update(run_info)
        # The temporary Docker working-directory snapshot must be removed
        # immediately after its contents have been copied to output_dir.
        project_root = tmp_path / "project"
        assert not list(project_root.rglob("container_run"))
        return result

    monkeypatch.setattr(golem_run, "load_exodus_results", fake_load_exodus_results)

    output_dir = tmp_path / "results"
    returned = golem_run.run_simulation_golem(
        builder,
        image_name="registry.example/golem:test",
        project_dir=str(tmp_path / "project"),
        docker_dir="/project",
        output_dir=str(output_dir),
        nproc=3,
    )

    assert returned is result

    # docker create/cp/cp/rm were constructed, but our fake handled all of them.
    create_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "create"]]
    assert len(create_commands) == 1
    create = create_commands[0]
    assert "registry.example/golem:test" in create
    assert any(
        isinstance(part, str) and "mpiexec --allow-run-as-root -n 3" in part
        for part in create
    )

    assert streamed["command"][0:3] == ["docker", "start", "-a"]
    assert any(cmd[:3] == ["docker", "rm", "-f"] for cmd in commands)

    # Real staging/output-preservation code still ran.
    assert (output_dir / "golem_stdout.txt").read_text() == "fake GOLEM stdout\n"
    assert (output_dir / "diagnostic.csv").exists()
    assert not (output_dir / "wbgeo_out_cp").exists()
    assert not any(
        path.is_dir() and path.name.endswith("_cp")
        for path in output_dir.rglob("*")
    )
    assert (output_dir / "wbgeo_generated.i").exists()
    assert (output_dir / "wbgeo_mesh.exo").read_bytes() == b"fake-exodus"

    generated = (output_dir / "wbgeo_generated.i").read_text()
    assert "${GOLEM_MESH_FILE}" not in generated
    assert "/tmp/wbgeo_golem_run/wbgeo_mesh.exo" in generated

    assert loaded["output_file"].endswith("wbgeo.e")
    assert loaded["stdout_file"].endswith("golem_stdout.txt")


def test_run_simulation_golem_create_failure_is_reported_without_starting_golem(
    monkeypatch, model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )

    fake_exodus = io.BytesIO(b"fake-exodus")
    monkeypatch.setattr(
        golem_run,
        "export_mesh_results_to_exodus",
        lambda mesh, type: fake_exodus,
    )

    monkeypatch.setattr(
        golem_run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=125,
            stdout="",
            stderr="image not found",
        ),
    )

    def must_not_stream(command):
        raise AssertionError("GOLEM must not start when docker create fails")

    monkeypatch.setattr(golem_run, "_stream_process", must_not_stream)

    with pytest.raises(RuntimeError, match=r"(?s)Could not create.*image not found"):
        golem_run.run_simulation_golem(
            builder,
            image_name="missing:image",
            project_dir=str(tmp_path),
        )


def test_run_simulation_golem_custom_problem_removes_checkpoint_directory(
    monkeypatch, model1_implicit_mesh, tmp_path
):
    """Custom GOLEM output must not expose MOOSE *_cp checkpoint directories."""
    custom_input = """[Mesh]
  type = FileMeshGenerator
  file = ${GOLEM_MESH_FILE}
[]
[Variables]
  [pore_pressure]
  []
[]
[Kernels]
  [k]
    variable = pore_pressure
  []
[]
[Executioner]
  type = Transient
[]
[Outputs]
  [exo]
    type = Exodus
    file_base = 'output/wbgeo'
  []
[]
[BCs]
  [dummy]
    variable = pore_pressure
    boundary = top
    value = 0
  []
[]
[Materials]
  [m]
    block = 1
  []
[]
"""

    builder = CustomGolemBuilder(
        input_file_contents=custom_input,
        mesh_results=model1_implicit_mesh,
    )

    fake_exodus = io.BytesIO(b"fake-exodus")
    monkeypatch.setattr(
        golem_run,
        "export_mesh_results_to_exodus",
        lambda mesh, type: fake_exodus,
    )

    monkeypatch.setattr(
        golem_run,
        "_stream_process",
        lambda command: "fake GOLEM stdout\n",
    )

    commands = []

    def fake_subprocess_run(command, text=True, capture_output=True, check=False):
        commands.append(command)

        if (
            len(command) >= 4
            and command[:2] == ["docker", "cp"]
            and ":" in command[2]
        ):
            destination = Path(command[3])
            destination.mkdir(parents=True, exist_ok=True)

            out_dir = destination / "output"
            out_dir.mkdir(exist_ok=True)
            (out_dir / "wbgeo.e").write_bytes(b"result")

            cp_dir = out_dir / "wbgeo_out_cp"
            cp_dir.mkdir(exist_ok=True)
            (cp_dir / "restart.cpr").write_bytes(b"checkpoint")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(golem_run.subprocess, "run", fake_subprocess_run)

    result = SimulationResults()
    result.nodes_by_time[0.0] = model1_implicit_mesh.nodes
    monkeypatch.setattr(
        golem_run,
        "load_exodus_results",
        lambda run_info: result,
    )

    output_dir = tmp_path / "custom_results"
    returned = golem_run.run_simulation_golem(
        builder,
        image_name="registry.example/golem:test",
        project_dir=str(tmp_path / "project"),
        output_dir=str(output_dir),
    )

    assert returned is result
    assert (output_dir / "wbgeo.e").exists()
    assert not (output_dir / "wbgeo_out_cp").exists()
    assert not any(
        path.is_dir() and path.name.endswith("_cp")
        for path in output_dir.rglob("*")
    )
