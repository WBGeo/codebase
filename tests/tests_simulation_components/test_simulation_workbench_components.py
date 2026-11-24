import json
import os
import pathlib
import re
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless -- these tests only check the inspector's
                       # pre-plot logic (final_time/origin), not rendering

import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    RockUnitProperties, FluidProperties, SfepyHydrothermalProblemBuilder, SfepyCustomBuilder, SfepyProblem,
    map_mat_id_to_lithology,
)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemCustomBuilder, GolemFaultProperties, GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirBuilder,
)
import core.simulation_components.simulation_workbench_components as swc
from core.simulation_components.simulation_workbench_components import (
    HydrothermalOptions, HydrothermalOptions_Root,
    build_hydrothermal_problem, hydrothermal_problem_smart_options,
    hydrothermal_problem_smart_options_to_data, run_simulation,
    build_custom_sfepy_problem,
    export_simulation_results, _DEFAULT_FAULT_ZONE_PROPERTIES,
    build_golem_hydrothermal_problem, build_golem_simple_reservoir_problem,
    build_custom_golem_problem, run_golem_simulation, GolemProblem,
    GolemHydrothermalOptions, GolemHydrothermalOptions_Root,
    golem_hydrothermal_problem_smart_options,
    golem_hydrothermal_problem_smart_options_to_data,
    _DEFAULT_GOLEM_FAULT_PROPERTIES,
)


# -----------------------------------------------------------------------------
# build_hydrothermal_problem -- options application / fault-zone default
# -----------------------------------------------------------------------------

def test_build_hydrothermal_problem_without_options_uses_builder_defaults(
    model1_implicit_mesh, model1_structural_result
):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    assert isinstance(problem, SfepyProblem)
    builder = problem.inner()
    assert set(builder.rock_properties.keys()) == {"basement", "rock1", "rock2"}
    assert builder.fluid == FluidProperties()
    assert builder.fault_zone_properties is None


def test_build_hydrothermal_problem_applies_options(model1_implicit_mesh, model1_structural_result):
    options = HydrothermalOptions(root=HydrothermalOptions_Root(
        fluid=FluidProperties(mu=5e-4),
        rock_properties={"rock1": RockUnitProperties(name="rock1", porosity=0.33)},
    ))
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, options=options,
    ).inner()
    assert problem.fluid.mu == pytest.approx(5e-4)
    assert problem.rock_properties["rock1"].porosity == pytest.approx(0.33)
    # basement/rock2 not given in options -> still filled with defaults
    assert "basement" in problem.rock_properties


def test_build_hydrothermal_problem_forwards_shared_time_and_initial_controls(
    model1_implicit_mesh, model1_structural_result
):
    """Same input names as Build GOLEM Hydrothermal Problem for the shared physics."""
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        start_time=10.0, end_time=1010.0, num_steps=4,
        initial_temperature=20.0, print_linear_residuals=True,
    ).inner()
    assert problem.start_time == pytest.approx(10.0)
    assert problem.end_time == pytest.approx(1010.0)
    assert problem.num_steps == 4
    assert problem.initial_temperature == pytest.approx(20.0)
    assert problem.print_linear_residuals is True


def test_enable_fault_zone_false_ignores_options_fault_zone_properties(
    model1_implicit_mesh, model1_structural_result
):
    options = HydrothermalOptions(root=HydrothermalOptions_Root(
        fault_zone_properties=RockUnitProperties(name="custom_fault", k_solid=999),
    ))
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        options=options, enable_fault_zone=False,
    ).inner()
    assert problem.fault_zone_properties is None


def test_enable_fault_zone_true_without_options_uses_shared_default(
    model1_implicit_mesh, model1_structural_result
):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        enable_fault_zone=True,
    ).inner()
    assert problem.fault_zone_properties == _DEFAULT_FAULT_ZONE_PROPERTIES


def test_enable_fault_zone_true_with_options_uses_options_value(model1_implicit_mesh, model1_structural_result):
    custom = RockUnitProperties(name="custom_fault", k_solid=999, porosity=0.01)
    options = HydrothermalOptions(root=HydrothermalOptions_Root(fault_zone_properties=custom))
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        options=options, enable_fault_zone=True,
    ).inner()
    assert problem.fault_zone_properties == custom


def test_build_hydrothermal_problem_rejects_mesh_with_no_lithology_mapping(
    model1_implicit_mesh, model1_structural_result
):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})
    with pytest.raises(ValueError, match="no lithology mapping"):
        build_hydrothermal_problem(mesh_results=bad_mesh, geomodel_result=model1_structural_result)


# -----------------------------------------------------------------------------
# SmartInput sidebar: form building + parsing round trip
# -----------------------------------------------------------------------------

def test_smart_options_form_enumerates_rock_units(model1_structural_result):
    ctrl_group = hydrothermal_problem_smart_options(geomodel_result=model1_structural_result)
    rock_group = next(c for c in ctrl_group.inner if getattr(c, "id", None) == "rock_properties")
    assert {g.id for g in rock_group.inner} == {"basement", "rock1", "rock2"}


def test_smart_options_to_data_roundtrip():
    form_submission = {
        "root.fluid.mu": 5e-4,
        "root.rock_properties.rock1.name": "rock1",
        "root.rock_properties.rock1.porosity": 0.25,
        "root.fault_zone_properties.name": "fault_zone",
        "root.fault_zone_properties.porosity": 0.02,
    }
    options = hydrothermal_problem_smart_options_to_data(_input=json.dumps(form_submission))
    assert isinstance(options, HydrothermalOptions)
    assert options.root.fluid.mu == pytest.approx(5e-4)
    assert options.root.rock_properties["rock1"].porosity == pytest.approx(0.25)
    assert options.root.fault_zone_properties.porosity == pytest.approx(0.02)


# -----------------------------------------------------------------------------
# build_custom_sfepy_problem -- reads/decodes input_file, a RemoteFile-
# controlled path (relative to the codebase root), not an uploaded byte
# stream -- see CustomSfepyFileDataType's own docstring for why. The file
# needs to live on the same drive as the repo root: os.path.relpath() can't
# express a path across two different Windows drives, and pytest's own
# tmp_path fixture lives under the system temp dir, which isn't guaranteed
# to share a drive with the repo (it doesn't on this project's dev machine,
# where the repo root is on D: but the system temp dir stays on C:) -- see
# repo_tmp_path below.
# -----------------------------------------------------------------------------

def _repo_root() -> pathlib.Path:
    return pathlib.Path(swc.__file__).parent.parent.parent.resolve()


@pytest.fixture
def repo_tmp_path():
    """
    Like pytest's own tmp_path, but guaranteed to be on the same drive as
    the repo root -- needed so os.path.relpath(file_path, _repo_root())
    below can actually produce a relative path (see the module-level note
    above this fixture for why tmp_path itself isn't safe to use here).
    """
    d = tempfile.mkdtemp(dir=_repo_root())
    try:
        yield pathlib.Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_build_custom_sfepy_problem_reads_and_decodes_file(
    repo_tmp_path, model1_implicit_mesh, model1_structural_result
):
    valid = map_mat_id_to_lithology(model1_implicit_mesh)
    text = "\n".join(f"'Omega{m}': 'cells of group {m}'," for m in valid)
    file_path = repo_tmp_path / "custom.py"
    file_path.write_text(text, encoding="utf-8")

    problem = build_custom_sfepy_problem(
        input_file=os.path.relpath(file_path, _repo_root()),
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    assert isinstance(problem, SfepyProblem)
    assert isinstance(problem.custom, SfepyCustomBuilder)
    assert problem.custom.input_file_contents == text


def test_build_custom_sfepy_problem_rejects_mesh_with_no_lithology_mapping(
    repo_tmp_path, model1_implicit_mesh, model1_structural_result
):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})
    file_path = repo_tmp_path / "custom.py"
    file_path.write_text("'Omega0': 'cells of group 0',", encoding="utf-8")
    with pytest.raises(ValueError, match="no lithology mapping"):
        build_custom_sfepy_problem(
            input_file=os.path.relpath(file_path, _repo_root()),
            mesh_results=bad_mesh, geomodel_result=model1_structural_result,
        )


def test_build_custom_sfepy_problem_rejects_non_utf8_file(
    repo_tmp_path, model1_implicit_mesh, model1_structural_result
):
    file_path = repo_tmp_path / "custom.py"
    file_path.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        build_custom_sfepy_problem(
            input_file=os.path.relpath(file_path, _repo_root()),
            mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        )


def test_build_custom_sfepy_problem_rejects_missing_file(model1_implicit_mesh, model1_structural_result):
    with pytest.raises(ValueError, match="not found"):
        build_custom_sfepy_problem(
            input_file="examples/own_data/does_not_exist.py",
            mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        )


# -----------------------------------------------------------------------------
# Thin wrappers -- call-through, without a real solve
# -----------------------------------------------------------------------------

def test_run_simulation_unwraps_hydrothermal_problem(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run(inner):
        received["inner"] = inner
        return sentinel_result

    monkeypatch.setattr(swc, "_run_simulation_sfepy", fake_run)
    fake_builder = SfepyHydrothermalProblemBuilder.__new__(SfepyHydrothermalProblemBuilder)
    problem = SfepyProblem.__new__(SfepyProblem)
    problem.hydrothermal, problem.custom = fake_builder, None

    result = run_simulation(problem)
    assert result is sentinel_result
    assert received["inner"] is fake_builder


def test_run_simulation_unwraps_custom_sfepy_problem(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run(inner):
        received["inner"] = inner
        return sentinel_result

    monkeypatch.setattr(swc, "_run_simulation_sfepy", fake_run)
    fake_builder = SfepyCustomBuilder.__new__(SfepyCustomBuilder)
    problem = SfepyProblem.__new__(SfepyProblem)
    problem.hydrothermal, problem.custom = None, fake_builder

    result = run_simulation(problem)
    assert result is sentinel_result
    assert received["inner"] is fake_builder


def test_export_simulation_results_calls_through(monkeypatch):
    sentinel_buf = object()
    received = {}

    def fake_export(sim, fmt):
        received["sim"], received["format"] = sim, fmt
        return sentinel_buf

    monkeypatch.setattr(swc, "_export_simulation_results", fake_export)
    sim = SimulationResults()
    result = export_simulation_results(sim, format="vtk")
    assert result is sentinel_buf
    assert received["sim"] is sim
    assert received["format"] == "vtk"



# -----------------------------------------------------------------------------
# GOLEM Workbench components -- same pattern as the SfePy Workbench tests.
# These tests never start Docker, GOLEM, MPI, or Gmsh.
# -----------------------------------------------------------------------------

def test_golem_smart_options_form_enumerates_rock_units(model1_structural_result):
    ctrl_group = golem_hydrothermal_problem_smart_options(
        geomodel_result=model1_structural_result
    )
    rock_group = next(
        c for c in ctrl_group.inner if getattr(c, "id", None) == "rock_properties"
    )
    assert {g.id for g in rock_group.inner} == {"basement", "rock1", "rock2"}

    fault_group = next(
        c for c in ctrl_group.inner if getattr(c, "id", None) == "fault_properties"
    )
    assert fault_group.id == "fault_properties"


def test_golem_smart_options_to_data_roundtrip():
    form_submission = {
        "root.fluid.mu": 7.5e-4,
        "root.rock_properties.rock1.name": "rock1",
        "root.rock_properties.rock1.porosity": 0.21,
        "root.fault_properties.name": "main_fault",
        "root.fault_properties.porosity": 0.8,
        "root.fault_properties.aperture": 0.025,
    }

    options = golem_hydrothermal_problem_smart_options_to_data(
        _input=json.dumps(form_submission)
    )

    assert isinstance(options, GolemHydrothermalOptions)
    assert options.root.fluid.mu == pytest.approx(7.5e-4)
    assert options.root.rock_properties["rock1"].porosity == pytest.approx(0.21)
    assert options.root.fault_properties.name == "main_fault"
    assert options.root.fault_properties.aperture == pytest.approx(0.025)


def test_build_golem_hydrothermal_problem_without_options_uses_defaults(
    model1_implicit_mesh, model1_structural_result
):
    problem = build_golem_hydrothermal_problem(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )

    assert isinstance(problem, GolemProblem)
    assert problem.hydrothermal is not None
    assert problem.simple_reservoir is None
    assert problem.custom is None

    builder = problem.inner()
    assert isinstance(builder, GolemHydrothermalProblemBuilder)
    assert builder.fluid == FluidProperties()
    assert set(builder.rock_properties) == {"basement", "rock1", "rock2"}
    # The Workbench supplies one default template even for a fault-free mesh.
    assert _DEFAULT_GOLEM_FAULT_PROPERTIES.name == "fault"


def test_build_golem_hydrothermal_problem_applies_smart_options(
    model1_implicit_mesh, model1_structural_result
):
    fault = GolemFaultProperties(
        name="fault_template",
        porosity=0.7,
        permeability=2.0e-12,
        aperture=0.03,
    )
    options = GolemHydrothermalOptions(
        root=GolemHydrothermalOptions_Root(
            fluid=FluidProperties(mu=8.0e-4),
            rock_properties={
                "rock1": RockUnitProperties(name="rock1", porosity=0.31)
            },
            fault_properties=fault,
        )
    )

    builder = build_golem_hydrothermal_problem(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        options=options,
        p_bottom=2.5e6,
        t_bottom=80.0,
        end_time=14400.0,
        num_steps=4,
        output_interval=2,
        print_linear_residuals=False,
    ).inner()

    assert isinstance(builder, GolemHydrothermalProblemBuilder)
    assert builder.mesh_results is model1_implicit_mesh
    assert builder.geomodel_result is model1_structural_result
    assert builder.fluid.mu == pytest.approx(8.0e-4)
    assert builder.rock_properties["rock1"].porosity == pytest.approx(0.31)
    assert builder.p_bottom == pytest.approx(2.5e6)
    assert builder.t_bottom == pytest.approx(80.0)
    assert builder.dt == pytest.approx(3600.0)
    assert builder.num_steps == 4
    assert builder.output_interval == 2
    assert builder.print_linear_residuals is False


def test_build_golem_hydrothermal_problem_rejects_missing_lithology_mapping(
    model1_implicit_mesh, model1_structural_result
):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})

    with pytest.raises(ValueError, match="no lithology mapping"):
        build_golem_hydrothermal_problem(
            mesh_results=bad_mesh,
            geomodel_result=model1_structural_result,
        )


def test_build_golem_simple_reservoir_problem_forwards_without_real_meshing(
    monkeypatch, model1_structural_result
):
    # Use an instance of the real expected type so GolemProblem's Pydantic
    # validation succeeds, but bypass __init__/__post_init__ so no Gmsh or
    # reservoir mesh generation occurs.
    sentinel = GolemSimpleReservoirBuilder.__new__(
        GolemSimpleReservoirBuilder
    )
    received = {}

    def fake_builder(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(swc, "GolemSimpleReservoirBuilder", fake_builder)

    result = build_golem_simple_reservoir_problem(
        geomodel_result=model1_structural_result,
        mesh_size=20.0,
        curve_mesh_size=3.0,
        mapping_litho="automatic_centers",
        initial_pressure=1.2e7,
        initial_temperature=145.0,
        boundary_pressure=1.1e7,
        boundary_temperature=140.0,
        injection_temperature=65.0,
        injection_rate=4.5,
        extraction_rate=4.0,
        start_time=10.0,
        end_time=1000.0,
    )

    assert isinstance(result, GolemProblem)
    assert result.hydrothermal is None
    assert result.simple_reservoir is sentinel
    assert result.custom is None
    assert result.inner() is sentinel

    assert received["geomodel_result"] is model1_structural_result
    assert received["mesh_size"] == pytest.approx(20.0)
    assert received["curve_mesh_size"] == pytest.approx(3.0)
    assert received["mapping_litho"] == "automatic_centers"
    assert received["initial_pressure"] == pytest.approx(1.2e7)
    assert received["initial_temperature"] == pytest.approx(145.0)
    assert received["boundary_pressure"] == pytest.approx(1.1e7)
    assert received["boundary_temperature"] == pytest.approx(140.0)
    assert received["injection_temperature"] == pytest.approx(65.0)
    assert received["injection_rate"] == pytest.approx(4.5)
    assert received["extraction_rate"] == pytest.approx(4.0)
    assert received["start_time"] == pytest.approx(10.0)
    assert received["end_time"] == pytest.approx(1000.0)


def test_build_custom_golem_problem_reads_input_file(
    repo_tmp_path, model1_implicit_mesh, model1_structural_result
):
    text = """[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
[]
[Outputs]
[]
"""
    file_path = repo_tmp_path / "custom_golem.i"
    file_path.write_text(text, encoding="utf-8")

    problem = build_custom_golem_problem(
        input_file=os.path.relpath(file_path, _repo_root()),
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        print_linear_residuals=False,
    )

    assert isinstance(problem, GolemProblem)
    assert problem.hydrothermal is None
    assert problem.simple_reservoir is None
    assert problem.custom is not None

    builder = problem.inner()
    assert isinstance(builder, GolemCustomBuilder)
    assert builder.input_file_contents == text
    assert builder.mesh_results is model1_implicit_mesh
    assert builder.geomodel_result is model1_structural_result
    assert builder.print_linear_residuals is False


def test_build_custom_golem_problem_rejects_missing_placeholder(
    repo_tmp_path, model1_implicit_mesh
):
    file_path = repo_tmp_path / "bad_custom_golem.i"
    file_path.write_text("[Mesh]\n[]\n[Outputs]\n[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="GOLEM_MESH_FILE"):
        build_custom_golem_problem(
            input_file=os.path.relpath(file_path, _repo_root()),
            mesh_results=model1_implicit_mesh,
        )


def test_build_custom_golem_problem_rejects_non_utf8_file(
    repo_tmp_path, model1_implicit_mesh
):
    file_path = repo_tmp_path / "custom_golem_bad_encoding.i"
    file_path.write_bytes(b"\xff\xfe\x00\x01not utf-8")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        build_custom_golem_problem(
            input_file=os.path.relpath(file_path, _repo_root()),
            mesh_results=model1_implicit_mesh,
        )


def test_build_custom_golem_problem_rejects_missing_file(model1_implicit_mesh):
    with pytest.raises(ValueError, match="not found"):
        build_custom_golem_problem(
            input_file="examples/own_data/does_not_exist.i",
            mesh_results=model1_implicit_mesh,
        )


def test_golem_problem_accepts_each_of_the_three_problem_kinds():
    fake_hydro = GolemHydrothermalProblemBuilder.__new__(
        GolemHydrothermalProblemBuilder
    )
    fake_simple = GolemSimpleReservoirBuilder.__new__(
        GolemSimpleReservoirBuilder
    )
    fake_custom = GolemCustomBuilder.__new__(GolemCustomBuilder)

    hydro = GolemProblem(hydrothermal=fake_hydro)
    simple = GolemProblem(simple_reservoir=fake_simple)
    custom = GolemProblem(custom=fake_custom)

    assert hydro.inner() is fake_hydro
    assert simple.inner() is fake_simple
    assert custom.inner() is fake_custom


def test_golem_problem_requires_exactly_one_inner_problem():
    with pytest.raises(ValueError, match="exactly one"):
        GolemProblem()

    fake_hydro = GolemHydrothermalProblemBuilder.__new__(
        GolemHydrothermalProblemBuilder
    )
    fake_simple = GolemSimpleReservoirBuilder.__new__(
        GolemSimpleReservoirBuilder
    )
    with pytest.raises(ValueError, match="exactly one"):
        GolemProblem(
            hydrothermal=fake_hydro,
            simple_reservoir=fake_simple,
        )


@pytest.mark.parametrize(
    "kind",
    ["hydrothermal", "simple_reservoir", "custom"],
)
def test_run_golem_simulation_unwraps_all_problem_types_without_running_docker(
    monkeypatch, kind
):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return sentinel_result

    monkeypatch.setattr(swc, "_run_simulation_golem", fake_run)

    if kind == "hydrothermal":
        fake_inner = GolemHydrothermalProblemBuilder.__new__(
            GolemHydrothermalProblemBuilder
        )
        problem = GolemProblem(hydrothermal=fake_inner)
    elif kind == "simple_reservoir":
        fake_inner = GolemSimpleReservoirBuilder.__new__(
            GolemSimpleReservoirBuilder
        )
        problem = GolemProblem(simple_reservoir=fake_inner)
    else:
        fake_inner = GolemCustomBuilder.__new__(GolemCustomBuilder)
        problem = GolemProblem(custom=fake_inner)

    result = run_golem_simulation(
        problem=problem,
        nproc=6,
    )

    assert result is sentinel_result
    assert received["problem"] is fake_inner
    assert received["image_name"] == os.environ.get("GOLEM_IMAGE")
    assert received["golem_binary"] == os.environ.get("GOLEM_BINARY", "/golem/golem-opt")
    assert received["workspace_dir"].startswith(os.environ.get("GOLEM_WORKSPACE", "golem_workspace") + "/")
    assert received["temporary"] is True
    # The runner now supplies an absolute, unique per-run output directory.
    output_dir = pathlib.Path(received["output_dir"])
    assert output_dir.is_absolute()
    assert output_dir.parent.name == "golem_output"
    assert re.fullmatch(r"[0-9a-f]{32}", output_dir.name)
    assert received["nproc"] == 6


@pytest.mark.parametrize("fails", [False, True])
def test_run_golem_simulation_deletes_output_folder_right_away(monkeypatch, tmp_path, fails):
    """The result is self-contained, so GOLEM's output folder goes immediately (X4)."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_run(**kwargs):
        output_dir = pathlib.Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "wbgeo.e").write_bytes(b"result")
        seen["output_dir"] = output_dir
        if fails:
            raise RuntimeError("GOLEM failed")
        return SimulationResults(solver="golem")

    monkeypatch.setattr(swc, "_run_simulation_golem", fake_run)
    problem = GolemProblem(simple_reservoir=GolemSimpleReservoirBuilder.__new__(
        GolemSimpleReservoirBuilder))
    if fails:
        with pytest.raises(RuntimeError, match="GOLEM failed"):
            run_golem_simulation(problem=problem)
    else:
        assert run_golem_simulation(problem=problem).solver == "golem"
    assert not seen["output_dir"].exists()


def test_run_golem_simulation_uses_runner_defaults_without_running_docker(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return sentinel_result

    monkeypatch.setattr(swc, "_run_simulation_golem", fake_run)

    fake_inner = GolemSimpleReservoirBuilder.__new__(
        GolemSimpleReservoirBuilder
    )
    problem = GolemProblem(simple_reservoir=fake_inner)

    result = run_golem_simulation(problem=problem)

    assert result is sentinel_result
    assert received["problem"] is fake_inner
    assert received["image_name"] == os.environ.get("GOLEM_IMAGE")
    assert received["golem_binary"] == os.environ.get("GOLEM_BINARY", "/golem/golem-opt")
    assert received["workspace_dir"].startswith(os.environ.get("GOLEM_WORKSPACE", "golem_workspace") + "/")
    assert received["temporary"] is True
    # The runner now supplies an absolute, unique per-run output directory.
    output_dir = pathlib.Path(received["output_dir"])
    assert output_dir.is_absolute()
    assert output_dir.parent.name == "golem_output"
    assert re.fullmatch(r"[0-9a-f]{32}", output_dir.name)
    assert received["nproc"] == 4


# -----------------------------------------------------------------------------
# Inspector components -- only the pre-plot logic (final_time/origin
# computation) and correct forwarding to the underlying plot_* function, not
# the actual rendering, matching this codebase's existing convention of not
# unit-testing plot output (see test_simulation_visualization.py). The real
# plot_variable_at_a_time/plot_cross_section_2D are monkeypatched out rather
# than actually called: plot_variable_at_a_time has no off_screen/show_plotter
# support (unlike plot_builder_materials), so calling it for real here would
# pop up a real interactive PyVista window needing a manual close, with no
# guarantee of safe headless behavior in CI either. The mesh-centroid origin
# fallback is a regression test: plot_cross_section_2D's own default
# origin=(0,0,0) sits on the domain boundary and misses cell-centered mesh
# nodes entirely.
# -----------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_inspect_plot_variable_at_a_time_uses_final_time(monkeypatch, model1_implicit_mesh, model1_structural_result):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, end_time=5e9,
    )
    sim = run_simulation(problem)

    received = {}

    def fake_plot(simulation_result, **kwargs):
        received["simulation_result"] = simulation_result
        received.update(kwargs)

    monkeypatch.setattr(swc, "plot_variable_at_a_time", fake_plot)
    swc.inspect_simulation_result_plot_variable_at_a_time(simulation_result=sim)

    assert received["simulation_result"] is sim
    assert received["cmap"] == "coolwarm"
    # show_edges=False and a capped off-screen/no-antialiasing render:
    # this is the only 3D plot in the Workbench colored by a continuous
    # scalar field with a colorbar (vs. every other 3D plot's flat
    # per-block colors) -- confirmed disproportionately expensive under
    # the Workbench container's software rendering, unlike those.
    assert received["show_edges"] is False
    assert received["off_screen"] is True
    assert received["window_size"] == (800, 600)
    assert received["anti_aliasing"] is False
    # The inspector forwards the final saved time to the plotting function.
    assert received["time"] == pytest.approx(max(sim.nodes_by_time))


@pytest.mark.integration
@pytest.mark.slow
def test_inspect_plot_cross_section_2D_uses_mesh_centroid_as_origin(
    monkeypatch, model1_implicit_mesh, model1_structural_result
):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, end_time=5e9,
    )
    sim = run_simulation(problem)
    first_time = min(sim.nodes_by_time.keys())
    expected_origin = tuple(sim.nodes_by_time[first_time].mean(axis=0))

    received = {}

    def fake_plot(simulation_result, **kwargs):
        received["simulation_result"] = simulation_result
        received.update(kwargs)

    monkeypatch.setattr(swc, "plot_cross_section_2D", fake_plot)
    swc.inspect_simulation_result_plot_cross_section_2D(simulation_result=sim, _inspector=None)

    assert received["simulation_result"] is sim
    assert received["origin"] == pytest.approx(expected_origin)
    assert received["cmap"] == "coolwarm"


def _offset_simulation_result():
    """Two saved steps on a small mesh in real-world coordinates, far from (500, 500, 500)."""
    nodes = np.array([[3.2e5, 5.6e6, -800.0], [3.3e5, 5.6e6, -800.0],
                      [3.2e5, 5.7e6, -800.0], [3.2e5, 5.6e6, -200.0]])
    sim = SimulationResults()
    for t in (0.0, 1.0):
        sim.nodes_by_time[t] = nodes
        sim.node_data_by_time[t] = {"T": np.full(len(nodes), 10.0 + t)}
    return sim


def test_inspect_plot_cross_section_2D_defaults_to_centroid_for_real_coordinates(monkeypatch):
    sim = _offset_simulation_result()
    received = {}
    monkeypatch.setattr(swc, "plot_cross_section_2D", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_cross_section_2D(simulation_result=sim, _inspector=None)

    assert received["origin"] == pytest.approx(tuple(sim.nodes_by_time[0.0].mean(axis=0)))
    assert received["normal"] == (0.0, 1.0, 0.0)


@pytest.mark.parametrize("axis, normal", [("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))])
def test_inspect_plot_cross_section_2D_axis_and_position(monkeypatch, axis, normal):
    sim = _offset_simulation_result()
    centroid = sim.nodes_by_time[0.0].mean(axis=0)
    position = {"x": 3.25e5, "y": 5.65e6, "z": -500.0}[axis]
    received = {}
    monkeypatch.setattr(swc, "plot_cross_section_2D", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_cross_section_2D(simulation_result=sim, _inspector=None,
                                                        axis=axis, position=position)

    expected_origin = list(centroid)
    expected_origin["xyz".index(axis)] = position
    assert received["normal"] == normal
    assert received["origin"] == pytest.approx(tuple(expected_origin))


def test_inspect_plot_cross_section_2D_rejects_position_outside_the_mesh(monkeypatch):
    monkeypatch.setattr(swc, "plot_cross_section_2D", lambda simulation_result, **kw: None)
    with pytest.raises(ValueError, match="along z it must be between -800 and -200"):
        swc.inspect_simulation_result_plot_cross_section_2D(
            simulation_result=_offset_simulation_result(), _inspector=None, axis="z", position=100.0)


def test_saved_time_for_step_lists_the_saved_times():
    with pytest.raises(ValueError, match=r"step 0: 0, step 1: 1"):
        swc._saved_time_for_step(_offset_simulation_result(), 5)


def test_line_and_time_series_inspectors_no_longer_expose_internal_settings():
    import inspect as _inspect
    assert "n_samples" not in _inspect.signature(swc.inspect_simulation_result_plot_variable_along_line).parameters
    assert "decimals" not in _inspect.signature(swc.inspect_simulation_result_plot_variable_time_series).parameters


def test_inspect_plot_variable_time_series_defaults_to_centroid(monkeypatch):
    # An empty Point input arrives as None; it used to reach PyVista and fail
    # with "cannot reshape array of size 1 into shape (3)".
    sim = _offset_simulation_result()
    received = {}
    monkeypatch.setattr(swc, "plot_variable_time_series", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_variable_time_series(simulation_result=sim, _inspector=None)

    assert received["point"] == pytest.approx(tuple(sim.nodes_by_time[0.0].mean(axis=0)))


def test_inspect_plot_variable_time_series_uses_explicit_point(monkeypatch):
    sim = _offset_simulation_result()
    received = {}
    monkeypatch.setattr(swc, "plot_variable_time_series", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_variable_time_series(
        simulation_result=sim, _inspector=None, point=(3.25e5, 5.65e6, -500.0))

    assert received["point"] == (3.25e5, 5.65e6, -500.0)


def test_plot_variable_time_series_rejects_missing_point():
    from core.simulation_components.simulation_visualization.simulation_visualization import (
        plot_variable_time_series,
    )
    with pytest.raises(ValueError, match="three coordinates"):
        plot_variable_time_series(_offset_simulation_result(), "T", point=None)


def _cube_simulation_result():
    """Tetrahedral cube mesh over [100, 900]^3 with three saved steps; T rises with depth."""
    import pyvista as pv
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(200, 200, 200), origin=(100, 100, 100))
    grid = grid.cast_to_unstructured_grid().triangulate()
    nodes = np.asarray(grid.points)
    sim = SimulationResults()
    for t in (0.0, 10.0, 20.0):
        sim.nodes_by_time[t] = nodes
        sim.cells_by_time[t] = np.asarray(grid.cells)
        sim.celltypes_by_time[t] = np.asarray(grid.celltypes)
        sim.node_data_by_time[t] = {"T": 60.0 - nodes[:, 2] * 0.05 + t}
    return sim


def test_plot_variable_along_line_shows_gaps_outside_the_mesh(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    lines = []
    monkeypatch.setattr(sv.plt, "show", lambda: lines.append(sv.plt.gca().lines[0].get_ydata()))

    # Line runs from z=0 to z=1000, the mesh only covers z=100..900.
    sv.plot_variable_along_line(_cube_simulation_result(), "T", 20.0, (500, 500, 0), (500, 500, 1000), n_samples=100)

    values = np.asarray(lines[0], dtype=float)
    assert np.isnan(values[0]) and np.isnan(values[-1])
    inside = values[~np.isnan(values)]
    assert inside.min() > 30.0  # no false zeros: T stays between 35 and 75 inside the mesh


def test_plot_variable_along_line_rejects_line_outside_the_mesh(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    monkeypatch.setattr(sv.plt, "show", lambda: None)
    with pytest.raises(RuntimeError, match="entirely outside the mesh"):
        sv.plot_variable_along_line(_cube_simulation_result(), "T", 20.0, (5000, 0, 0), (6000, 0, 0))


def test_inspect_plot_variable_along_line_defaults(monkeypatch):
    # Empty inputs: last saved step and a vertical line through the mesh centre.
    sim = _cube_simulation_result()
    received = {}
    monkeypatch.setattr(swc, "plot_variable_along_line", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_variable_along_line(simulation_result=sim, _inspector=None)

    assert received["time"] == 20.0
    assert received["p0"] == pytest.approx((500.0, 500.0, 100.0))
    assert received["p1"] == pytest.approx((500.0, 500.0, 900.0))


def test_inspect_plot_variable_along_line_keeps_explicit_inputs(monkeypatch):
    sim = _cube_simulation_result()
    received = {}
    monkeypatch.setattr(swc, "plot_variable_along_line", lambda simulation_result, **kw: received.update(kw))

    swc.inspect_simulation_result_plot_variable_along_line(
        simulation_result=sim, _inspector=None, time_step=0, p0=(0, 500, 500), p1=(1000, 500, 500))

    assert received["time"] == 0.0
    assert received["p0"] == (0, 500, 500)
    assert received["p1"] == (1000, 500, 500)


def test_inspect_plot_variable_along_line_runs_with_empty_inputs(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    monkeypatch.setattr(sv.plt, "show", lambda: sv.plt.close("all"))
    swc.inspect_simulation_result_plot_variable_along_line(simulation_result=_cube_simulation_result(), _inspector=None)


def test_plot_variable_time_series_rejects_point_outside_the_mesh(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    monkeypatch.setattr(sv.plt, "show", lambda: None)
    with pytest.raises(ValueError, match="lies outside the mesh"):
        sv.plot_variable_time_series(_cube_simulation_result(), "T", point=(5000, 500, 500))


def test_plot_variable_time_series_title_rounds_the_point(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    titles = []
    monkeypatch.setattr(sv.plt, "show", lambda: titles.append(sv.plt.gca().get_title()))

    sv.plot_variable_time_series(_cube_simulation_result(), "T", point=(500.123456789, 500.0, 499.987654321))

    assert "(500.123, 500, 499.988)" in titles[0]
    sv.plt.close("all")


def test_plot_variable_time_series_uses_whole_step_ticks(monkeypatch):
    from core.simulation_components.simulation_visualization import simulation_visualization as sv
    ticks = []
    monkeypatch.setattr(sv.plt, "show", lambda: ticks.append(sv.plt.gca().get_xticks()))

    sv.plot_variable_time_series(_cube_simulation_result(), "T", point=(500, 500, 500))

    assert all(float(t).is_integer() for t in ticks[0])
    sv.plt.close("all")


def test_inspect_simulation_result_stdout_prints_each_stage(capsys):
    sim = SimulationResults()
    sim.solver_output = {"pressure": "pressure solver log", "heat": "heat solver log"}

    swc.inspect_simulation_result_stdout(simulation_result=sim, _inspector=None)

    captured = capsys.readouterr().out
    assert "pressure" in captured and "pressure solver log" in captured
    assert "heat" in captured and "heat solver log" in captured


def test_inspect_simulation_result_stdout_handles_missing_stdout(capsys):
    sim = SimulationResults()  # solver_output left empty

    swc.inspect_simulation_result_stdout(simulation_result=sim, _inspector=None)

    assert "No solver output" in capsys.readouterr().out


def test_sfepy_hydrothermal_node_keeps_input_positions_of_saved_pipelines():
    """Saved Workbench pipelines (e.g. examples/.../WBGeo1.0_model1.json) store
    input values by position, so existing inputs must keep their index and new
    inputs may only be appended."""
    import inspect
    func = getattr(build_hydrothermal_problem, "__wrapped__", build_hydrothermal_problem)
    names = list(inspect.signature(func).parameters)
    assert names[:16] == [
        "mesh_results", "geomodel_result", "options", "include_flow", "enable_fault_zone",
        "fault_zone_n_voxels", "start_time", "end_time", "num_steps", "p_top", "p_bottom",
        "t_top", "t_bottom", "linear_solver", "linear_solver_i_max", "linear_solver_eps_r",
    ]
