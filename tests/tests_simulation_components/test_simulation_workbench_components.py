import json
import os
import pathlib
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless -- these tests only check the inspector's
                       # pre-plot logic (final_time/origin), not rendering

import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    RockUnitProperties, FluidProperties, HydrothermalProblemBuilder, CustomSfepyBuilder, SfepyProblem,
    map_mat_id_to_lithology,
)
import core.simulation_components.simulation_workbench_components as swc
from core.simulation_components.simulation_workbench_components import (
    HydrothermalOptions, HydrothermalOptions_Root,
    build_hydrothermal_problem, hydrothermal_problem_smart_options,
    hydrothermal_problem_smart_options_to_data, run_simulation,
    build_custom_sfepy_problem,
    export_simulation_results, _DEFAULT_FAULT_ZONE_PROPERTIES,
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
    assert isinstance(problem.custom, CustomSfepyBuilder)
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
    fake_builder = HydrothermalProblemBuilder.__new__(HydrothermalProblemBuilder)
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
    fake_builder = CustomSfepyBuilder.__new__(CustomSfepyBuilder)
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
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=5e9,
    )
    sim = run_simulation(problem)

    received = {}

    def fake_plot(simulation_result, **kwargs):
        received["simulation_result"] = simulation_result
        received.update(kwargs)

    monkeypatch.setattr(swc, "plot_variable_at_a_time", fake_plot)
    swc.inspect_simulation_result_plot_variable_at_a_time(simulation_result=sim, _inspector=None)

    assert received["simulation_result"] is sim
    assert received["cmap"] == "coolwarm"
    assert received["show_edges"] is True
    # time/var_name intentionally left unset here -- plot_variable_at_a_time's
    # own defaults (final saved time step, "T" if present) apply, matching
    # this inspector's documented behavior.
    assert "time" not in received
    assert "var_name" not in received


@pytest.mark.integration
@pytest.mark.slow
def test_inspect_plot_cross_section_2D_uses_mesh_centroid_as_origin(
    monkeypatch, model1_implicit_mesh, model1_structural_result
):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=5e9,
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


def test_inspect_simulation_result_stdout_prints_each_stage(capsys):
    sim = SimulationResults()
    sim.sfepy_stdout = {"pressure": "pressure solver log", "heat": "heat solver log"}

    swc.inspect_simulation_result_stdout(simulation_result=sim, _inspector=None)

    captured = capsys.readouterr().out
    assert "pressure" in captured and "pressure solver log" in captured
    assert "heat" in captured and "heat solver log" in captured


def test_inspect_simulation_result_stdout_handles_missing_stdout(capsys):
    sim = SimulationResults()  # sfepy_stdout left at its default (None)

    swc.inspect_simulation_result_stdout(simulation_result=sim, _inspector=None)

    assert "No solver output" in capsys.readouterr().out
