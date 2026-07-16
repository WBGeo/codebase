import zipfile

import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, RockUnitProperties,
)
import core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_run as sfepy_run
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_run import (
    _check_convergence,
    export_simulation_results,
    run_simulation_sfepy,
)


# -----------------------------------------------------------------------------
# _check_convergence -- pure stdout-parsing, no subprocess/I/O involved
# -----------------------------------------------------------------------------

def test_check_convergence_passes_for_converged_run():
    stdout = "nls: iter: 0, residual: 1.0\ncond: 0, iter: 3, err0: 1.0, err: 1e-9\n"
    _check_convergence(stdout, "dummy_input.py")  # must not raise


def test_check_convergence_raises_for_non_converged_run():
    stdout = "cond: 1, iter: 5, err0: 1568.9, err: 8.86\n"
    with pytest.raises(RuntimeError, match="did not converge"):
        _check_convergence(stdout, "dummy_input.py")


def test_check_convergence_raises_for_linesearch_failure():
    stdout = "cond: 2, iter: 5, err0: 1.0, err: 0.5\n"
    with pytest.raises(RuntimeError, match="linesearch gave up"):
        _check_convergence(stdout, "dummy_input.py")


def test_check_convergence_catches_earlier_failing_step_in_multi_step_run():
    """A transient run does one nonlinear solve per time step -- an earlier
    step failing while a later one happens to converge must not go unnoticed."""
    stdout = (
        "cond: 1, iter: 5, err0: 100.0, err: 50.0\n"  # step 0: failed
        "cond: 0, iter: 2, err0: 1.0, err: 1e-9\n"      # step 1: converged
    )
    with pytest.raises(RuntimeError, match="1 of its"):
        _check_convergence(stdout, "dummy_input.py")


def test_check_convergence_ignores_unrelated_output():
    stdout = "Setting Backend To: AvailableBackends.numpy\nsome other line\n"
    _check_convergence(stdout, "dummy_input.py")  # must not raise


# -----------------------------------------------------------------------------
# export_simulation_results -- synthetic SimulationResults, no real solve needed
# -----------------------------------------------------------------------------

def _synthetic_simulation_results(num_times=2):
    sim = SimulationResults()
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for t in range(num_times):
        sim.nodes_by_time[float(t)] = nodes
        sim.node_data_by_time[float(t)] = {"T": np.array([10.0, 20.0, 30.0]) + t}
    return sim


def test_export_simulation_results_default_format_is_vtk():
    sim = _synthetic_simulation_results()
    buf = export_simulation_results(sim)
    assert buf.filename == "simulation_results.zip"


def test_export_simulation_results_rejects_unsupported_format():
    sim = _synthetic_simulation_results()
    with pytest.raises(NotImplementedError, match="format='vtk'"):
        export_simulation_results(sim, format="exodus")


def test_export_simulation_results_zip_contains_pvd_and_one_vtk_per_step():
    sim = _synthetic_simulation_results(num_times=3)
    buf = export_simulation_results(sim)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    assert "results.pvd" in names
    assert sorted(n for n in names if n != "results.pvd") == [
        "result_0000.vtk", "result_0001.vtk", "result_0002.vtk",
    ]


def test_export_simulation_results_raises_for_no_time_steps():
    sim = SimulationResults()
    with pytest.raises(ValueError, match="no time steps"):
        export_simulation_results(sim)


# -----------------------------------------------------------------------------
# run_simulation_sfepy -- dispatch by problem type, no real solve needed
# -----------------------------------------------------------------------------

def test_run_simulation_sfepy_dispatches_hydrothermal_problem_builder(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run_hydrothermal(builder, keep_files_dir=None):
        received["builder"] = builder
        received["keep_files_dir"] = keep_files_dir
        return sentinel_result

    def fake_run_custom(problem, keep_files_dir=None):
        raise AssertionError("should not be called for a HydrothermalProblemBuilder")

    monkeypatch.setattr(sfepy_run, "_run_hydrothermal_problem", fake_run_hydrothermal)
    monkeypatch.setattr(sfepy_run, "_run_custom_sfepy_problem", fake_run_custom)

    fake_builder = HydrothermalProblemBuilder.__new__(HydrothermalProblemBuilder)
    result = run_simulation_sfepy(fake_builder, keep_files_dir="some/dir")
    assert result is sentinel_result
    assert received["builder"] is fake_builder
    assert received["keep_files_dir"] == "some/dir"


def test_run_simulation_sfepy_dispatches_custom_sfepy_builder(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run_hydrothermal(builder, keep_files_dir=None):
        raise AssertionError("should not be called for a CustomSfepyBuilder")

    def fake_run_custom(problem, keep_files_dir=None):
        received["problem"] = problem
        return sentinel_result

    monkeypatch.setattr(sfepy_run, "_run_hydrothermal_problem", fake_run_hydrothermal)
    monkeypatch.setattr(sfepy_run, "_run_custom_sfepy_problem", fake_run_custom)

    fake_problem = CustomSfepyBuilder.__new__(CustomSfepyBuilder)
    result = run_simulation_sfepy(fake_problem)
    assert result is sentinel_result
    assert received["problem"] is fake_problem


def test_run_custom_sfepy_problem_passes_fault_zone_args_through(monkeypatch, model1_implicit_mesh):
    """_run_custom_sfepy_problem must hand its problem's fault_zone_cell_mask/
    fault_group_id through to _run_sfepy_input_file, same as
    _run_hydrothermal_problem already does -- this is what actually makes
    the fault-zone relabeling take effect for a custom file."""
    received = {}

    def fake_run_sfepy_input_file(input_file, mesh_results, mesh_type_code, output_dir=None,
                                   fault_zone_cell_mask=None, fault_group_id=None):
        received["fault_zone_cell_mask"] = fault_zone_cell_mask
        received["fault_group_id"] = fault_group_id
        return {"output_dir": "unused", "is_temp": True}

    sentinel_sim = SimulationResults()
    sentinel_sim.nodes_by_time[0.0] = np.zeros((1, 3))

    monkeypatch.setattr(sfepy_run, "_run_sfepy_input_file", fake_run_sfepy_input_file)
    monkeypatch.setattr(sfepy_run, "load_vtk_results", lambda out: sentinel_sim)

    fake_mask = np.array([True, False, True])
    problem = CustomSfepyBuilder.__new__(CustomSfepyBuilder)
    problem.input_file_contents = "regions = {}"
    problem.mesh_results = model1_implicit_mesh
    problem.mesh_type = model1_implicit_mesh.mesh_type
    problem.fault_zone_cell_mask = fake_mask
    problem.fault_group_id = 42

    result = sfepy_run._run_custom_sfepy_problem(problem)
    assert result is sentinel_sim
    assert received["fault_group_id"] == 42
    assert np.array_equal(received["fault_zone_cell_mask"], fake_mask)


# -----------------------------------------------------------------------------
# run_simulation_sfepy -- real end-to-end solve (slow: launches sfepy-run)
# -----------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_run_simulation_sfepy_end_to_end_implicit_mesh(model1_implicit_mesh, model1_structural_result):
    """
    Full two-stage solve (pressure then heat) on the smallest realistic mesh.
    Checks the result is physically sensible (boundary values respected,
    temperature stays within [t_top, t_bottom]) rather than just "didn't crash".
    """
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        include_flow=True,
        t1=5e9,
        num_steps=1,
        t_top=10.0,
        t_bottom=60.0,
    )
    result = run_simulation_sfepy(builder)

    assert result.nodes_by_time  # at least one saved time step
    final_time = max(result.nodes_by_time.keys())
    T = result.node_data_by_time[final_time]["T"]
    assert T.min() >= 10.0 - 1e-6
    assert T.max() <= 60.0 + 1e-6
    p = result.node_data_by_time[final_time]["p"]
    assert p.min() >= 0.0 - 1e-6  # p_top=0.0 (default)


@pytest.mark.integration
@pytest.mark.slow
def test_run_simulation_sfepy_pure_conduction_is_static_with_homogeneous_properties(
    model1_implicit_mesh, model1_structural_result
):
    """
    include_flow=False + homogeneous rock properties across all lithologies
    -> the initial condition already IS the exact steady-state solution, so
    the domain-mean temperature must stay exactly constant across every saved
    step (a real correctness check, not just "did it run").
    """
    homogeneous = {
        name: RockUnitProperties(name=name, porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6)
        for name in ("basement", "rock1", "rock2")
    }
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        rock_properties=homogeneous,
        include_flow=False,
        num_steps=2,
    )
    result = run_simulation_sfepy(builder)
    means = [float(np.mean(data["T"])) for data in result.node_data_by_time.values()]
    assert means[0] == pytest.approx(35.0, abs=1e-6)  # exact linear T_top/T_bottom midpoint
    for m in means[1:]:
        assert m == pytest.approx(means[0], abs=1e-6)


@pytest.mark.integration
@pytest.mark.slow
def test_run_simulation_sfepy_fault_zone_with_flow_end_to_end(
    model2_implicit_mesh, model2_faulted_structural_result
):
    """
    include_flow=True + an active fault zone -- exercises
    compute_darcy_velocity's fault-zone cell-splitting for real, not just
    the generated input files' text.
    """
    fault_zone_properties = RockUnitProperties(
        name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
    )
    builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh,
        geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=fault_zone_properties,
        fault_zone_n_voxels=1,
        include_flow=True,
        num_steps=1,
    )
    assert builder.fault_zone_cell_mask is not None  # sanity: fault zone actually active

    result = run_simulation_sfepy(builder)
    final_time = max(result.nodes_by_time.keys())
    assert "T" in result.node_data_by_time[final_time]
    assert "p" in result.node_data_by_time[final_time]


@pytest.mark.integration
@pytest.mark.slow
def test_run_simulation_sfepy_custom_builder_end_to_end(model1_implicit_mesh, model1_structural_result, tmp_path):
    """
    Real end-to-end run of a CustomSfepyBuilder. Reuses
    HydrothermalProblemBuilder.build_heat_input_file's own generated text
    as the "custom" input -- an already-trusted, real SfePy file with all
    staging/sequencing self-contained (no velocity hand-off needed since
    include_flow=False), giving genuine coverage of the whole custom
    pathway (_run_custom_sfepy_problem -> _run_sfepy_input_file ->
    load_vtk_results) without hand-authoring a new SfePy file here.
    """
    homogeneous = {
        name: RockUnitProperties(name=name, porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6)
        for name in ("basement", "rock1", "rock2")
    }
    reference_builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        rock_properties=homogeneous,
        include_flow=False,
        num_steps=1,
    )
    heat_file = str(tmp_path / "heat.py")
    reference_builder.build_heat_input_file(heat_file)
    input_file_contents = open(heat_file).read()

    custom_builder = CustomSfepyBuilder(
        input_file_contents=input_file_contents,
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    result = run_simulation_sfepy(custom_builder)

    assert result.nodes_by_time  # at least one saved time step
    final_time = max(result.nodes_by_time.keys())
    T = result.node_data_by_time[final_time]["T"]
    # Same homogeneous-properties/pure-conduction setup as
    # test_run_simulation_sfepy_pure_conduction_is_static_with_homogeneous_properties:
    # the initial condition already IS the exact steady-state solution.
    assert float(T.mean()) == pytest.approx(35.0, abs=1e-6)


@pytest.mark.integration
@pytest.mark.slow
def test_run_simulation_sfepy_custom_builder_fault_zone_end_to_end(
    model2_implicit_mesh, model2_faulted_structural_result, tmp_path
):
    """
    Real end-to-end run of a CustomSfepyBuilder with fault_zone_n_voxels set,
    on a real faulted mesh -- proves the extracted
    lithology_to_group_idx/compute_fault_zone_cell_mask functions produce a
    fault_group_id that SfePy can actually resolve (not just one
    check_custom_sfepy_regions accepts). Reuses HydrothermalProblemBuilder's
    own generated Omega_fault region text as the "custom" input, same trick
    as test_run_simulation_sfepy_custom_builder_end_to_end -- homogeneous
    properties (including the fault zone) so pure conduction stays exactly
    static, same assertion pattern as
    test_run_simulation_sfepy_pure_conduction_is_static_with_homogeneous_properties.
    """
    rock_properties = {
        name: RockUnitProperties(name=name) for name in ("basement", "rock1", "rock2", "rock3", "rock4")
    }
    fault_zone_properties = RockUnitProperties(name="fault_zone")

    reference_builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh,
        geomodel_result=model2_faulted_structural_result,
        rock_properties=rock_properties,
        fault_zone_properties=fault_zone_properties,
        fault_zone_n_voxels=1,
        include_flow=False,
        num_steps=1,
    )
    heat_file = str(tmp_path / "heat.py")
    reference_builder.build_heat_input_file(heat_file)
    input_file_contents = open(heat_file).read()

    custom_builder = CustomSfepyBuilder(
        input_file_contents=input_file_contents,
        mesh_results=model2_implicit_mesh,
        geomodel_result=model2_faulted_structural_result,
        fault_zone_n_voxels=1,
    )
    # Sanity: both builders compute fault_group_id deterministically from
    # the same mesh_results, so they must agree -- if the extraction
    # refactor changed anything, this would already catch it.
    assert custom_builder.fault_group_id == reference_builder.fault_group_id

    result = run_simulation_sfepy(custom_builder)

    assert result.nodes_by_time
    final_time = max(result.nodes_by_time.keys())
    T = result.node_data_by_time[final_time]["T"]
    assert float(T.mean()) == pytest.approx(35.0, abs=1e-6)
