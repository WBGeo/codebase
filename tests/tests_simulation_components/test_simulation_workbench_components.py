import json

import matplotlib
matplotlib.use("Agg")  # headless -- these tests only check the inspector's
                       # pre-plot logic (final_time/origin), not rendering

import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    RockUnitProperties, FluidProperties,
)
import core.simulation_components.simulation_workbench_components as swc
from core.simulation_components.simulation_workbench_components import (
    HydrothermalOptions, HydrothermalOptions_Root,
    build_hydrothermal_problem, hydrothermal_problem_smart_options,
    hydrothermal_problem_smart_options_to_data, run_hydrothermal_simulation,
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
    assert set(problem.rock_properties.keys()) == {"basement", "rock1", "rock2"}
    assert problem.fluid == FluidProperties()
    assert problem.fault_zone_properties is None


def test_build_hydrothermal_problem_applies_options(model1_implicit_mesh, model1_structural_result):
    options = HydrothermalOptions(root=HydrothermalOptions_Root(
        fluid=FluidProperties(mu=5e-4),
        rock_properties={"rock1": RockUnitProperties(name="rock1", porosity=0.33)},
    ))
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, options=options,
    )
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
    )
    assert problem.fault_zone_properties is None


def test_enable_fault_zone_true_without_options_uses_shared_default(
    model1_implicit_mesh, model1_structural_result
):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        enable_fault_zone=True,
    )
    assert problem.fault_zone_properties == _DEFAULT_FAULT_ZONE_PROPERTIES


def test_enable_fault_zone_true_with_options_uses_options_value(model1_implicit_mesh, model1_structural_result):
    custom = RockUnitProperties(name="custom_fault", k_solid=999, porosity=0.01)
    options = HydrothermalOptions(root=HydrothermalOptions_Root(fault_zone_properties=custom))
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        options=options, enable_fault_zone=True,
    )
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
# Thin wrappers -- call-through, without a real solve
# -----------------------------------------------------------------------------

def test_run_hydrothermal_simulation_calls_through(monkeypatch):
    sentinel_result = SimulationResults()
    received = {}

    def fake_run(builder):
        received["builder"] = builder
        return sentinel_result

    monkeypatch.setattr(swc, "_run_simulation_sfepy", fake_run)
    fake_builder = object()
    result = run_hydrothermal_simulation(fake_builder)
    assert result is sentinel_result
    assert received["builder"] is fake_builder


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
# computation), not the actual rendering, matching this codebase's existing
# convention of not unit-testing plot output. The mesh-centroid origin
# fallback is a regression test: plot_cross_section_2D's own default
# origin=(0,0,0) sits on the domain boundary and misses cell-centered mesh
# nodes entirely.
# -----------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_inspect_plot_variable_at_a_time_uses_final_time(model1_implicit_mesh, model1_structural_result):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=5e9,
    )
    sim = run_hydrothermal_simulation(problem)
    swc.inspect_simulation_result_plot_variable_at_a_time(simulation_result=sim, _inspector=None)


@pytest.mark.integration
@pytest.mark.slow
def test_inspect_plot_cross_section_2D_uses_mesh_centroid_as_origin(model1_implicit_mesh, model1_structural_result):
    problem = build_hydrothermal_problem(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=5e9,
    )
    sim = run_hydrothermal_simulation(problem)
    swc.inspect_simulation_result_plot_cross_section_2D(simulation_result=sim, _inspector=None)
