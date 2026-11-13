"""
GolemSimpleReservoirProblemBuilder on structural models of different size.

The doublet layout keeps its physical size, the refinement around it stays
in metres, the background mesh size follows the model, and a layout that
doesn't fit (with its refinement zone) is rejected naming the axis. Meshing
is replaced by a small synthetic mesh, so only the layout logic runs.
"""
import copy

import numpy as np
import pytest

import core.simulation_components.simulation_packages.golem.golem_model_builder as golem_builder
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemSimpleReservoirProblemBuilder,
)
from core.simulation_components.simulation_workbench_components import build_golem_simple_reservoir_problem
from test_golem_model_builder import _synthetic_unstructured_mesh


@pytest.fixture
def fake_mesher(monkeypatch, model1_implicit_mesh):
    calls = []
    fake_mesh = _synthetic_unstructured_mesh(model1_implicit_mesh)

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_mesh

    monkeypatch.setattr(golem_builder, "create_unstructured_mesh_data", fake)
    return calls


def _model_with_extent(model, extent):
    resized = copy.deepcopy(model)
    resized.structural_frame.grid.extent = tuple(float(v) for v in extent)
    return resized


def _planes(builder):
    return [np.asarray(p, dtype=float).reshape(4, 3) for p in builder.fracture_planes]


@pytest.mark.parametrize("extent, mesh_size", [
    ((0, 1000, 0, 1000, 0, 1000), 30.0),            # model 1
    ((0, 2500, 0, 1000, 0, 1000), 30.0),            # model 2
    ((0, 10000, 0, 10000, 0, 10000), 300.0),        # large model
    ((5000, 7000, -300, 900, -2000, -500), 36.0),   # off-origin, smallest axis 1200 m
])
def test_layout_keeps_physical_size_and_mesh_follows_model(fake_mesher, model1_structural_result, extent, mesh_size):
    builder = GolemSimpleReservoirProblemBuilder(
        geomodel_result=_model_with_extent(model1_structural_result, extent))
    center = np.array([(extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2, (extent[4] + extent[5]) / 2])

    first, second = _planes(builder)
    for plane in (first, second):
        assert np.ptp(plane[:, 1]) == pytest.approx(100.0) and np.ptp(plane[:, 2]) == pytest.approx(100.0)
        np.testing.assert_allclose(plane.mean(axis=0)[1:], center[1:])
    assert second[0, 0] - first[0, 0] == pytest.approx(200.0)
    assert (first[0, 0] + second[0, 0]) / 2 == pytest.approx(center[0])
    np.testing.assert_allclose(builder.injection_point, first.mean(axis=0))
    np.testing.assert_allclose(builder.extraction_point, second.mean(axis=0))
    # The well keeps its Model-1 offsets relative to the second plane.
    well = np.asarray(builder.wells[0]).reshape(-1, 3)
    np.testing.assert_allclose(well[2], second.mean(axis=0))
    assert np.ptp(well[:, 0]) == pytest.approx(50.0)

    assert builder.mesh_size == pytest.approx(mesh_size)
    assert fake_mesher[-1]["mesh_size"] == pytest.approx(mesh_size)
    assert builder.refinement.wells.SizeMin == 5.0 and builder.refinement.extra_planes.DistMax == 60.0


def test_model1_layout_is_unchanged(fake_mesher, model1_structural_result):
    builder = GolemSimpleReservoirProblemBuilder(geomodel_result=model1_structural_result)
    np.testing.assert_allclose(builder.fracture_planes[0], GolemSimpleReservoirProblemBuilder.REFERENCE_FRACTURE_PLANES[0])
    np.testing.assert_allclose(builder.fracture_planes[1], GolemSimpleReservoirProblemBuilder.REFERENCE_FRACTURE_PLANES[1])
    np.testing.assert_allclose(builder.wells[0], GolemSimpleReservoirProblemBuilder.REFERENCE_WELLS[0])


def test_too_small_model_is_rejected_naming_the_axis(fake_mesher, model1_structural_result):
    small = _model_with_extent(model1_structural_result, (0, 300, 0, 300, 0, 300))
    with pytest.raises(ValueError, match=r"does not fit into the model: x: .*reservoir is 0\.\.300") as error:
        GolemSimpleReservoirProblemBuilder(geomodel_result=small)
    assert "y:" not in str(error.value) and "z:" not in str(error.value)
    assert not fake_mesher  # rejected before meshing


def test_custom_size_spacing_and_center(fake_mesher, model1_structural_result):
    builder = GolemSimpleReservoirProblemBuilder(
        geomodel_result=model1_structural_result,
        fracture_size=50.0, fracture_spacing=300.0, layout_center=(450.0, 400.0, 600.0))
    first, second = _planes(builder)
    assert first[0, 0] == pytest.approx(300.0) and second[0, 0] == pytest.approx(600.0)
    assert np.ptp(first[:, 1]) == pytest.approx(50.0)
    np.testing.assert_allclose(builder.injection_point, (300.0, 400.0, 600.0))
    np.testing.assert_allclose(fake_mesher[-1]["extra_planes"][1], builder.fracture_planes[1])


def test_scale_refinement_with_model_restores_old_scaling(fake_mesher, model1_structural_result):
    large = _model_with_extent(model1_structural_result, (0, 10000, 0, 10000, 0, 10000))
    builder = GolemSimpleReservoirProblemBuilder(geomodel_result=large, scale_refinement_with_model=True)
    assert builder.refinement.wells.SizeMin == pytest.approx(50.0)


def test_hierarchical_wells_from_the_workbench_are_rejected(fake_mesher, model1_structural_result):
    with pytest.raises(ValueError, match="hierarchical"):
        build_golem_simple_reservoir_problem(geomodel_result=model1_structural_result,
                                             wells={0: [(500.0, 500.0, 400.0, 500.0, 500.0, 600.0)]})


def test_workbench_node_forwards_layout_inputs(fake_mesher, model1_structural_result):
    well = (560.0, 500.0, 480.0, 600.0, 500.0, 500.0, 640.0, 500.0, 520.0)
    builder = build_golem_simple_reservoir_problem(
        geomodel_result=model1_structural_result,
        fracture_size=80.0, fracture_spacing=250.0, layout_center=(500.0, 500.0, 500.0),
        injection_point=(375.0, 500.0, 500.0), extraction_point=(625.0, 500.0, 500.0),
        wells=[well],
    ).inner()
    assert builder.fracture_size == 80.0 and builder.fracture_spacing == 250.0
    assert builder.injection_point == (375.0, 500.0, 500.0)
    assert builder.wells == [well]
    assert builder.mesh_size == pytest.approx(30.0)  # node default now follows the model
