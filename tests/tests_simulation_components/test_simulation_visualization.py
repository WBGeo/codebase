"""
Unit tests for simulation_visualization.py's pure-computation helpers only
(build_grid_from_class, _variable_label, _format_rock_properties,
_in_plane_axes) -- not the actual plot_*/render functions, matching this
codebase's existing convention of not unit-testing plotting output directly
(see e.g. tests_structural_modeling_components, which only calls
plot_structural_model_* conditionally for on-fail debugging, never asserts
on rendered output).
"""
import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import RockUnitProperties, FluidProperties
from core.simulation_components.simulation_visualization.simulation_visualization import (
    build_grid_from_class, _variable_label, _format_rock_properties, _in_plane_axes,
)


def test_variable_label_known_variables_get_units():
    assert _variable_label("T") == "Temperature (°C)"
    assert _variable_label("p") == "Pressure (Pa)"


def test_variable_label_unknown_variable_falls_back_to_raw_name():
    assert _variable_label("block_id") == "block_id"


def test_format_rock_properties_contains_all_fields():
    props = RockUnitProperties(name="rock1", porosity=0.2, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6)
    label = _format_rock_properties(props)
    assert "poro=0.2" in label
    assert "1.0e-14" in label
    assert "k_s=2.5" in label


def test_build_grid_from_class_point_cloud_when_no_cells():
    sim = SimulationResults()
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sim.nodes_by_time[0.0] = nodes
    sim.node_data_by_time[0.0] = {"T": np.array([10.0, 20.0, 30.0])}

    grid = build_grid_from_class(sim, 0.0)
    assert grid.n_points == 3
    assert np.allclose(grid.point_data["T"], [10.0, 20.0, 30.0])


def test_build_grid_from_class_raises_for_unknown_time():
    sim = SimulationResults()
    sim.nodes_by_time[0.0] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="not found"):
        build_grid_from_class(sim, 99.0)


@pytest.mark.parametrize("normal,expected_labels", [
    ((1, 0, 0), ("y", "z")),
    ((0, 1, 0), ("x", "z")),
    ((0, 0, 1), ("x", "y")),
])
def test_in_plane_axes_axis_aligned_normals(normal, expected_labels):
    u, v, labels = _in_plane_axes(normal)
    assert labels == expected_labels
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.dot(u, v), 0.0)
    assert np.isclose(np.dot(u, normal), 0.0)
    assert np.isclose(np.dot(v, normal), 0.0)


def test_in_plane_axes_arbitrary_normal_is_orthonormal():
    normal = (1.0, 1.0, 1.0)
    u, v, labels = _in_plane_axes(normal)
    n = np.array(normal) / np.linalg.norm(normal)
    assert labels == ("u", "v")
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.dot(u, v), 0.0, atol=1e-9)
    assert np.isclose(np.dot(u, n), 0.0, atol=1e-9)
    assert np.isclose(np.dot(v, n), 0.0, atol=1e-9)
