"""
Unit tests for simulation_visualization.py's pure-computation helpers only
(build_grid_from_class, _variable_label, _in_plane_axes,
_default_var_name, _resolve_var_name) -- not the actual plot_*/render functions, matching this
codebase's existing convention of not unit-testing plotting output directly
(see e.g. tests_structural_modeling_components, which only calls
plot_structural_model_* conditionally for on-fail debugging, never asserts
on rendered output).
"""
import numpy as np
import pytest

from core.object_components import SimulationResults
from core.simulation_components.simulation_visualization.simulation_visualization import (
    build_grid_from_class, _variable_label, _in_plane_axes, _default_var_name, _resolve_var_name,
)


def test_variable_label_known_variables_get_units():
    assert _variable_label("T") == "Temperature (°C)"
    assert _variable_label("p") == "Pressure (Pa)"


def test_variable_label_covers_golem_names():
    assert _variable_label("temperature") == _variable_label("T")
    assert _variable_label("pore_pressure") == _variable_label("p")


def _sim_with_fields(*names):
    sim = SimulationResults()
    sim.nodes_by_time[0.0] = np.zeros((3, 3))
    sim.node_data_by_time[0.0] = {name: np.arange(3.0) for name in names}
    return sim


@pytest.mark.parametrize("fields, expected", [
    (("p", "T"), "T"),                                   # SfePy
    (("pore_pressure", "temperature"), "temperature"),   # GOLEM, temperature not first
    (("concentration", "head"), "concentration"),        # custom builder: first field
])
def test_default_var_name_prefers_temperature(fields, expected):
    assert _default_var_name(_sim_with_fields(*fields), 0.0) == expected


@pytest.mark.parametrize("fields, requested, expected", [
    (("temperature", "pore_pressure"), "T", "temperature"),
    (("temperature", "pore_pressure"), "p", "pore_pressure"),
    (("T", "p"), "temperature", "T"),
    (("T", "p"), "pore_pressure", "p"),
    (("T", "temperature"), "temperature", "temperature"),  # existing name wins
    (("head",), "T", "T"),                                 # no alias present: unchanged
    (("head",), "head", "head"),
])
def test_resolve_var_name_maps_only_onto_existing_fields(fields, requested, expected):
    assert _resolve_var_name(_sim_with_fields(*fields), 0.0, requested) == expected


def test_resolve_var_name_none_uses_default():
    assert _resolve_var_name(_sim_with_fields("pore_pressure", "temperature"), 0.0, None) == "temperature"


def test_variable_label_unknown_variable_falls_back_to_raw_name():
    assert _variable_label("block_id") == "block_id"


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
    with pytest.raises(ValueError, match="Invalid time"):
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


# -----------------------------------------------------------------------------
# _cross_sections_on_plane -- sampling the 3D mesh directly (GitHub issue #3)
# -----------------------------------------------------------------------------

def _hex_sim(times=(0.0, 1.0), drop_corner=False):
    """Box of hexahedra on [0, 4] x [0, 3] x [0, 5]; T = (x + 2z) * (1 + t), linear in space.

    drop_corner removes the cells with x > 2 and z > 3, giving an L-shaped (concave) outline in y-planes.
    """
    import pyvista as pv
    grid = pv.ImageData(dimensions=(5, 4, 6)).cast_to_unstructured_grid()
    if drop_corner:
        centers = grid.cell_centers().points
        keep = np.flatnonzero(~((centers[:, 0] > 2) & (centers[:, 2] > 3)))
        grid = grid.extract_cells(keep)
    nodes = np.asarray(grid.points, dtype=float)
    sim = SimulationResults()
    for t in times:
        sim.nodes_by_time[t] = nodes
        sim.cells_by_time[t] = np.asarray(grid.cells, dtype=np.int64)
        sim.celltypes_by_time[t] = np.asarray(grid.celltypes, dtype=np.int64)
        sim.node_data_by_time[t] = {"T": (nodes[:, 0] + 2 * nodes[:, 2]) * (1 + t)}
    return sim


@pytest.mark.parametrize("origin", [(0.0, 1.5, 0.0), (0.0, 0.0, 0.0)])  # inside and on the boundary face y = 0
def test_cross_section_reproduces_a_linear_field_exactly(origin):
    from core.simulation_components.simulation_visualization.simulation_visualization import _cross_sections_on_plane
    sim = _hex_sim()
    u_axis, v_axis, _ = _in_plane_axes((0, 1, 0))

    UG, VG, (Za, Zb) = _cross_sections_on_plane(sim, "T", (0.0, 1.0), origin, (0, 1, 0), u_axis, v_axis, 21)

    # the grid spans the model in x (u) and z (v); every point lies in the mesh
    assert (UG.min(), UG.max(), VG.min(), VG.max()) == (0.0, 4.0, 0.0, 5.0)
    np.testing.assert_allclose(Za, UG + 2 * VG, atol=1e-9)
    np.testing.assert_allclose(Zb, 2 * (UG + 2 * VG), atol=1e-9)


def test_cross_section_leaves_points_outside_a_concave_mesh_empty():
    from core.simulation_components.simulation_visualization.simulation_visualization import _cross_sections_on_plane
    sim = _hex_sim(times=(0.0,), drop_corner=True)
    u_axis, v_axis, _ = _in_plane_axes((0, 1, 0))

    UG, VG, (Z,) = _cross_sections_on_plane(sim, "T", (0.0,), (0.0, 1.5, 0.0), (0, 1, 0), u_axis, v_axis, 41)

    missing = (UG > 2.05) & (VG > 3.05)  # the removed corner: no mesh, so no value
    assert np.isnan(Z[missing]).all()
    inside = ~((UG > 1.95) & (VG > 2.95))
    np.testing.assert_allclose(Z[inside], (UG + 2 * VG)[inside], atol=1e-9)


def test_cross_section_on_an_oblique_plane():
    from core.simulation_components.simulation_visualization.simulation_visualization import _cross_sections_on_plane
    sim = _hex_sim(times=(0.0,))
    normal = (1.0, 1.0, 0.0)
    u_axis, v_axis, _ = _in_plane_axes(normal)
    origin = (2.0, 1.5, 2.5)

    UG, VG, (Z,) = _cross_sections_on_plane(sim, "T", (0.0,), origin, normal, u_axis, v_axis, 31)

    n = np.array(normal) / np.linalg.norm(normal)
    points = UG[..., None] * u_axis + VG[..., None] * v_axis + np.dot(origin, n) * n
    inside = ~np.isnan(Z)
    assert inside.sum() > 100
    np.testing.assert_allclose(Z[inside], (points[..., 0] + 2 * points[..., 2])[inside], atol=1e-9)


def test_cross_section_raises_when_the_plane_misses_the_mesh():
    from core.simulation_components.simulation_visualization.simulation_visualization import _cross_sections_on_plane
    sim = _hex_sim(times=(0.0,))
    u_axis, v_axis, _ = _in_plane_axes((0, 1, 0))
    with pytest.raises(RuntimeError, match="does not cut the mesh"):
        _cross_sections_on_plane(sim, "T", (0.0,), (0.0, 50.0, 0.0), (0, 1, 0), u_axis, v_axis, 11)
