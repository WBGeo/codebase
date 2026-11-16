"""
plot_builder_materials for every builder type (rendered off screen, never
shown) plus the two custom-input parsers it relies on. Checks what the
preview assigns to cells (values, colours, legend), not pixels.
"""
import os

import numpy as np
import pytest
import pyvista as pv
from matplotlib.colors import to_rgb

import core.simulation_components.simulation_packages.golem.golem_model_builder as golem_builder
from core.simulation_components.simulation_packages.golem.golem_input_parser import custom_material_blocks
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemCustomBuilder,
    GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_input_parser import custom_sfepy_group_properties
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    SfepyCustomBuilder,
    FluidProperties,
    SfepyHydrothermalProblemBuilder,
    RockUnitProperties,
)
from core.simulation_components.simulation_visualization.material_preview import (
    _FAULT_COLOR,
    _material_layout,
    plot_builder_materials,
)
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_builder_materials as reexported_plot_builder_materials,
)
from test_golem_model_builder import _synthetic_unstructured_mesh

_EXAMPLES = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "synthetic_examples", "model1", "input_data", "simulation_files"
)

_ROCKS = {
    "basement": RockUnitProperties(name="basement", porosity=0.10),
    "rock1": RockUnitProperties(name="rock1", porosity=0.20),
    "rock2": RockUnitProperties(name="rock2", porosity=0.30),
}


def _read_example(name):
    with open(os.path.join(_EXAMPLES, name), encoding="utf-8") as fh:
        return fh.read()


def _legend(plotter):
    return [plotter.legend.GetEntryString(i) for i in range(plotter.legend.GetNumberOfEntries())]


def _scalar_actors(plotter, name):
    """(cell values, opacity) of every actor coloured by `name`."""
    result = []
    for actor in plotter.actors.values():
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        data = pv.wrap(mapper.GetInput()) if mapper is not None and mapper.GetInput() is not None else None
        if data is not None and name in data.cell_data.keys():
            result.append((np.asarray(data.cell_data[name]), actor.GetProperty().GetOpacity()))
    return result


def _structural_color(geomodel, name):
    for group in geomodel.structural_frame.structural_groups:
        for elem in group.structural_elements:
            if elem.name == name:
                return elem.color
    return "#808080"


@pytest.fixture
def sfepy_builder(model1_implicit_mesh, model1_structural_result):
    return SfepyHydrothermalProblemBuilder(mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
                                      rock_properties=_ROCKS, linear_solver="iterative")


@pytest.fixture
def golem_builder_m1(model1_implicit_mesh, model1_structural_result):
    return GolemHydrothermalProblemBuilder(mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
                                           rock_properties=_ROCKS)


# -----------------------------------------------------------------------------
# Same layout, colours and legend for both simulators
# -----------------------------------------------------------------------------

def test_sfepy_and_golem_hydrothermal_give_the_same_materials_and_legend(sfepy_builder, golem_builder_m1):
    def cells_per_material(builder):
        layout = _material_layout(builder)
        names = np.array([m.name for m in layout.materials] + [None], dtype=object)[layout.cell_material]
        return {m.name: int((names == m.name).sum()) for m in layout.materials}

    assert cells_per_material(sfepy_builder) == cells_per_material(golem_builder_m1)

    legends = [_legend(plot_builder_materials(b, show_plotter=False)) for b in (sfepy_builder, golem_builder_m1)]
    assert legends[0] == legends[1]
    assert legends[0][0] == "Strat_Series1" and legends[0][-2:] == ["Basement", "• basement"]


@pytest.mark.parametrize("builder_fixture", ["sfepy_builder", "golem_builder_m1"])
def test_lithologies_use_their_structural_model_colour(request, builder_fixture, model1_structural_result):
    builder = request.getfixturevalue(builder_fixture)
    plotter = plot_builder_materials(builder, show_plotter=False)
    entries = {plotter.legend.GetEntryString(i): plotter.legend.GetEntryColor(i)
               for i in range(plotter.legend.GetNumberOfEntries())}
    for name in ("basement", "rock1", "rock2"):
        np.testing.assert_allclose(entries[f"• {name}"], to_rgb(_structural_color(model1_structural_result, name)),
                                   atol=1e-3)


@pytest.mark.parametrize("builder_fixture", ["sfepy_builder", "golem_builder_m1"])
def test_property_values_follow_each_lithology(request, builder_fixture):
    builder = request.getfixturevalue(builder_fixture)
    layout = _material_layout(builder)
    plotter = plot_builder_materials(builder, show_plotter=False, property_name="porosity")
    (values, _), = _scalar_actors(plotter, "porosity")
    expected = {m.name: _ROCKS[m.name].porosity for m in layout.materials}
    counts = {name: int((np.array([m.name for m in layout.materials], dtype=object)[layout.cell_material] == name).sum())
              for name in expected}
    for name, porosity in expected.items():
        assert int(np.isclose(values, porosity).sum()) == counts[name]


@pytest.mark.parametrize("requested", ["fluid.viscosity", "viscosity", "mu", "fluid.mu"])
@pytest.mark.parametrize("builder_fixture", ["sfepy_builder", "golem_builder_m1"])
def test_fluid_property_aliases(request, builder_fixture, requested):
    builder = request.getfixturevalue(builder_fixture)
    plotter = plot_builder_materials(builder, show_plotter=False, property_name=requested)
    (values, _), = _scalar_actors(plotter, "fluid.viscosity")
    np.testing.assert_allclose(values, FluidProperties().mu)


def test_unknown_property_lists_available_ones(golem_builder_m1):
    with pytest.raises(ValueError, match=r"'aperture' is not available.*Available properties: .*porosity"):
        plot_builder_materials(golem_builder_m1, show_plotter=False, property_name="aperture")


def test_unsupported_builder_raises():
    with pytest.raises(TypeError, match="Unsupported builder type"):
        plot_builder_materials(object(), show_plotter=False)


def test_simulation_visualization_reexports_the_preview():
    assert reexported_plot_builder_materials is plot_builder_materials


# -----------------------------------------------------------------------------
# Faults
# -----------------------------------------------------------------------------

def test_sfepy_fault_zone_is_an_opaque_fault_material(model2_implicit_mesh, model2_faulted_structural_result):
    fault = RockUnitProperties(name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6)
    builder = SfepyHydrothermalProblemBuilder(mesh_results=model2_implicit_mesh,
                                         geomodel_result=model2_faulted_structural_result,
                                         fault_zone_properties=fault, fault_zone_n_voxels=1,
                                         linear_solver="iterative")
    layout = _material_layout(builder)
    fault_index = next(i for i, m in enumerate(layout.materials) if m.name == "fault_zone")
    assert layout.materials[fault_index].category == "fault"
    assert (layout.cell_material == fault_index).sum() == builder.fault_zone_cell_mask.sum()

    plotter = plot_builder_materials(builder, show_plotter=False)
    legend = _legend(plotter)
    assert legend[:2] == ["Faults", "• fault_zone"]

    actors = _scalar_actors(plot_builder_materials(builder, show_plotter=False, property_name="porosity"), "porosity")
    assert len(actors) == 2  # all material cells, plus the fault zone drawn again on top
    (all_cells, _), (fault_cells, fault_opacity) = actors
    assert np.isclose(fault_cells, 0.02).all() and len(fault_cells) == builder.fault_zone_cell_mask.sum()
    assert fault_opacity >= 0.9


# -----------------------------------------------------------------------------
# Custom builders
# -----------------------------------------------------------------------------

def test_custom_sfepy_values_come_from_the_file(model1_implicit_mesh, model1_structural_result):
    builder = SfepyCustomBuilder(input_file_contents=_read_example("custom_hydrothermal.py"),
                                 mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result)
    plotter = plot_builder_materials(builder, show_plotter=False, property_name="porosity")
    (values, _), = _scalar_actors(plotter, "porosity")
    assert sorted(np.unique(values)) == pytest.approx([0.1, 0.2, 0.3])
    assert _legend(plot_builder_materials(builder, show_plotter=False))[-1] == "• basement"


def test_custom_sfepy_without_geomodel_raises(model1_implicit_mesh):
    builder = SfepyCustomBuilder(input_file_contents=_read_example("custom_hydrothermal.py"),
                                 mesh_results=model1_implicit_mesh)
    with pytest.raises(ValueError, match="geomodel_result"):
        plot_builder_materials(builder, show_plotter=False)


def test_custom_golem_blocks_get_their_lithology(model1_implicit_mesh, model1_structural_result):
    builder = GolemCustomBuilder(input_file_contents=_read_example("custom_hydrothermal.i"),
                                 mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result)
    layout = _material_layout(builder)
    assert {m.lithology for m in layout.materials} == {"basement", "rock1", "rock2"}
    assert all(m.name.startswith("block_") for m in layout.materials)

    plotter = plot_builder_materials(builder, show_plotter=False, property_name="porosity_initial")
    (values, _), = _scalar_actors(plotter, "porosity_initial")
    assert sorted(np.unique(values)) == pytest.approx([0.1, 0.2, 0.3])
    (viscosity, _), = _scalar_actors(
        plot_builder_materials(builder, show_plotter=False, property_name="viscosity"), "fluid.viscosity")
    assert np.isfinite(viscosity).all()


def test_simple_reservoir_categories(monkeypatch, model1_implicit_mesh, model1_structural_result):
    fake_mesh = _synthetic_unstructured_mesh(model1_implicit_mesh, include_geological_fault=True)
    monkeypatch.setattr(golem_builder, "create_unstructured_mesh_data", lambda **kwargs: fake_mesh)
    builder = GolemSimpleReservoirProblemBuilder(geomodel_result=model1_structural_result, mesh_size=25.0,
                                                 curve_mesh_size=4.0, time_sequence=[10.0, 20.0, 50.0])
    layout = _material_layout(builder)
    assert [(m.name, m.category) for m in layout.materials] == [
        ("Reservoir rock", "rock"), ("Geological fault", "fault"),
        ("Reservoir fracture", "fracture"), ("Well", "well"),
    ]
    plotter = plot_builder_materials(builder, show_plotter=False)
    entries = {plotter.legend.GetEntryString(i): plotter.legend.GetEntryColor(i)
               for i in range(plotter.legend.GetNumberOfEntries())}
    np.testing.assert_allclose(entries["• Geological fault"], to_rgb(_FAULT_COLOR), atol=1e-3)
    assert list(entries) == ["Faults", "• Geological fault", "Materials", "• Reservoir rock",
                             "Fractures", "• Reservoir fracture", "Wells", "• Well"]


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------

_GOLEM_INPUT = """
[Mesh]
  [file]
    type = FileMeshGenerator
    file = ${GOLEM_MESH_FILE}
  []
  [rename]
    type = RenameBlockGenerator
    input = file
    old_block = '2 3'
    new_block = 'upper fault'
  []
[]
[UserObjects]
  [perm_upper]
    type = GolemPermeabilityConstant
    value = 1e-14
  []
  [perm_expr]
    type = GolemPermeabilityConstant
    value = '2*x'
  []
[]
[Materials]
  [lower]
    type = GolemMaterialTH
    block = 1
    porosity_initial = 0.1
    permeability_uo = perm_expr
  []
  [upper]
    type = GolemMaterialTH
    block = upper
    porosity_initial = 0.25
    permeability_uo = perm_upper
    fluid_viscosity_initial = 1e-3
  []
  [frac]
    type = GolemMaterialTH
    block = fault
    porosity_initial = 1.0
  []
[]
"""


def test_custom_material_blocks_resolves_names_and_literal_user_objects():
    blocks = custom_material_blocks(_GOLEM_INPUT, block_count=3)
    assert {i: m.name for i, m in blocks.items()} == {0: "lower", 1: "upper", 2: "frac"}
    assert blocks[0].values == {"porosity_initial": 0.1}  # expression-valued UserObject left out
    assert blocks[1].values == {"porosity_initial": 0.25, "permeability": 1e-14, "fluid_viscosity_initial": 1e-3}


@pytest.mark.parametrize("change, message", [
    (("block = 1\n", "block = 1 upper\n"), "assigned to both"),
    (("block = fault\n", "block = 4\n"), "outside 1..3"),
    (("block = fault\n", "block = nowhere\n"), "unresolved block"),
])
def test_custom_material_blocks_rejects_ambiguous_input(change, message):
    with pytest.raises(ValueError, match=message):
        custom_material_blocks(_GOLEM_INPUT.replace(*change), block_count=3)


_SFEPY_INPUT = """
scale = 2.0
regions = {
    'Omega0': 'cells of group 0',
    'Omega1': ('cells of group 1', 'cell'),
}
materials = {
    'a': ({'k': 1.0 * scale, 'porosity': 0.1},),
    'b': ({'k': 3.0},),
    'c': ({'k': 5.0, 'porosity': 0.3},),
    'orphan': ({'extra': 7.0},),
}
equations = {
    'eq': '''dw_laplace.i.Omega0(a.k, s, T)
           + dw_laplace.i.Omega1(b.k, s, T)
           + dw_volume_dot.i.Omega1(c.k, s, T)''',
}
"""


def test_custom_sfepy_group_properties_links_materials_through_equations():
    parsed = custom_sfepy_group_properties(_SFEPY_INPUT)
    assert parsed.group_values[0] == {"k": 2.0, "porosity": 0.1}
    # b and c both act on group 1 with different k: reported, not guessed.
    assert parsed.group_values[1] == {"porosity": 0.3}
    assert parsed.conflicts == {(1, "k")}
    assert parsed.unlinked == {"extra"}
