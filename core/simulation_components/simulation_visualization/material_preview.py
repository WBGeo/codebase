"""
Pre-flight material preview for SfePy and GOLEM problem builders.

Before a (possibly slow) solve, plot_builder_materials shows which material
each mesh cell gets, either one colour per material or coloured by a
numeric material property. Every builder type is first reduced to the same
description (_MaterialLayout: a grid, the material of each cell and each
material's literal property values); colours, legend and plotter setup are
then shared by both simulators.

Colours follow the structural model: a rock material that belongs to a
lithology gets that StructuralElement's colour (basement grey), as in
plot_structural_model_3D and plot_mesh_3d. Faults, fractures and wells get
fixed colours that no lithology uses and are drawn opaque, so they stay
visible inside the semi-transparent rock.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from core.meshing_components.mesh_format.vtk.VTK_format import VTKInputs
from core.object_components import MeshResults, StructuralModelResults
from core.simulation_components.simulation_packages.golem.golem_input_parser import custom_material_blocks
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
    GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
    _block_lithology_ids,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_input_parser import custom_sfepy_group_properties
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    CustomSfepyBuilder,
    HydrothermalProblemBuilder,
    enumerate_rock_units,
    is_custom_sfepy_builder,
)

logger = logging.getLogger(__name__)

_BASEMENT_COLOR = "#808080"
# Deliberately not used for any lithology, so faults always stand out.
_FAULT_COLOR = "#111111"
_CATEGORY_COLORS = {"fault": _FAULT_COLOR, "fracture": "#ff1744", "well": "#ffd600"}
# Rock materials without a lithology (custom GOLEM blocks without a structural
# model, the simple reservoir's single rock). The first is pale on purpose so
# small features stay visible against a single reservoir volume.
_FALLBACK_ROCK_COLORS = ["#d9e2e8", "#4e79a7", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]
_CATEGORY_HEADERS = {"fault": "Faults", "fracture": "Fractures", "well": "Wells"}

# CustomSfepyBuilder's damage zone is only a cell selection, without a
# properties object that could carry a name.
_CUSTOM_SFEPY_FAULT_ZONE_NAME = "fault_zone"

# Requested names that mean a fluid property when no material property of
# that name exists ("viscosity" -> "fluid.viscosity").
_FLUID_ALIASES = {
    "mu": "viscosity", "k_fluid": "thermal_conductivity",
    "fluid_viscosity": "viscosity", "fluid_density": "density",
    "fluid_heat_capacity": "heat_capacity",
    "fluid_thermal_conductivity": "thermal_conductivity",
    "fluid_modulus": "modulus", "rho_c_fluid": "volumetric_heat_capacity",
    "fluid_viscosity_initial": "viscosity",
    "fluid_density_initial": "density",
    "fluid_heat_capacity_initial": "heat_capacity",
    "fluid_thermal_conductivity_initial": "thermal_conductivity",
    "fluid_modulus_initial": "modulus",
}
_BARE_FLUID_NAMES = {"viscosity", "mu", "k_fluid", "rho_c_fluid", "fluid_viscosity", "fluid_density",
                     "fluid_heat_capacity", "fluid_thermal_conductivity", "fluid_modulus"}


@dataclass
class _Material:
    """One material of the preview.

    name: legend label. category: "rock", "fault", "fracture" or "well".
    lithology: structural element name, for colour and legend grouping.
    properties: literal numeric values, fluid ones as "fluid.<name>".
    """
    name: str
    category: str = "rock"
    lithology: Optional[str] = None
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class _MaterialLayout:
    """A builder reduced to what the preview needs.

    cell_material: index into materials for every grid cell, -1 for cells
    that are not a material (exterior boundary blocks, unassigned blocks).
    property_errors: requested property -> reason it can't be shown (for
    example conflicting values in a custom file), raised instead of the
    generic "not available" message.
    """
    simulator: str
    grid: pv.UnstructuredGrid
    cell_material: np.ndarray
    materials: List[_Material]
    geomodel_result: Optional[StructuralModelResults] = None
    property_errors: Dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def _mesh_grid(mesh: MeshResults) -> Tuple[pv.UnstructuredGrid, List[Tuple[int, int]]]:
    """PyVista grid of all mesh blocks and each block's (start, stop) cell range in it."""
    grid = VTKInputs(mesh.nodes, mesh.elements).create_mesh()
    offsets = []
    cursor = 0
    for block in mesh.elements:
        offsets.append((cursor, cursor + len(block.data)))
        cursor += len(block.data)
    if grid.n_cells != cursor:
        raise ValueError(
            f"Mesh cell count mismatch: VTK has {grid.n_cells}, but mesh blocks "
            f"contain {cursor}; cannot safely map materials to cells."
        )
    return grid, offsets


def _numeric_properties(obj) -> Dict[str, float]:
    """Finite scalar fields of a properties object or dict (booleans excluded)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        data = obj
    elif hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif hasattr(obj, "__dict__"):
        data = vars(obj)
    else:
        return {}
    return {k: float(v) for k, v in data.items()
            if not isinstance(v, bool) and isinstance(v, (int, float, np.number)) and np.isfinite(v)}


def _canonical_fluid_name(name: str) -> str:
    """'fluid.mu' / 'mu' / 'fluid_viscosity_initial' -> 'viscosity'."""
    name = name.removeprefix("fluid.")
    return _FLUID_ALIASES.get(name, name)


def _fluid_properties(values: Dict[str, float]) -> Dict[str, float]:
    """Fluid values under their canonical 'fluid.<name>' keys."""
    return {"fluid." + _canonical_fluid_name(key): value for key, value in values.items()}


def _custom_golem_fluid_properties(values: Dict[str, float]) -> Dict[str, float]:
    """'fluid.<name>' entries for the fluid parameters a custom GOLEM material gives."""
    result = {}
    for key, value in values.items():
        if key.startswith("fluid_"):
            canonical = "fluid." + _canonical_fluid_name(key.removesuffix("_value"))
            result.setdefault(canonical, value)
    return result


def _volume_cells(offsets, block_indices) -> np.ndarray:
    """Grid cell ids of the given blocks, in that order."""
    ranges = [np.arange(*offsets[i], dtype=np.int64) for i in block_indices]
    return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)


def _volume_fault_mask(raw_mask, n_total: int, volume_cells: np.ndarray) -> np.ndarray:
    """Damage-zone mask over the volume cells (accepts a mask over all mesh cells too)."""
    mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
    if mask.size == volume_cells.size:
        return mask
    if mask.size == n_total:
        return mask[volume_cells]
    raise ValueError(
        f"Unexpected SfePy damage-zone mask length {mask.size}; expected "
        f"{n_total} mesh cells or {volume_cells.size} volume cells."
    )


# ----------------------------------------------------------------------
# Builder -> layout
# ----------------------------------------------------------------------

def _sfepy_layout(builder) -> _MaterialLayout:
    """Generated or custom SfePy problem: per-cell materials of the volume groups."""
    is_custom = is_custom_sfepy_builder(builder)
    if builder.geomodel_result is None:
        raise ValueError(
            "plot_builder_materials needs builder.geomodel_result to resolve lithology "
            "names and colors -- construct the CustomSfepyBuilder with geomodel_result set."
        )
    mesh = builder.mesh_results
    grid, offsets = _mesh_grid(mesh)
    # SfePy's material groups are the volume blocks in mat_id_to_lithology
    # order; compute_cell_materials() and fault_zone_cell_mask use that order.
    volume_blocks = list(builder.mat_id_to_lithology)
    volume_cells = _volume_cells(offsets, volume_blocks)
    cell_material = np.full(grid.n_cells, -1, dtype=np.int64)
    materials: List[_Material] = []
    errors: Dict[str, str] = {}

    if not is_custom:
        fluid = _fluid_properties(_numeric_properties(builder.fluid))
        fault = builder.fault_zone_properties
        names = np.asarray(builder.compute_cell_materials(), dtype=object)
        for name in dict.fromkeys(names.tolist()):
            if fault is not None and name == fault.name:
                material = _Material(name, "fault", None, {**_numeric_properties(fault), **fluid})
            else:
                material = _Material(name, "rock", name,
                                     {**_numeric_properties(builder.rock_properties.get(name)), **fluid})
            cell_material[volume_cells[names == name]] = len(materials)
            materials.append(material)
    else:
        parsed = custom_sfepy_group_properties(builder.input_file_contents)
        for group_id, coefficient in parsed.conflicts:
            errors[coefficient] = f"Conflicting {coefficient!r} values in SfePy group {group_id}."
        for coefficient in parsed.unlinked:
            errors.setdefault(coefficient, (
                f"{coefficient!r} exists in custom SfePy materials, but no material containing "
                "it can be linked to a 'cells of group N' region through the equations."
            ))
        lith_names = dict(enumerate_rock_units(builder.geomodel_result))
        for mat_id, lith_id in builder.mat_id_to_lithology.items():
            name = lith_names[lith_id]
            cell_material[np.arange(*offsets[mat_id])] = len(materials)
            materials.append(_Material(name, "rock", name, dict(parsed.group_values.get(mat_id, {}))))
        if builder.fault_zone_cell_mask is not None and builder.fault_group_id is not None:
            fault_mask = _volume_fault_mask(builder.fault_zone_cell_mask, grid.n_cells, volume_cells)
            if np.any(fault_mask):
                cell_material[volume_cells[fault_mask]] = len(materials)
                materials.append(_Material(_CUSTOM_SFEPY_FAULT_ZONE_NAME, "fault", None,
                                           dict(parsed.group_values.get(builder.fault_group_id, {}))))

    return _MaterialLayout("SfePy", grid, cell_material, materials, builder.geomodel_result, errors)


def _golem_hydrothermal_layout(builder: GolemHydrothermalProblemBuilder) -> _MaterialLayout:
    """Generated GOLEM problem: 3D rock blocks by lithology, 2D fault blocks."""
    grid, offsets = _mesh_grid(builder.mesh_results)
    cell_material = np.full(grid.n_cells, -1, dtype=np.int64)
    fluid = _fluid_properties(_numeric_properties(builder.fluid))
    materials: List[_Material] = []
    for index in builder.volume_blocks:
        name = builder._rock_name(builder.block_lithology[index])
        cell_material[np.arange(*offsets[index])] = len(materials)
        materials.append(_Material(name, "rock", name,
                                   {**_numeric_properties(builder.rock_properties.get(name)), **fluid}))
    for ordinal, index in enumerate(builder.fault_blocks):
        props = builder._fault_property(ordinal)
        cell_material[np.arange(*offsets[index])] = len(materials)
        materials.append(_Material(props.name, "fault", None, {**_numeric_properties(props), **fluid}))
    return _MaterialLayout("GOLEM", grid, cell_material, materials, builder.geomodel_result)


def _custom_golem_layout(builder: CustomGolemBuilder) -> _MaterialLayout:
    """Custom GOLEM input: the material each [Materials] subsection assigns to its blocks.

    Blocks are classified by dimension (3D rock, 2D fault, lower-dimensional
    well). With a structural model and the mesh's block_id mapping, each 3D
    block also gets its lithology, for structural colours and legend groups.
    """
    meshes = builder.referenced_meshes()
    if not meshes:
        raise ValueError("Custom GOLEM preview requires a WBGeo MeshResults object, not an external mesh file.")
    mesh = next(iter(meshes.values()))
    grid, offsets = _mesh_grid(mesh)
    assigned = custom_material_blocks(builder.input_file_contents, len(mesh.elements))

    lithology_of_block: Dict[int, str] = {}
    if builder.geomodel_result is not None and mesh.cell_data and "block_id" in mesh.cell_data:
        names = dict(enumerate_rock_units(builder.geomodel_result))
        try:
            lithology_of_block = {index: names[lith_id] for index, lith_id in _block_lithology_ids(mesh).items()
                                  if lith_id in names}
        except ValueError:
            lithology_of_block = {}

    cell_material = np.full(grid.n_cells, -1, dtype=np.int64)
    materials: List[_Material] = []
    for index, material in sorted(assigned.items()):
        dim = mesh.elements[index].dim
        category = "rock" if dim == 3 else "fault" if dim == 2 else "well"
        lithology = lithology_of_block.get(index) if category == "rock" else None
        label = material.name if lithology in (None, material.name) else f"{material.name} ({lithology})"
        properties = {**material.values, **_custom_golem_fluid_properties(material.values)}
        cell_material[np.arange(*offsets[index])] = len(materials)
        materials.append(_Material(label, category, lithology, properties))
    return _MaterialLayout("GOLEM", grid, cell_material, materials, builder.geomodel_result)


def _simple_reservoir_layout(builder: GolemSimpleReservoirProblemBuilder) -> _MaterialLayout:
    """Simple reservoir: one rock, geological faults, two fractures, a well.

    Its properties are scalar fields on the builder (rock_porosity, ...),
    not per-lithology objects; block ids are 1-based Exodus ids.
    """
    grid, offsets = _mesh_grid(builder.mesh_results)
    rock = {name: getattr(builder, name) for name in (
        "rock_porosity", "rock_permeability", "rock_density",
        "rock_thermal_conductivity", "rock_heat_capacity",
    )}
    fluid = _fluid_properties({key: float(getattr(builder, key)) for key in (
        "fluid_viscosity", "fluid_density", "fluid_heat_capacity",
        "fluid_thermal_conductivity", "fluid_modulus",
    )})
    groups = (
        (builder.volume_blocks, "Reservoir rock", "rock", rock),
        (builder.geological_fault_blocks, "Geological fault", "fault",
         {**rock, "geological_fault_aperture": builder.geological_fault_aperture}),
        (builder.fracture_blocks, "Reservoir fracture", "fracture",
         {"fracture_porosity": builder.fracture_porosity, "fracture_aperture": builder.fracture_aperture}),
        (builder.well_blocks, "Well", "well",
         {"well_porosity": builder.well_porosity, "well_permeability": builder.well_permeability,
          "well_scaling_factor": builder.well_scaling_factor}),
    )
    # Optional visualization-only porosity overrides per category.
    preview = getattr(builder, "preview_material_porosity", None) or {}
    cell_material = np.full(grid.n_cells, -1, dtype=np.int64)
    materials: List[_Material] = []
    for block_ids, label, category, properties in groups:
        if not block_ids:
            continue
        values = {**_numeric_properties(properties), **fluid}
        if category in preview:
            values["porosity"] = float(preview[category])
        for block_id in block_ids:
            cell_material[np.arange(*offsets[block_id - 1])] = len(materials)
        materials.append(_Material(label, category, None, values))
    return _MaterialLayout("GOLEM", grid, cell_material, materials, builder.geomodel_result)


def _material_layout(builder) -> _MaterialLayout:
    if isinstance(builder, (HydrothermalProblemBuilder, CustomSfepyBuilder)):
        layout = _sfepy_layout(builder)
    elif isinstance(builder, GolemHydrothermalProblemBuilder):
        layout = _golem_hydrothermal_layout(builder)
    elif isinstance(builder, CustomGolemBuilder):
        layout = _custom_golem_layout(builder)
    elif isinstance(builder, GolemSimpleReservoirProblemBuilder):
        layout = _simple_reservoir_layout(builder)
    else:
        raise TypeError(
            f"Unsupported builder type {type(builder).__name__!r}; expected a "
            "generated or custom SfePy/GOLEM builder."
        )
    if not np.any(layout.cell_material >= 0):
        raise ValueError(f"No {layout.simulator} material blocks were identified for preview.")
    return layout


# ----------------------------------------------------------------------
# Colours, legend, plotting (shared by both simulators)
# ----------------------------------------------------------------------

def _structural_colors(geomodel_result: Optional[StructuralModelResults]) -> Dict[str, str]:
    colors = {"basement": _BASEMENT_COLOR}
    frame = getattr(geomodel_result, "structural_frame", None)
    if frame is not None:
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                colors[elem.name] = elem.color
    return colors


def _material_colors(layout: _MaterialLayout) -> List[str]:
    structural = _structural_colors(layout.geomodel_result)
    fallback: Dict[str, str] = {}
    colors = []
    for material in layout.materials:
        if material.category != "rock":
            colors.append(_CATEGORY_COLORS[material.category])
        elif material.lithology in structural:
            colors.append(structural[material.lithology])
        else:
            if material.name not in fallback:
                fallback[material.name] = _FALLBACK_ROCK_COLORS[len(fallback) % len(_FALLBACK_ROCK_COLORS)]
            colors.append(fallback[material.name])
    return colors


def _legend_labels(layout: _MaterialLayout, colors: List[str], present: set) -> List[Tuple[str, str]]:
    """Legend rows grouped like plot_structural_model_3D's legend.

    Faults first, then the structural groups in their stored order (youngest
    at the top), basement, rock materials without a lithology, fractures and
    wells. Each group has a header row followed by "• name" rows.
    """
    sections: Dict[str, List[Tuple[str, str]]] = {}
    group_of = {}
    frame = getattr(layout.geomodel_result, "structural_frame", None)
    group_order = []
    if frame is not None:
        for group in frame.structural_groups:
            group_order.append(group.name)
            for elem in group.structural_elements:
                group_of[elem.name] = group.name
        element_order = [elem.name for group in frame.structural_groups for elem in group.structural_elements]
    else:
        element_order = []

    def rank(index):
        material = layout.materials[index]
        if material.lithology in element_order:
            return element_order.index(material.lithology)
        return len(element_order)

    for index in sorted(present, key=lambda i: (rank(i), i)):
        material = layout.materials[index]
        if material.category != "rock":
            header = _CATEGORY_HEADERS[material.category]
        elif material.lithology in group_of:
            header = group_of[material.lithology]
        elif material.lithology == "basement":
            header = "Basement"
        else:
            header = "Materials"
        row = (f"• {material.name}", colors[index])
        if row not in sections.setdefault(header, []):
            sections[header].append(row)

    order = ["Faults"] + group_order + ["Basement", "Materials", "Fractures", "Wells"]
    labels: List[Tuple[str, str]] = []
    for header in order:
        if header in sections:
            labels.append((header, "black"))
            labels.extend(sections[header])
    return labels


def _finish(plotter: pv.Plotter, show_plotter: bool) -> pv.Plotter:
    plotter.show_bounds(location="furthest", grid=True)
    # Same camera as structural_modeling_visualization and plot_mesh_3d.
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0
    if show_plotter:
        plotter.show()
    return plotter


def _plot_materials(layout: _MaterialLayout, *, style: str, show_edges: bool,
                    show_plotter: bool, opacity: float) -> pv.Plotter:
    colors = _material_colors(layout)
    plotter = pv.Plotter(off_screen=not show_plotter)
    present = set()
    for index, (material, color) in enumerate(zip(layout.materials, colors)):
        cells = np.flatnonzero(layout.cell_material == index)
        if cells.size == 0:
            continue
        present.add(index)
        plotter.add_mesh(
            layout.grid.extract_cells(cells), color=color, style=style, show_edges=show_edges,
            opacity=opacity if material.category == "rock" else 1.0, line_width=1.0,
        )
    labels = _legend_labels(layout, colors, present)
    if labels:
        plotter.add_legend(labels=labels, size=(0.32, min(0.03 * len(labels), 0.65)),
                           loc="lower right", face="rectangle")
    return _finish(plotter, show_plotter)


def _resolve_property_name(layout: _MaterialLayout, property_name: str) -> str:
    available = {key for material in layout.materials for key in material.properties}
    if property_name in available or property_name in layout.property_errors:
        return property_name
    if property_name.startswith("fluid.") or property_name in _BARE_FLUID_NAMES:
        return "fluid." + _canonical_fluid_name(property_name)
    return property_name


def _plot_property(layout: _MaterialLayout, property_name: str, *, style: str, show_edges: bool,
                   show_plotter: bool, cmap: str, opacity: float) -> pv.Plotter:
    name = _resolve_property_name(layout, property_name)
    if name in layout.property_errors:
        raise ValueError(layout.property_errors[name])
    available = sorted({key for material in layout.materials for key in material.properties})
    if name not in available:
        raise ValueError(
            f"Material property {property_name!r} is not available as a numeric constant "
            f"for this {layout.simulator} problem. Available properties: {', '.join(available) or '(none)'}. "
            "Temperature/pressure-dependent or expression-based values require evaluated "
            "simulation output, not a pre-solve material preview."
        )
    per_material = np.array([m.properties.get(name, np.nan) for m in layout.materials] + [np.nan])
    values = per_material[layout.cell_material]  # -1 picks the trailing NaN
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f"No mesh cells have a numeric value for {name!r}.")

    plotter = pv.Plotter(off_screen=not show_plotter)
    part = layout.grid.extract_cells(np.flatnonzero(valid))
    part.cell_data[name] = values[valid]
    plotter.add_mesh(part, scalars=name, cmap=cmap, style=style, show_edges=show_edges,
                     opacity=opacity, scalar_bar_args={"title": name})
    # A single semi-transparent grid shows interior faults, fractures and
    # wells only where they reach the outer surface. Draw them again as a
    # separate, nearly opaque actor on the same colour range.
    feature = np.array([m.category != "rock" for m in layout.materials] + [False])[layout.cell_material]
    feature_cells = np.flatnonzero(feature & valid)
    if feature_cells.size:
        features = layout.grid.extract_cells(feature_cells)
        features.cell_data[name] = values[feature_cells]
        plotter.add_mesh(features, scalars=name, cmap=cmap, style=style, show_edges=False,
                         clim=(float(values[valid].min()), float(values[valid].max())),
                         opacity=min(1.0, max(opacity, 0.9)), show_scalar_bar=False)
    return _finish(plotter, show_plotter)


def plot_builder_materials(
    builder,
    style: str = "surface",
    show_edges: bool = False,
    show_plotter: bool = True,
    property_name: Optional[str] = None,
    cmap: str = "viridis",
    opacity: float = 0.65,
) -> pv.Plotter:
    """Preview the material assignment of a SfePy or GOLEM problem builder.

    Works for HydrothermalProblemBuilder, CustomSfepyBuilder,
    GolemHydrothermalProblemBuilder, CustomGolemBuilder and
    GolemSimpleReservoirProblemBuilder, with the same colours, legend and
    view for all of them.

    Without property_name, every material gets one colour: lithologies their
    structural-model colour, faults (SfePy damage zones and GOLEM 2D fault
    blocks), fractures and wells fixed opaque colours. The legend is grouped
    like the structural model plot's.

    With property_name, cells are coloured by that literal material value
    (for example "porosity", "k_solid", or a custom file's own coefficient
    names). Fluid properties are shared by all materials and can be given
    as "fluid.<name>" or as "viscosity", "mu", "rho_c_fluid" and similar.
    Expression- or state-dependent values can't be previewed before a solve.

    :param builder: The problem builder to preview.
    :param style: PyVista style of the material meshes ("surface", "wireframe", ...).
    :param show_edges: Draw cell edges (noisy above roughly 50k cells).
    :param show_plotter: Show the window; False renders off screen and only returns the plotter.
    :param property_name: Numeric material property to colour by; None for one colour per material.
    :param cmap: Colormap for property_name.
    :param opacity: Opacity of rock materials (faults, fractures and wells are drawn opaque).
    :return: The PyVista plotter.
    """
    layout = _material_layout(builder)
    if property_name is None:
        return _plot_materials(layout, style=style, show_edges=show_edges,
                               show_plotter=show_plotter, opacity=opacity)
    return _plot_property(layout, property_name, style=style, show_edges=show_edges,
                          show_plotter=show_plotter, cmap=cmap, opacity=opacity)
