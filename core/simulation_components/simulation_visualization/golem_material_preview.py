"""Pre-solve GOLEM mesh material/fault preview; independent of SfePy plotting."""
from __future__ import annotations

import re
import shlex
import pyvista as pv
from core.simulation_components.simulation_packages.golem.golem_input_parser import locate_blocks
from core.meshing_components.mesh_format.vtk.VTK_format import VTKInputs
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder, GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
)


def _section_parameters(lines, section):
    """Read scalar assignments in a MOOSE section, without evaluating expressions."""
    result = {}
    for line in lines[section.start + 1:section.end]:
        match = re.match(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(2).split("#", 1)[0].strip()
        try:
            tokens = shlex.split(value)
        except ValueError:
            tokens = [value]
        result[match.group(1)] = " ".join(tokens)
    return result


def _numeric_literal(value):
    """Only accept actual numeric literals; don't pretend expressions are constants."""
    if value is None or not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value
    ):
        return None
    return float(value)


def _custom_material_details(content: str, block_count: int):
    """Discover any explicitly assigned custom materials and resolvable values.

    A material may cover several mesh blocks. Property names are discovered
    rather than hardcoded. UserObject references are followed only if the
    referenced object declares a literal `value`; no MOOSE evaluation occurs.
    """
    lines, sections = locate_blocks(content)
    parents = {i for i, b in enumerate(sections) if b.depth == 0 and b.name == "Materials"}
    material_sections = [b for b in sections if b.parent in parents]
    if not material_sections:
        raise ValueError("Custom GOLEM input has no [Materials] subsections to preview.")

    user_parents = {i for i, b in enumerate(sections) if b.depth == 0 and b.name == "UserObjects"}
    user_objects = {b.name: _section_parameters(lines, b)
                    for b in sections if b.parent in user_parents}

    renamed = {}
    for section in sections:
        if section.name == "Materials" or section.parent in parents:
            continue
        params = _section_parameters(lines, section)
        if params.get("type") != "RenameBlockGenerator":
            continue
        old_ids = params.get("old_block", "").split()
        new_names = params.get("new_block", "").split()
        if len(old_ids) != len(new_names):
            raise ValueError("RenameBlockGenerator old_block/new_block counts differ.")
        for number, label in zip(old_ids, new_names):
            renamed[label] = int(number)

    assignments = {}
    summaries = {}
    for section in material_sections:
        params = _section_parameters(lines, section)
        if "block" not in params:
            raise ValueError(f"Material [{section.name}] has no explicit block assignment; cannot preview safely.")
        resolved = []
        for key, raw in params.items():
            if key in ("block", "type"):
                continue
            numeric = _numeric_literal(raw)
            if numeric is not None:
                resolved.append(f"{key}={numeric:g}")
            elif key.endswith("_uo") and raw in user_objects:
                uo_value = _numeric_literal(user_objects[raw].get("value"))
                if uo_value is not None:
                    resolved.append(f"{key[:-3]}={uo_value:g}")
        # Use line breaks to keep legends readable even with many parameters.
        # PyVista legends can be crowded; show a limited number and report the rest.
        shown = resolved[:6]
        if len(resolved) > 6:
            shown.append(f"+{len(resolved) - 6} more")
        summary = " | ".join(shown)
        for token in params["block"].split():
            try:
                block_id = int(token)
            except ValueError:
                if token not in renamed:
                    raise ValueError(f"Material [{section.name}] references unresolved block {token!r}.")
                block_id = renamed[token]
            if not 1 <= block_id <= block_count:
                raise ValueError(f"Material [{section.name}] references block {block_id}, outside 1..{block_count}.")
            previous = assignments.get(block_id)
            if previous is not None and previous != section.name:
                raise ValueError(f"Block {block_id} is assigned to both [{previous}] and [{section.name}].")
            assignments[block_id] = section.name
            summaries[block_id - 1] = summary
    if not assignments:
        raise ValueError("No explicit material block assignments found in custom GOLEM input.")
    entries = [(block_id - 1, f"Material: {label}", "rock")
               for block_id, label in sorted(assignments.items())]
    return entries, summaries


def _custom_material_entries(content: str, block_count: int):
    """Compatibility API: return the original three-column material entries."""
    return _custom_material_details(content, block_count)[0]


def _material_property_summary(properties):
    """SfePy-compatible rock summary, with a safe fallback for GOLEM properties."""
    if properties is None:
        return ""
    rock_fields = ("porosity", "permeability", "k_solid", "rho_c_solid")
    if all(getattr(properties, field, None) is not None for field in rock_fields):
        return (
            f"poro={properties.porosity:g} "
            f"perm={properties.permeability:.1e}m2 "
            f"k_s={properties.k_solid:g}W/mK "
            f"rhoc={properties.rho_c_solid:.1e}J/m3K"
        )
    # GOLEM fault properties may have different field names. Show only
    # actual scalar values rather than guessing units or inventing values.
    values = (properties.model_dump() if hasattr(properties, "model_dump")
              else vars(properties) if hasattr(properties, "__dict__") else {})
    parts = []
    for key, value in values.items():
        if key.startswith("_") or key == "name" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.3g}")
    return " ".join(parts)


def _rock_properties_for_name(builder, name):
    properties = getattr(builder, "rock_properties", None)
    if isinstance(properties, dict):
        return properties.get(name)
    return None


def plot_golem_materials(builder, *, show_edges: bool = False, show_plotter: bool = True) -> pv.Plotter:
    """Color actual GOLEM material blocks, including faults/fractures/wells.

    The builder's own material-block classifications are used. The six trailing
    exterior-boundary blocks of unstructured WBGeo meshes are not materials.
    """
    if not isinstance(builder, (CustomGolemBuilder, GolemHydrothermalProblemBuilder,
                                GolemSimpleReservoirProblemBuilder)):
        raise TypeError(f"Unsupported GOLEM builder: {type(builder).__name__}")
    if isinstance(builder, CustomGolemBuilder):
        meshes = builder.referenced_meshes()
        if not meshes:
            raise ValueError("Custom GOLEM preview requires a WBGeo MeshResults object, not an external mesh file.")
        mesh = next(iter(meshes.values()))
    else:
        mesh = builder.mesh_results
    grid = VTKInputs(mesh.nodes, mesh.elements).create_mesh()
    blocks = mesh.elements
    offsets = []
    cursor = 0
    for block in blocks:
        count = len(block.data)
        offsets.append((cursor, cursor + count))
        cursor += count
    if grid.n_cells != cursor:
        raise ValueError(
            f"Mesh cell count mismatch: VTK has {grid.n_cells}, "
            f"but mesh blocks contain {cursor}; cannot safely color assignments."
        )

    # (zero-based block index, displayed material label, visual category)
    entries = []
    property_by_index = {}
    custom_summaries = {}
    if isinstance(builder, CustomGolemBuilder):
        entries, custom_summaries = _custom_material_details(builder.input_file_contents, len(blocks))
    elif isinstance(builder, GolemHydrothermalProblemBuilder):
        for index in builder.volume_blocks:
            lith_id = builder.block_lithology[index]
            name = builder._rock_name(lith_id)
            entries.append((index, f"Rock: {name}", "rock"))
            property_by_index[index] = _rock_properties_for_name(builder, name)
        for ordinal, index in enumerate(builder.fault_blocks):
            fault_props = builder._fault_property(ordinal)
            name = fault_props.name
            entries.append((index, f"Fault: {name}", "fault"))
            property_by_index[index] = fault_props
    else:
        entries.extend((block_id - 1, "Reservoir rock", "rock") for block_id in builder.volume_blocks)
        entries.extend((block_id - 1, "Geological fault", "fault") for block_id in builder.geological_fault_blocks)
        entries.extend((block_id - 1, "Reservoir fracture", "fracture") for block_id in builder.fracture_blocks)
        entries.extend((block_id - 1, "Well", "well") for block_id in builder.well_blocks)

    if not entries:
        raise ValueError("No GOLEM material blocks were identified for preview.")

    palette = {"fault": "#d62728", "fracture": "#ff7f0e", "well": "#111111"}
    rock_colors = ["#4e79a7", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]
    rock_labels = list(dict.fromkeys(label for _, label, category in entries if category == "rock"))
    colors = {label: rock_colors[i % len(rock_colors)] for i, label in enumerate(rock_labels)}
    plotter = pv.Plotter(off_screen=not show_plotter)
    legend = {}
    property_details = {}
    for index, label, category in entries:
        if not 0 <= index < len(offsets):
            raise ValueError(f"GOLEM material block index {index} is outside the mesh.")
        start, stop = offsets[index]
        if start == stop:
            continue
        color = colors[label] if category == "rock" else palette[category]
        part = grid.extract_cells(range(start, stop))
        plotter.add_mesh(part, color=color, show_edges=show_edges,
                         opacity=0.65 if category == "rock" else 1.0,
                         line_width=1.0)
        summary = custom_summaries.get(index) or _material_property_summary(property_by_index.get(index))
        # Keep the on-plot legend compact: long property strings force VTK
        # to shrink every label until it becomes unreadable.
        legend[label] = color
        if summary:
            property_details.setdefault(label, summary)
    if legend:
        plotter.add_legend(
            labels=list(legend.items()), loc="lower right",
            face="rectangle", size=(0.23, min(0.08 + 0.055 * len(legend), 0.65)),
            bcolor="white", border=True,
        )
    # Workbench inspectors also capture stdout. Report the complete available
    # summaries there, without imposing a fixed-width/height plot legend.
    if property_details:
        print("GOLEM material properties (resolved constants):")
        for label, summary in property_details.items():
            print(f"  {label}:")
            for field in summary.split(" | "):
                print(f"    {field}")
    plotter.show_bounds(location="furthest", grid=True)
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0
    if show_plotter:
        plotter.show()
    return plotter
