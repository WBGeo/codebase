"""
Interpolation pipeline utilities for geological structural modeling with optional fault domains.

Main stages (per-domain when faults are provided):
  1) Interpolation per structural group
  2) Age-mask computation per group
  3) Lithology block assembly
  4) (Optional) Masked surface mesh extraction for each element
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import itertools
import colorsys

from core.object_components import InputData_StructuralElements, StructuralModelResults
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes

from core.structuralmodeling_components.structural_objects.structural_objects import (
    StructuralFrame,
    StructuralGroup,
    StructuralElement,
)
from core.structuralmodeling_components.structural_objects.structural_objects import FaultFrame

from typing import Dict, Optional

from core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group import (
    interpolate_group_ordinary_kriging,
)
from core.structuralmodeling_components.interpolators_per_group.radial_basis_function_per_group import (
    interpolate_group_radial_basis_function,
)
from core.structuralmodeling_components.interpolators_per_group.universal_cokriging_per_group import (
    interpolate_group_universal_cokriging,
)
from core.structuralmodeling_components.interpolators_per_group.geoinr_per_group import (
    interpolate_group_geo_inr,
)
from core.structuralmodeling_components.interpolators_per_group.loop_structural_per_group import (
    interpolate_group_loop_structural,
)

from core.structuralmodeling_components.structural_objects.structural_objects import InterpolationMethod


# -----------------------------------------------------------------------------
# Interpolator dispatch
# -----------------------------------------------------------------------------
interpolate_dispatch = {
    InterpolationMethod.ORDINARY_KRIGING: interpolate_group_ordinary_kriging,
    InterpolationMethod.RADIAL_BASIS_FUNCTION: interpolate_group_radial_basis_function,
    InterpolationMethod.UNIVERSAL_COKRIGING: interpolate_group_universal_cokriging,
    InterpolationMethod.GEOINR: interpolate_group_geo_inr,
    InterpolationMethod.LOOP_STRUCTURAL: interpolate_group_loop_structural,
}


def run_interpolation_with_fault_domains(
    frame: StructuralFrame,
    fault_frame: Optional[FaultFrame],
    *,
    crop_to_domain: bool = True,
) -> None:
    """Interpolate structural groups inside fault domains, respecting fault activity.

    Groups not affected by any fault get a single continuous scalar field
    across all points.

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame containing groups, elements, grid, and input_data.
    fault_frame : Optional[FaultFrame]
        Fault frame providing a 3D domain_map with domain identifiers.
    crop_to_domain : bool, default True
        Crop computation to minimal bounding box for speed.
    """

    domain_map = (
        fault_frame.domain_map if fault_frame is not None else np.zeros(frame.grid.resolution, dtype=int)
    )

    domain_ids = np.unique(domain_map)

    # Assign each point to a domain
    sp_in_domain = assign_domain_ids_to_points(frame.grid, domain_map, frame.surface_points)
    ori_in_domain = (
        assign_domain_ids_to_points(frame.grid, domain_map, frame.orientations)
        if frame.orientations is not None
        else None
    )

    for group_idx, group in enumerate(frame.structural_groups):
        group_formations = [e.name for e in group.structural_elements]

        # # --- Determine which domains are active for this group ---
        if fault_frame is None or frame.fault_activity is None:
            components = [set(map(int, domain_ids))]
        else:
            components = effective_domain_components_for_group(
                domain_ids=domain_ids,
                faults=list(fault_frame.fault_elements),
                fault_activity=frame.fault_activity,
                group_idx=group_idx,
            )

        # --- Interpolate per active domain ---
        for comp_ids in components:
            comp_ids_arr = np.array(sorted(comp_ids), dtype=int)

            # points from ALL domains in this merged component
            sp_filtered_all = sp_in_domain[sp_in_domain["domain_id"].isin(comp_ids_arr)].drop(columns="domain_id")
            ori_filtered_all = (
                ori_in_domain[ori_in_domain["domain_id"].isin(comp_ids_arr)].drop(columns="domain_id")
                if ori_in_domain is not None
                else None
            )

            # Filter points for this group
            sp_filtered = sp_filtered_all[sp_filtered_all["formation"].isin(group_formations)]
            ori_filtered = None
            if ori_filtered_all is not None:
                ori_filtered = ori_filtered_all[ori_filtered_all["formation"].isin(group_formations)]

            if sp_filtered.empty:
                continue

            # bbox from merged component mask (optional crop)
            use_grid = frame.grid
            bbox = None
            if crop_to_domain and len(domain_ids) > 1:
                comp_mask = np.isin(domain_map, comp_ids_arr)
                # compute bbox from comp_mask (XYZ)
                xs = np.where(comp_mask.any(axis=(1, 2)))[0]
                ys = np.where(comp_mask.any(axis=(0, 2)))[0]
                zs = np.where(comp_mask.any(axis=(0, 1)))[0]
                if xs.size and ys.size and zs.size:
                    bbox = (int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1]), int(zs[0]), int(zs[-1]))
                    use_grid = build_subgrid_from_bbox(frame.grid, bbox)

            # Pick interpolator
            method = group.interpolation_method
            if method not in interpolate_dispatch:
                     raise ValueError(f"Unsupported interpolation method: {method}")

            # interpolate once for the merged component
            scalar_field_sub, scalar_values = interpolate_dispatch[method](
                group=group,
                grid=use_grid,
                group_surface_points_df=sp_filtered,
                group_orientations_points_df=ori_filtered,
            )

            scalar_field_sub = scalar_field_sub.transpose(2, 1, 0)  # if needed for your interpolator output

            if bbox is not None and len(domain_ids) > 1:
                full_shape = tuple(frame.grid.resolution)
                scalar_field_full = np.full(full_shape, np.nan, dtype=scalar_field_sub.dtype)
                kx0, kx1, ky0, ky1, kz0, kz1 = bbox
                scalar_field_full[kx0:kx1 + 1, ky0:ky1 + 1, kz0:kz1 + 1] = scalar_field_sub
                scalar_field = scalar_field_full
            else:
                scalar_field = scalar_field_sub

            comp_vox_mask = np.isin(domain_map, comp_ids_arr)
            group._scalar_field = np.where(comp_vox_mask, scalar_field, group.get_scalar_field())

            # Store scalar values per element
            for elem in group.structural_elements:
                if elem.name not in scalar_values:
                    raise ValueError(
                        f"Interpolator did not return a scalar value for element '{elem.name}' in group '{group.name}'."
                    )
                elem.set_scalar_value(float(scalar_values[elem.name]))

# --- 2) Age masks per domain --------------------------------------------------


def set_scalar_masks_per_domain(frame: StructuralFrame) -> None:
    """Compute and set age masks per group, respecting fault activity.

    Stores a **single mask per group** that combines all active domains.
    """
    ff = frame.fault_frame
    domain_map = ff.domain_map if ff is not None else np.zeros(frame.grid.resolution, dtype=int).T
    domain_ids = np.unique(domain_map)

    if not frame.structural_groups:
        raise ValueError("The structural frame contains no groups.")

    groups = frame.structural_groups

    for group_idx, group in enumerate(groups):
        sf = group.get_scalar_field()
        if sf is None:
            raise ValueError(f"Group '{group.name}' has no scalar field set.")

        # Compute "oldest group" mask
        if group_idx == len(groups) - 1:
            mask = np.ones_like(sf, dtype=bool)
        else:
            if not group.structural_elements:
                raise ValueError(f"Group '{group.name}' has no structural elements.")
            oldest = group.structural_elements[-1]
            sval = oldest.get_scalar_value()
            if sval is None:
                raise ValueError(
                    f"Oldest element '{oldest.name}' in group '{group.name}' has no scalar value."
                )
            mask = sf >= sval

        # If fault frame exists, only keep mask in active domains
        if ff is not None:
            combined_mask = np.zeros_like(mask, dtype=bool)
            for d in domain_ids:
                # Check if this domain is active for this group
                domain_ok = False
                for fault in ff.fault_elements:
                    youngest_idx = frame._fault_activity[fault.name]
                    if group_idx < youngest_idx:
                        continue
                    if d in fault.separated_domains_flat():
                        domain_ok = True
                        break
                # Groups unaffected by any fault in this domain are allowed
                if not domain_ok and all(group_idx < frame.fault_activity[f.name] for f in ff.fault_elements):
                    domain_ok = True

                if domain_ok:
                    combined_mask[domain_map == d] = mask[domain_map == d]

            mask = combined_mask

        # Store the final mask
        group.set_mask(mask)

# --- 3) Combine all domains to final lithology block --------------------------


def compute_lithology_block_with_domains(frame: StructuralFrame) -> np.ndarray:
    """Combine group results into a single lithology block, respecting fault domains
    and per-fault activity.

    Stores element IDs globally (same across domains) and honors age/group masks.

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame containing groups/elements and per-domain scalar fields/masks.

    Returns
    -------
    np.ndarray
        Final lithology volume (integer IDs) with shape frame.grid.resolution.
    """
    shape = tuple(map(int, frame.grid.resolution))  # (X,Y,Z)
    final_lith = np.zeros(shape, dtype=int)

    # Assign stable global IDs to elements if not already set
    current_id = 1
    for group in reversed(frame.structural_groups):
        for elem in reversed(group.structural_elements):
            if elem.id is None:
                elem.set_id(current_id)
                current_id += 1

    # Fault frame and domain map
    ff = frame.fault_frame
    if ff is not None:
        domain_map = ff.domain_map
        domain_ids = np.unique(domain_map)
    else:
        domain_map = np.zeros(shape, dtype=int)
        domain_ids = [0]

    # Process groups from oldest -> youngest so younger overwrites older
    groups = frame.structural_groups
    n_groups = len(groups)

    for group_idx in range(n_groups - 1, -1, -1):  # oldest -> youngest (by index)
        group = groups[group_idx]

        sf = group.get_scalar_field()
        gm = group.get_mask()
        if sf is None or gm is None:
            continue

        # --- compute effective domain components for THIS group ---
        if ff is None or frame.fault_activity is None:
            components = [set(map(int, domain_ids))]
        else:
            components = effective_domain_components_for_group(
                domain_ids=domain_ids,
                faults=list(ff.fault_elements),
                fault_activity=frame.fault_activity,  # dict[str,int] youngest affected idx
                group_idx=group_idx,  # IMPORTANT: original index (youngest->oldest)
            )

        # print(f"[{group.name}] components:", [sorted(list(c)) for c in components])

        # Process each merged component
        for comp_ids in components:

            comp_ids_arr = np.array(sorted(comp_ids), dtype=int)
            comp_mask = np.isin(domain_map, comp_ids_arr)

            group_block = np.zeros(shape, dtype=int)

            # Fill elements oldest -> youngest inside the group (so younger elem overwrites older within group_block)
            # Your element order seems youngest->oldest inside group; you want oldest->youngest fill with (==0) priority:
            for elem in group.structural_elements:
                sval = elem.get_scalar_value()
                if sval is None:
                    continue
                write_mask = (sf >= sval) & (group_block == 0)
                group_block[write_mask] = elem.id

            # Apply age mask and component mask
            group_block = np.where(gm, group_block, 0)
            group_block = np.where(comp_mask, group_block, 0)

            # Younger groups overwrite older groups in final lith
            final_lith = np.where(group_block > 0, group_block, final_lith)

    return final_lith

# --- 4) Extract per-domain “masked” meshes for each element -------------------


def extract_all_meshes_per_domain(
    frame: StructuralFrame,
) -> None:
    """Extract and store per-domain masked and unmasked surface meshes for every element.

    Masked meshes stop at unconformities and domain boundaries.
    Unmasked meshes go through unconformities.
    Combined meshes come from the final lithology block.

    Parameters
    ----------
    frame : StructuralFrame
        Frame with groups/elements and per-domain values.
    """
    # Use the fault frame stored inside the structural frame
    fault_frame = frame.fault_frame
    if fault_frame is None:
        domain_ids = [0]
        dom_map = np.zeros(tuple(frame.grid.resolution), dtype=int)
    else:
        dom_map = fault_frame.domain_map
        domain_ids = np.unique(dom_map)

    # Temporary storage: { (elem, domain_id) : (verts, faces) }
    temp_domain_meshes = {}

    for d in domain_ids:
        dommask = dom_map == d  # keep in XYZ

        for i, group in enumerate(frame.structural_groups):
            sf = group.get_scalar_field()  # XYZ
            if sf is None:
                continue

            # Age-mask logic
            if i == 0:
                # Youngest group: no truncation
                erosion_mask = np.ones_like(sf, dtype=bool)
            else:
                prev_mask = frame.structural_groups[i - 1].get_mask()  # XYZ
                erosion_mask = ~prev_mask if prev_mask is not None else np.ones_like(sf, dtype=bool)

            # Combined mask
            mc_mask = erosion_mask & dommask  # keep in XYZ

            for elem in group.structural_elements:
                sval = elem.scalar_value
                if sval is None:
                    continue

                verts, faces = marching_cubes_per_element(
                    sf,        # XYZ
                    sval,
                    frame.grid.spacing,
                    frame.grid.extent,
                    mask=mc_mask  # XYZ
                )

                temp_domain_meshes[(elem.name, d)] = (verts, faces)

    # Combine meshes across domains for each element
    def combine_meshes(vertices_list, edges_list):
        combined_vertices = []
        combined_edges = []
        offset = 0
        for V, E in zip(vertices_list, edges_list):
            combined_vertices.append(V)
            combined_edges.append(E + offset)
            offset += V.shape[0]
        return np.vstack(combined_vertices), np.vstack(combined_edges)

    for group in frame.structural_groups:
        for elem in group.structural_elements:
            verts_list, faces_list = [], []
            for d in domain_ids:
                verts, faces = temp_domain_meshes.get((elem.name, d), (np.empty((0, 3)), np.empty((0, 3))))
                verts_list.append(verts)
                faces_list.append(faces)
            combined_vertices, combined_faces = combine_meshes(verts_list, faces_list)
            elem.set_mesh("masked", combined_vertices, combined_faces)

    # ---- unmasked meshes ----
    for group in frame.structural_groups:
        sf = group.get_scalar_field()
        if sf is None:
            continue
        for elem in group.structural_elements:
            sval = elem.scalar_value
            if sval is None:
                continue
            verts, faces = marching_cubes_per_element(sf, sval, frame.grid.spacing, frame.grid.extent, mask=None)
            elem.set_mesh("unmasked", verts, faces)

    # ---- combined meshes from lithology block ----
    lith_block = frame.get_LithBlock()
    if lith_block is None:
        return

    unique_ids = [
        element.id - 0.1
        for group in frame.structural_groups
        for element in group.structural_elements
        if element.id is not None
    ]
    combined_vertices, combined_edges = marching_cubes(lith_block, unique_ids, frame.grid.spacing, frame.grid.extent)

    idx = 0
    for group in frame.structural_groups:
        for element in group.structural_elements:
            element.set_mesh("combined", combined_vertices[idx], combined_edges[idx])
            idx += 1

    # ---- masked meshes for faults using existing age masks ----
    if fault_frame is not None:
        for fault in fault_frame.fault_elements:

            # this mask is the age mask defined by youngest group affected by this fault
            index = frame.fault_activity_verbose[fault.name]["youngest_group_index"]-1
            mc_fault_mask = frame.structural_groups[index].get_mask() if index >= 0 else np.ones_like(sf, dtype=bool)
            # inverse
            mc_fault_mask = ~mc_fault_mask

            verts, faces = marching_cubes_per_element(
                fault.scalar_field,  # scalar field
                fault.scalar_value,  # isovalue
                frame.grid.spacing,
                frame.grid.extent,
                mask=mc_fault_mask,  # only keep mesh where fault_mask is True
            )
            fault.set_mesh("masked", verts, faces)

# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------


def compute_structural_model(
    frame: StructuralFrame,
    *,
    extract_meshes: bool = True,
    verbose: bool = True,
):
    """Run the full pipeline with (optional) fault domains.

    Steps
    -----
    1) Per-domain interpolation (stores scalar fields/values per domain)
    2) Per-domain age masks
    3) Final lithology block combining domains
    4) Per-domain masked surface meshes (optional)

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame containing grid, groups, elements, and input input_data.
    extract_meshes : bool, default True
        Whether to extract masked meshes at the end.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    StructuralModelResults
        Results object containing the updated structural frame with scalar fields, masks, and meshes.
    """

    # --- prepare domain map (supports "no faults" case) ---
    ff = frame.fault_frame
    if ff is None:
        # synthetic single domain
        dom_map = np.zeros(tuple(map(int, frame.grid.resolution)), dtype=int)

        class _TmpFF:
            domain_map = dom_map

        ff = _TmpFF()
        if verbose:
            print("ℹ️ No fault_frame provided in StructuralFrame — running in single-domain mode.")

    # --- 1) per-domain interpolation, store results into per-domain slots ---
    if verbose:
        print("① Interpolation per domain ...")
    run_interpolation_with_fault_domains(
        frame=frame,
        fault_frame=ff,
    )

    # --- 2) per-domain age masks ---
    if verbose:
        print("② Computing age masks per domain ...")
    set_scalar_masks_per_domain(frame)

    # --- 3) final lithology block combining domains ---
    if verbose:
        print("③ Building final lithology block ...")
    frame._lith_block = compute_lithology_block_with_domains(frame)

    # --- 4) per-domain meshes (optional) ---
    if extract_meshes:
        if verbose:
            print("④ Extracting per-domain masked meshes ...")
        extract_all_meshes_per_domain(
            frame=frame
        )

    if verbose:
        print("✅ Pipeline complete.")

    # return a StructuralModelResults object instead
    result = StructuralModelResults(
        structural_frame=copy.deepcopy(frame),
    )

    return result

# -----------------------------------------------------------------------------
# Builders / helpers
# -----------------------------------------------------------------------------


def compute_domain_bbox_indices(domain_map: np.ndarray, domain_id: int):
    """Return tight bounding box in XYZ index order."""
    mask = domain_map == domain_id
    if not np.any(mask):
        return None

    # domain_map shape = (X,Y,Z)
    # Compute index ranges along each axis
    x_any = mask.any(axis=(1, 2))  # collapse Y,Z → X
    y_any = mask.any(axis=(0, 2))  # collapse X,Z → Y
    z_any = mask.any(axis=(0, 1))  # collapse X,Y → Z

    kx0, kx1 = np.where(x_any)[0][[0, -1]]
    ky0, ky1 = np.where(y_any)[0][[0, -1]]
    kz0, kz1 = np.where(z_any)[0][[0, -1]]

    return int(kx0), int(kx1), int(ky0), int(ky1), int(kz0), int(kz1)


def build_subgrid_from_bbox(grid, bbox):
    """
    bbox: (kx0, kx1, ky0, ky1, kz0, kz1) in [X,Y,Z] index space
    Returns a RegularGrid in [X,Y,Z] space
    """
    kx0, kx1, ky0, ky1, kz0, kz1 = bbox

    # Original grid info
    dx, dy, dz = grid.spacing  # spacing in X,Y,Z
    x0, x1, y0, y1, z0, z1 = grid.extent

    # Convert index bbox → physical extents
    sub_extent = (
        x0 + kx0 * dx,
        x0 + (kx1 + 1) * dx,
        y0 + ky0 * dy,
        y0 + (ky1 + 1) * dy,
        z0 + kz0 * dz,
        z0 + (kz1 + 1) * dz,
    )

    sub_resolution = (
        kx1 - kx0 + 1,
        ky1 - ky0 + 1,
        kz1 - kz0 + 1,
    )

    return RegularGrid(
        extent=sub_extent,
        resolution=sub_resolution,
    )


def generate_grouped_colors_per_element(groups: list, base_colormap: str = "Accent") -> Dict[str, str]:
    """Generate distinct color shades for each element in each group.

    Parameters
    ----------
    groups : list[StructuralGroup]
        Structural groups whose elements will receive colors.
    base_colormap : str, default "Accent"
        Matplotlib colormap used for base group colors.

    Returns
    -------
    dict
        Mapping ``{element_name: hex_color}``.
    """
    group_cmap = plt.get_cmap(base_colormap)
    color_map: Dict[str, str] = {}

    n_groups = len(groups)
    for group_idx, group in enumerate(groups):
        base_rgb = group_cmap(group_idx / max(n_groups, 1))[:3]  # Get RGB triple (ignore alpha)
        base_hls = colorsys.rgb_to_hls(*base_rgb)

        n_elements = len(group.structural_elements)
        for i, element in enumerate(group.structural_elements):
            # Create variation in lightness (from dark to light)
            lightness = 0.35 + 0.5 * (i / max(n_elements - 1, 1))  # Range ~[0.35, 0.85]
            varied_rgb = colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2])
            hex_color = to_hex(varied_rgb)
            color_map[element.name] = hex_color

    return color_map


def build_structural_frame(
    input_data_elements: InputData_StructuralElements,
    grid: RegularGrid,
    default_interpolation: InterpolationMethod = InterpolationMethod.ORDINARY_KRIGING,
    fault_frame: Optional["FaultFrame"] = None,
) -> StructuralFrame:
    """Construct a :class:`StructuralFrame` from mapping, grid info, and input_data.

    Parameters
    ----------
    input_data_elements : InputData_StructuralElements
        Input data containing mapping, surface points, and optional orientations.
    grid : RegularGrid
        Model grid.
    default_interpolation : InterpolationMethod, default ``ORDINARY_KRIGING``
        Interpolator assigned to each group (can be overridden later).
    fault_frame : FaultFrame, optional

    Returns
    -------
    StructuralFrame
        Frame with groups/elements, colors assigned, grid and inputs attached.

    Args:
        input_data_elements:
        input_data_elements:
    """
    # Collect input_data
    surface_points = input_data_elements.surface_points.copy()
    if hasattr(input_data_elements, 'orientations'):
        orientations = input_data_elements.orientations.copy()
    else:
        orientations = None
    mapping_object = input_data_elements.mapping_object.copy()

    # Normalize column names for consistency
    surface_points = surface_points.rename(columns=str.strip)
    if orientations is not None:
        orientations = orientations.rename(columns=str.strip)

    # Validate required columns
    required_surface_cols = {"X", "Y", "Z", "formation"}
    if not required_surface_cols.issubset(surface_points.columns):
        missing = required_surface_cols - set(surface_points.columns)
        raise ValueError(f"Surface points missing required columns: {missing}")

    if orientations is not None:
        required_orientation_cols = {"X", "Y", "Z", "G_x", "G_y", "G_z", "formation"}
        if not required_orientation_cols.issubset(orientations.columns):
            missing = required_orientation_cols - set(orientations.columns)
            raise ValueError(f"Orientations missing required columns: {missing}")

    # Turn every value in mapping_object into a tuple to ensure consistency for loops and itertools
    for k, v in mapping_object.items():
        if not isinstance(v, tuple):
            mapping_object[k] = (v,)

    # Collect all unique element names
    all_element_names = list(itertools.chain.from_iterable(mapping_object.values()))
    unique_elements = list(dict.fromkeys(all_element_names))  # preserve order

    # Validate that each element has at least 1 surface point
    for elem in unique_elements:
        count = surface_points[surface_points["formation"] == elem].shape[0]
        if count == 0:
            raise ValueError(f"No surface points found for structural element '{elem}'")

    # If orientations are provided, ensure each group has at least one relevant entry
    if orientations is not None:
        for group_name, element_names in mapping_object.items():
            group_orient = orientations[orientations["formation"].isin(element_names)]
            if group_orient.empty:
                raise ValueError(f"No orientations found for structural group '{group_name}'")

    # Build groups first
    group_objects: list[StructuralGroup] = []
    for group_name, element_names in mapping_object.items():
        elements = [StructuralElement(name=name) for name in element_names]
        group = StructuralGroup(
            name=group_name,
            structural_elements=elements,
        )

        # Initialize interpolation context
        group_surface_points = surface_points[
            surface_points["formation"].isin(element_names)
        ][["X", "Y", "Z"]].to_numpy()
        group.update_interpolation_context(points=group_surface_points)

        group.set_interpolation_method(default_interpolation)
        group_objects.append(group)
        group._scalar_field = np.zeros(grid.resolution, dtype=float)

    # Generate colors AFTER groups exist
    color_map = generate_grouped_colors_per_element(group_objects)

    # Now assign colors to each element
    element_objects: Dict[str, StructuralElement] = {}
    for group in group_objects:
        for elem in group.structural_elements:
            elem.set_color(color_map[elem.name])
            element_objects[elem.name] = elem

    frame = StructuralFrame(structural_groups=group_objects)
    frame._grid = grid
    frame._surface_points = surface_points
    frame._orientations = orientations

    if fault_frame is not None:
        frame.set_fault_frame(fault_frame)

    return frame


def assign_domain_ids_to_points(grid: RegularGrid, domain_map: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["domain_id"] = pd.Series(dtype=int)
        return out

    coords = df[["X", "Y", "Z"]].values
    indices = grid.xyz_to_indices(coords).astype(int)  # [N,3] in [X,Y,Z]

    # Clamp to bounds for domain_map [X,Y,Z]
    indices[:, 0] = np.clip(indices[:, 0], 0, domain_map.shape[0] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, domain_map.shape[1] - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, domain_map.shape[2] - 1)

    domain_ids = domain_map[indices[:, 0], indices[:, 1], indices[:, 2]]

    out = df.copy()
    out["domain_id"] = domain_ids
    return out


def effective_domain_components_for_group(
    domain_ids: np.ndarray,
    faults: list,
    fault_activity: dict[str, int],
    group_idx: int,
) -> list[set[int]]:
    """Return list of merged domain-id components for a given group.

    Assumes fault_activity[fault.name] stores the *youngest affected group index*.
    Fault affects group iff group_idx >= youngest_idx.
    """
    parent = {int(d): int(d) for d in domain_ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f in faults:
        youngest_idx = fault_activity[f.name]
        is_active = group_idx >= youngest_idx

        if is_active:
            continue  # ACTIVE: keep this fault split

        pairs = f.get_domain_pairs()
        if not pairs:
            continue

        # INACTIVE: merge only the adjacent domain pairs across this fault
        for a, b in pairs:
            union(int(a), int(b))

    comps: dict[int, set[int]] = {}
    for d in domain_ids:
        r = find(int(d))
        comps.setdefault(r, set()).add(int(d))
    return list(comps.values())

