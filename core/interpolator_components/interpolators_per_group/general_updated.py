import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import itertools
import colorsys

from core.object_components import InputData, GeomodelResults
from core.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes_new

import gempy as gp

from typing import Callable, Dict, Tuple, Optional

from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import \
    interpolate_group_ordinary_kriging
from core.interpolator_components.interpolators_per_group.radial_basis_function_per_group import \
    interpolate_group_radial_basis_function
from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import \
    interpolate_group_universal_cokriging
from core.interpolator_components.interpolators_per_group.loop_structural_per_group import \
    interpolate_group_loop_structural
from core.interpolator_components.interpolators_per_group.geoinr_per_group import interpolate_group_geo_inr
from core.interpolator_components.interpolators_per_group.geo_ml_per_group import interpolate_group_geoml
from core.interpolator_components.interpolators_per_group.inverse_distance_per_group import interpolate_group_idw

from core.structural_objects.objects_updated import StructuralFrame, StructuralGroup, StructuralElement, InterpolationMethod
from core.structural_objects.objects_updated import FaultFrame, FaultElement

from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple

# --- 1) Orchestrate per-domain interpolation and store into per-domain slots ---

def run_interpolation_with_fault_domains(
    frame,
    fault_frame,
    interpolate_group_dispatch: Dict,  # {InterpolationMethod: callable(group, grid, sp_df, ori_df)}
    assign_domain_ids_to_points: Callable,  # function(grid, domain_map, df) -> df_with_domain_col
) -> None:
    """
    Interpolate every structural group independently inside each fault domain and
    store results into per-domain slots on the groups/elements.
    """
    # Tag input data with domain ids
    domain_map = fault_frame.domain_map
    domain_ids = np.unique(domain_map)

    sp_in_domain = assign_domain_ids_to_points(frame.grid, domain_map, frame.surface_points)
    ori_in_domain = assign_domain_ids_to_points(frame.grid, domain_map, frame.orientations) \
        if frame.orientations is not None else None

    for domain_id in domain_ids:
        # Filter input for this domain
        sp_filtered = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
        ori_filtered = None
        if ori_in_domain is not None:
            ori_filtered = ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")

        # Sanity: at least a couple of points per element present in this domain
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                n_pts = (sp_filtered["formation"] == elem.name).sum()
                if n_pts < 2:
                    raise ValueError(f"❌ Not enough surface points for '{elem.name}' in domain {domain_id}")

        # Work on a copy so interpolators can keep using legacy single-setters internally
        domain_groups = deepcopy(frame.structural_groups)

        # Interpolate per group with the correct method
        for g_copy in domain_groups:
            sp_group = sp_filtered  # your group interpolators already filter by formation internally
            ori_group = ori_filtered if ori_filtered is not None else None

            method = g_copy.interpolation_method
            if method not in interpolate_group_dispatch:
                raise ValueError(f"Unsupported interpolation method: {method}")

            interpolate_group_dispatch[method](
                group=g_copy,
                grid=frame.grid,
                group_surface_points_df=sp_group,
                group_orientations_points_df=ori_group,
            )

        # Copy results back into the original groups, storing PER DOMAIN
        for g_copy in domain_groups:
            g_orig = next(g for g in frame.structural_groups if g.name == g_copy.name)
            # scalar field per domain
            g_orig.set_scalar_field_for_domain(domain_id, g_copy.scalar_field)
            # scalar values per domain
            for e_copy in g_copy.structural_elements:
                e_orig = next(e for e in g_orig.structural_elements if e.name == e_copy.name)
                e_orig.set_scalar_value_for_domain(domain_id, float(e_copy.scalar_value))


# --- 2) Age masks per domain ---

def set_scalar_masks_per_domain(frame, fault_frame) -> None:
    """
    Compute and set age masks per group *per domain*.
    Oldest group => full True; others => field >= oldest_element.scalar_value (within that group/domain).
    """
    domain_ids = np.unique(fault_frame.domain_map)

    if not frame.structural_groups:
        raise ValueError("The structural frame contains no groups.")

    groups = frame.structural_groups

    for d in domain_ids:
        for i, group in enumerate(groups):
            sf = group.get_scalar_field_for_domain(d)
            if sf is None:
                raise ValueError(f"Domain {d}: Group '{group.name}' has no scalar field set.")

            if i == len(groups) - 1:
                mask = np.ones_like(sf, dtype=bool)
            else:
                if not group.structural_elements:
                    raise ValueError(f"Domain {d}: Group '{group.name}' has no structural elements.")
                oldest = group.structural_elements[-1]
                sval = oldest.get_scalar_value_for_domain(d)
                if sval is None:
                    raise ValueError(
                        f"Domain {d}: Oldest element '{oldest.name}' in group '{group.name}' has no scalar value."
                    )
                mask = sf >= sval

            group.set_mask_for_domain(d, mask)


# --- 3) Combine all domains to final lithology block ---

def compute_lithology_block_with_domains(frame, fault_frame) -> np.ndarray:
    """
    Combine domain-specific group results into a single lithology block,
    honoring age masks and domain masks. Element IDs are global (same color across domains).
    """
    shape = tuple(frame.grid.resolution.astype(int))
    final_lith = np.zeros(shape, dtype=int)

    # Assign stable global IDs if not already set: oldest -> youngest
    current_id = 1
    for group in reversed(frame.structural_groups):
        for element in reversed(group.structural_elements):
            if element.id is None:
                element.set_id(current_id)
                current_id += 1

    domain_ids = np.unique(fault_frame.domain_map)

    # Process domains independently
    for d in domain_ids:
        dommask = (fault_frame.domain_map == d)
        domain_lith = np.zeros(shape, dtype=int)

        # Process groups from oldest -> youngest so younger overwrites older
        for group in reversed(frame.structural_groups):
            sf = group.get_scalar_field_for_domain(d)
            gm = group.get_mask_for_domain(d)
            if sf is None or gm is None:
                # No data in this domain for this group => skip
                continue

            group_block = np.zeros(shape, dtype=int)

            # Fill elements in natural (oldest->youngest) order; use "== 0" to give younger priority
            for elem in group.structural_elements:
                sval = elem.get_scalar_value_for_domain(d)
                if sval is None:
                    continue
                write_mask = (sf >= sval) & (group_block == 0)
                group_block[write_mask] = elem.id

            # Apply age mask then domain mask
            group_block = np.where(gm, group_block, 0)
            group_block = np.where(dommask, group_block, 0)

            # Younger groups overwrite older
            domain_lith = np.where(group_block > 0, group_block, domain_lith)

        # Write this domain into final (domain masks are disjoint)
        final_lith[dommask] = domain_lith[dommask]

    return final_lith


# --- 4) Extract per-domain “masked” meshes for each element ---

def extract_all_meshes_per_domain(
    frame,
    fault_frame,
    grid_spacing: np.ndarray,
    extent: np.ndarray,
    marching_cubes_per_element: Callable,
) -> None:
    """
    For every domain, extract 'masked' surface meshes for every element and store them
    with element.set_mesh_for_domain(domain_id, 'masked', vertices, faces).

    Mask used for extraction = (domain mask) & (age mask of the *previous* group),
    mirroring your earlier approach that enforces unconformity truncation.
    """
    domain_ids = np.unique(fault_frame.domain_map)

    for d in domain_ids:
        dommask = (fault_frame.domain_map == d)

        for i, group in enumerate(frame.structural_groups):
            sf = group.get_scalar_field_for_domain(d)
            if sf is None:
                continue

            # Age-mask logic mirroring your old function:
            if i == 0:
                erosion_mask = np.ones_like(sf, dtype=bool)
            else:
                prev_mask = frame.structural_groups[i - 1].get_mask_for_domain(d)
                erosion_mask = ~prev_mask if prev_mask is not None else np.ones_like(sf, dtype=bool)

            mc_mask = erosion_mask & dommask

            for elem in group.structural_elements:
                sval = elem.get_scalar_value_for_domain(d)
                if sval is None:
                    continue

                verts, faces = marching_cubes_per_element(
                    sf.T,
                    sval,
                    grid_spacing,
                    extent,
                    mask=mc_mask.T
                )

                # Store per-domain masked mesh
                elem.set_mesh_for_domain(d, "masked", verts, faces)

def combined_interpolator_with_domains(
    frame,
    fault_frame=None,
    *,
    # mapping from InterpolationMethod -> callable(group, grid, sp_df, ori_df)
    interpolate_dispatch: dict,
    # helper: tag points/orientations with domain ids
    assign_domain_ids_to_points,
    # mesh extraction callable (per-element)
    marching_cubes_per_element,
    # control steps
    extract_meshes: bool = True,
    verbose: bool = True,
):
    """
    Run the full pipeline with (optional) fault domains:
      1) per-domain interpolation (stores scalar fields/values per domain)
      2) per-domain age masks
      3) final lithology block combining domains
      4) per-domain masked surface meshes (optional)

    Returns
    -------
    lith_block : np.ndarray
        Final lithology volume (IDs are global across domains).
    """

    # --- prepare domain map (supports "no faults" case) ---
    if fault_frame is None:
        # synthetic single domain
        dom_map = np.zeros(tuple(frame.grid.resolution.astype(int)), dtype=int)
        class _TmpFF:
            domain_map = dom_map
        fault_frame = _TmpFF()
        if verbose:
            print("ℹ️ No fault_frame provided — running in single-domain mode.")

    # --- 1) per-domain interpolation, store results into per-domain slots ---
    if verbose:
        print("① Interpolation per domain ...")
    run_interpolation_with_fault_domains(
        frame=frame,
        fault_frame=fault_frame,
        interpolate_group_dispatch=interpolate_dispatch,
        assign_domain_ids_to_points=assign_domain_ids_to_points,
    )

    # --- 2) per-domain age masks ---
    if verbose:
        print("② Computing age masks per domain ...")
    set_scalar_masks_per_domain(frame, fault_frame)

    # --- 3) final lithology block combining domains ---
    if verbose:
        print("③ Building final lithology block ...")
    lith_block = compute_lithology_block_with_domains(frame, fault_frame)

    # --- 4) per-domain meshes (optional) ---
    if extract_meshes:
        if verbose:
            print("④ Extracting per-domain masked meshes ...")
        extract_all_meshes_per_domain(
            frame=frame,
            fault_frame=fault_frame,
            grid_spacing=frame.grid.spacing,
            extent=frame.grid.extent,
            marching_cubes_per_element=marching_cubes_per_element,
        )

    if verbose:
        print("✅ Pipeline complete.")
    return lith_block

def build_fault_frame(
        fault_surface_points_df: pd.DataFrame,
        fault_orientations_df: pd.DataFrame,
        fault_names: list,  # youngest to oldest
        grid: RegularGrid,
        colors: list = None
    ) -> FaultFrame:
    """
    Build a FaultFrame from ordered fault names, surface data, and a grid.

    Args:
        fault_surface_points_df (pd.DataFrame): ['X', 'Y', 'Z', 'formation'].
        fault_orientations_df (pd.DataFrame): ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'formation'].
        fault_names (list): Fault names ordered from youngest to oldest.
        grid (RegularGrid): Model grid.
        colors (list, optional): Hex colors for faults, same order. Defaults to dark grey.

    Returns:
        FaultFrame
    """
    if colors is None:
        colors = ["#555555"] * len(fault_names)
    if len(colors) != len(fault_names):
        raise ValueError("Length of colors must match fault_names")

    fault_elements = []
    for name, color in reversed(list(zip(fault_names, colors))):  # oldest to youngest internally
        fault = FaultElement(name=name)
        fault.set_color(color)
        fault_elements.append(fault)

    fault_frame = FaultFrame(fault_elements=fault_elements)
    fault_frame.set_surface_points_df(fault_surface_points_df)
    fault_frame.set_orientations_df(fault_orientations_df)
    fault_frame.set_grid(grid)

    return fault_frame

def generate_grouped_colors_per_element(groups, base_colormap="Accent"):
    """
    Generate distinct color shades for each element in each group.

    Args:
        groups: List of StructuralGroup instances.
        base_colormap: Matplotlib colormap for base group colors.

    Returns:
        dict: {element_name: hex_color}
    """
    group_cmap = plt.get_cmap(base_colormap)
    color_map = {}

    n_groups = len(groups)
    for group_idx, group in enumerate(groups):
        base_rgb = group_cmap(group_idx / n_groups)[:3]  # Get RGB triple (ignore alpha)
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
        mapping_object: Dict[str, Tuple[str, ...]],
        extent: np.ndarray,
        resolution: np.ndarray,
        surface_points: pd.DataFrame,
        orientations: Optional[pd.DataFrame] = None,
        default_interpolation: InterpolationMethod = InterpolationMethod.ORDINARY_KRIGING
) -> StructuralFrame:
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
    group_objects = []
    for group_name, element_names in mapping_object.items():
        elements = [StructuralElement(name=name) for name in element_names]
        group = StructuralGroup(
            name=group_name,
            interpolation_method=None,  # placeholder
            scalar_field=np.array([]),
            structural_elements=elements
        )
        group.set_interpolation_method(default_interpolation)
        group_objects.append(group)

    # Generate colors AFTER groups exist
    color_map = generate_grouped_colors_per_element(group_objects)

    # Now assign colors to each element
    element_objects = {}
    for group in group_objects:
        for elem in group.structural_elements:
            elem.set_color(color_map[elem.name])
            element_objects[elem.name] = elem

    # Create regular grid
    grid = RegularGrid(extent=extent, resolution=resolution)

    frame = StructuralFrame(structural_groups=group_objects)
    frame._grid = grid
    frame._surface_points = surface_points
    frame._orientations = orientations

    return frame

