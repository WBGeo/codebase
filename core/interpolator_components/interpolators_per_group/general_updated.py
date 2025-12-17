"""
Interpolation pipeline utilities for geological structural modeling with optional fault domains.

Main stages (per-domain when faults are provided):
  1) Interpolation per structural group
  2) Age-mask computation per group
  3) Lithology block assembly
  4) (Optional) Masked surface mesh extraction for each element
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import itertools
import colorsys

from core.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_per_element

from core.structural_objects.objects_updated import (
    StructuralFrame,
    StructuralGroup,
    StructuralElement,
    InterpolationMethod,
)
from core.structural_objects.objects_updated import FaultFrame, FaultElement

from typing import Dict, Optional, Tuple

from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import (
    interpolate_group_ordinary_kriging,
)
from core.interpolator_components.interpolators_per_group.radial_basis_function_per_group import (
    interpolate_group_radial_basis_function,
)
from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import (
    interpolate_group_universal_cokriging,
)
from core.interpolator_components.interpolators_per_group.geoinr_per_group import (
    interpolate_group_geo_inr,
)
from core.interpolator_components.interpolators_per_group.loop_structural_per_group import (
    interpolate_group_loop_structural,
)

from core.structural_objects.objects_updated import InterpolationMethod


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
    fault_frame: FaultFrame,
    *,
    crop_to_domain: bool = True,
) -> None:
    """Interpolate each structural group independently inside each fault domain.

    This version optionally **crops computation to the domain's bounding box** to
    speed up interpolation while still storing full-grid scalar fields per domain.

    Results are stored in per-domain slots on the groups/elements via
    - ``group.set_scalar_field_for_domain(domain_id, field)``
    - ``element.set_scalar_value_for_domain(domain_id, value)``

    Assumes each dispatch function returns a tuple:
    ``(scalar_field: np.ndarray, scalar_values_by_element: Dict[str, float])``.
    The input groups are not mutated besides the explicit setters above.

    Parameters
    ----------
    frame : StructuralFrame
        Structural frame containing groups, elements, grid and input data.
    fault_frame : FaultFrame
        Fault frame providing a 3D ``domain_map`` with domain identifiers.
    crop_to_domain : bool, default True
        If ``True``, perform the interpolation on the minimal axis-aligned sub-grid
        that bounds the domain voxels, then write results back into a full-size
        scalar field. This preserves storage layout and downstream behavior but
        reduces compute for small domains.
    """
    domain_map = fault_frame.domain_map
    domain_ids = np.unique(domain_map)

    # Tag inputs with domain ids
    sp_in_domain = assign_domain_ids_to_points(frame.grid, domain_map, frame.surface_points)
    ori_in_domain = (
        assign_domain_ids_to_points(frame.grid, domain_map, frame.orientations)
        if frame.orientations is not None
        else None
    )

    for domain_id in domain_ids:
        # Filter input for this domain
        sp_filtered_all = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
        ori_filtered_all = (
            ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
            if ori_in_domain is not None
            else None
        )

        # Determine computation grid (full grid or cropped sub-grid)
        use_grid = frame.grid
        bbox = None
        if crop_to_domain:
            bbox = compute_domain_bbox_indices(domain_map, domain_id)  # (kz0,kz1, ky0,ky1, kx0,kx1)
            if bbox is not None:
                use_grid = build_subgrid_from_bbox(frame.grid, bbox)

        # Interpolate group-by-group
        for group in frame.structural_groups:
            # Formations in this group
            group_formations = [e.name for e in group.structural_elements]

            # Filter to this group’s formations
            sp_filtered = sp_filtered_all[sp_filtered_all["formation"].isin(group_formations)]
            ori_filtered: Optional[pd.DataFrame] = None
            if ori_filtered_all is not None:
                ori_filtered = ori_filtered_all[ori_filtered_all["formation"].isin(group_formations)]

            # Sanity: at least 2 points per element (kept as-is)
            for elem in group.structural_elements:
                n_pts = (sp_filtered["formation"] == elem.name).sum()
                if n_pts < 2:
                    raise ValueError(
                        f"❌ Not enough surface points for '{elem.name}' in domain {domain_id} (have {n_pts})"
                    )

            # Pick interpolator
            method = group.interpolation_method
            if method not in interpolate_dispatch:
                raise ValueError(f"Unsupported interpolation method: {method}")

            # ---- CALL: must return scalar_field + dict of element scalar values ----
            scalar_field_sub, scalar_values = interpolate_dispatch[method](
                group=group,
                grid=use_grid,
                group_surface_points_df=sp_filtered,
                group_orientations_points_df=ori_filtered,
            )

            # Write sub-grid result back into a full-size scalar field if cropped
            if crop_to_domain and bbox is not None:
                full_shape = tuple(frame.grid.resolution.astype(int))
                scalar_field_full = np.empty(full_shape, dtype=scalar_field_sub.dtype)
                scalar_field_full.fill(np.nan)

                kz0, kz1, ky0, ky1, kx0, kx1 = bbox
                # Note: arrays are in [Z, Y, X] order
                scalar_field_full[kz0:kz1 + 1, ky0:ky1 + 1, kx0:kx1 + 1] = scalar_field_sub
                scalar_field = scalar_field_full
            else:
                scalar_field = scalar_field_sub

            # Store per-domain scalar field on the group
            group.set_scalar_field_for_domain(domain_id, scalar_field)

            # Store per-domain scalar values on elements
            for elem in group.structural_elements:
                if elem.name not in scalar_values:
                    raise ValueError(
                        "Interpolator did not return a scalar value for element "
                        f"'{elem.name}' in group '{group.name}'."
                    )
                elem.set_scalar_value_for_domain(domain_id, float(scalar_values[elem.name]))


# def run_interpolation_with_fault_domains(
#     frame: StructuralFrame,
#     fault_frame: FaultFrame,
# ) -> None:
#     """Interpolate each structural group independently inside each fault domain.
#
#     Results are stored in per-domain slots on the groups/elements via
#     - ``group.set_scalar_field_for_domain(domain_id, field)``
#     - ``element.set_scalar_value_for_domain(domain_id, value)``
#
#     Assumes each dispatch function returns a tuple:
#     ``(scalar_field: np.ndarray, scalar_values_by_element: Dict[str, float])``.
#     The input groups are not mutated besides the explicit setters above.
#
#     Parameters
#     ----------
#     frame : StructuralFrame
#         Structural frame containing groups, elements, grid and input data.
#     fault_frame : FaultFrame
#         Fault frame providing a 3D ``domain_map`` with domain identifiers.
#     """
#     domain_map = fault_frame.domain_map
#     domain_ids = np.unique(domain_map)
#
#     # Tag inputs with domain ids
#     sp_in_domain = assign_domain_ids_to_points(frame.grid, domain_map, frame.surface_points)
#     ori_in_domain = (
#         assign_domain_ids_to_points(frame.grid, domain_map, frame.orientations)
#         if frame.orientations is not None
#         else None
#     )
#
#     for domain_id in domain_ids:
#         # Filter input for this domain
#         sp_filtered_all = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
#         ori_filtered_all = (
#             ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
#             if ori_in_domain is not None
#             else None
#         )
#
#         # Interpolate group-by-group
#         for group in frame.structural_groups:
#             # Formations in this group
#             group_formations = [e.name for e in group.structural_elements]
#
#             # Filter to this group’s formations
#             sp_filtered = sp_filtered_all[sp_filtered_all["formation"].isin(group_formations)]
#             ori_filtered: Optional[pd.DataFrame] = None
#             if ori_filtered_all is not None:
#                 ori_filtered = ori_filtered_all[ori_filtered_all["formation"].isin(group_formations)]
#
#             # Sanity: at least 2 points per element (kept as-is)
#             for elem in group.structural_elements:
#                 n_pts = (sp_filtered["formation"] == elem.name).sum()
#                 if n_pts < 2:
#                     raise ValueError(
#                         f"❌ Not enough surface points for '{elem.name}' in domain {domain_id} (have {n_pts})"
#                     )
#
#             # Pick interpolator
#             method = group.interpolation_method
#             if method not in interpolate_dispatch:
#                 raise ValueError(f"Unsupported interpolation method: {method}")
#
#             # ---- CALL: must return scalar_field + dict of element scalar values ----
#             scalar_field, scalar_values = interpolate_dispatch[method](
#                 group=group,
#                 grid=frame.grid,
#                 group_surface_points_df=sp_filtered,
#                 group_orientations_points_df=ori_filtered,
#             )
#
#             # Store per-domain scalar field on the group
#             group.set_scalar_field_for_domain(domain_id, scalar_field)
#
#             # Store per-domain scalar values on elements
#             for elem in group.structural_elements:
#                 if elem.name not in scalar_values:
#                     raise ValueError(
#                         "Interpolator did not return a scalar value for element "
#                         f"'{elem.name}' in group '{group.name}'."
#                     )
#                 elem.set_scalar_value_for_domain(domain_id, float(scalar_values[elem.name]))


# --- 2) Age masks per domain --------------------------------------------------

def set_scalar_masks_per_domain(frame: StructuralFrame, fault_frame: FaultFrame) -> None:
    """Compute and set age masks per group *per domain*.

    Oldest group => mask of all ``True``; otherwise mask is ``sf >= oldest_element.scalar_value``
    within that group/domain.

    Parameters
    ----------
    frame : StructuralFrame
        Model frame with groups and per-domain scalar fields/values set.
    fault_frame : FaultFrame
        Fault frame providing a 3D ``domain_map`` with domain identifiers.
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
                # Oldest group: everywhere True
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


# --- 3) Combine all domains to final lithology block --------------------------

def compute_lithology_block_with_domains(frame: StructuralFrame, fault_frame: FaultFrame) -> np.ndarray:
    """Combine domain-specific group results into a single lithology block.

    Honors age masks and domain masks. Element IDs are global (same color across domains).

    Parameters
    ----------
    frame : StructuralFrame
        Frame containing groups/elements and per-domain fields.
    fault_frame : FaultFrame
        Provides the ``domain_map`` used to clip each domain.

    Returns
    -------
    np.ndarray
        Final lithology volume (integer IDs) with shape ``frame.grid.resolution``.
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
        dommask = fault_frame.domain_map == d
        domain_lith = np.zeros(shape, dtype=int)

        # Process groups from oldest -> youngest so younger overwrites older
        for group in reversed(frame.structural_groups):
            sf = group.get_scalar_field_for_domain(d)
            gm = group.get_mask_for_domain(d)
            if sf is None or gm is None:
                # No data in this domain for this group => skip
                continue

            group_block = np.zeros(shape, dtype=int)

            # Fill elements in natural (oldest->youngest) order; use "== 0" so younger has priority
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


# --- 4) Extract per-domain “masked” meshes for each element -------------------

def extract_all_meshes_per_domain(
    frame: StructuralFrame,
    fault_frame: FaultFrame,
    grid_spacing: np.ndarray,
    extent: np.ndarray,
) -> None:
    """Extract and store per-domain masked surface meshes for every element.

    Mask used for extraction = (domain mask) & (age mask of the *previous* group),
    mirroring the approach that enforces unconformity truncation.

    Parameters
    ----------
    frame : StructuralFrame
        Frame with groups/elements and per-domain values.
    fault_frame : FaultFrame
        Fault frame providing the 3D domain map.
    grid_spacing : np.ndarray
        Voxel spacing used by marching cubes per element.
    extent : np.ndarray
        Spatial extent passed to marching cubes.
    """
    domain_ids = np.unique(fault_frame.domain_map)

    for d in domain_ids:
        dommask = fault_frame.domain_map == d

        for i, group in enumerate(frame.structural_groups):
            sf = group.get_scalar_field_for_domain(d)
            if sf is None:
                continue

            # Age-mask logic mirroring the original function
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
                    sf.T,  # keep transposition exactly as before
                    sval,
                    grid_spacing,
                    extent,
                    mask=mc_mask.T,  # keep transposition exactly as before
                )

                # Store per-domain masked mesh
                elem.set_mesh_for_domain(d, "masked", verts, faces)


# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------

def combined_interpolator_with_domains(
    frame: StructuralFrame,
    fault_frame: Optional[FaultFrame] = None,
    *,
    extract_meshes: bool = True,
    verbose: bool = True,
) -> np.ndarray:
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
        Structural frame containing grid, groups, elements, and input data.
    fault_frame : Optional[FaultFrame]
        If ``None``, a synthetic single domain (all zeros) is used.
    extract_meshes : bool, default True
        Whether to extract masked meshes at the end.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    np.ndarray
        Final lithology volume (IDs are global across domains).
    """

    # --- prepare domain map (supports "no faults" case) ---
    if fault_frame is None:
        # synthetic single domain
        dom_map = np.zeros(tuple(frame.grid.resolution.astype(int)), dtype=int)

        class _TmpFF:  # simple shim with a "domain_map" attribute
            domain_map = dom_map

        fault_frame = _TmpFF()  # type: ignore[assignment]
        if verbose:
            print("ℹ️ No fault_frame provided — running in single-domain mode.")

    # --- 1) per-domain interpolation, store results into per-domain slots ---
    if verbose:
        print("① Interpolation per domain ...")
    run_interpolation_with_fault_domains(
        frame=frame,
        fault_frame=fault_frame,
    )

    # --- 2) per-domain age masks ---
    if verbose:
        print("② Computing age masks per domain ...")
    set_scalar_masks_per_domain(frame, fault_frame)  # type: ignore[arg-type]

    # --- 3) final lithology block combining domains ---
    if verbose:
        print("③ Building final lithology block ...")
    frame._lith_block = compute_lithology_block_with_domains(frame, fault_frame)  # type: ignore[arg-type]

    # --- 4) per-domain meshes (optional) ---
    if extract_meshes:
        if verbose:
            print("④ Extracting per-domain masked meshes ...")
        extract_all_meshes_per_domain(
            frame=frame,
            fault_frame=fault_frame,  # type: ignore[arg-type]
            grid_spacing=frame.grid.spacing,
            extent=frame.grid.extent,
        )

    if verbose:
        print("✅ Pipeline complete.")

# -----------------------------------------------------------------------------
# Builders / helpers
# -----------------------------------------------------------------------------

def compute_domain_bbox_indices(domain_map: np.ndarray, domain_id: int):
    """Return the tight [Z, Y, X] index bbox for a given domain id.


    Parameters
    ----------
    domain_map : np.ndarray
    3D array shaped like the grid resolution, indexed as ``[Z, Y, X]``.
    domain_id : int
    Target domain identifier.


    Returns
    -------
    tuple[int, int, int, int, int, int] | None
    ``(kz0, kz1, ky0, ky1, kx0, kx1)`` inclusive bounds, or ``None`` if domain is empty.
    """
    mask = domain_map == domain_id
    if not np.any(mask):
        return None


    # Find ranges along each axis
    z_any = mask.any(axis=(1, 2))
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))


    kz0, kz1 = np.where(z_any)[0][[0, -1]]
    ky0, ky1 = np.where(y_any)[0][[0, -1]]
    kx0, kx1 = np.where(x_any)[0][[0, -1]]
    return int(kz0), int(kz1), int(ky0), int(ky1), int(kx0), int(kx1)


def build_subgrid_from_bbox(grid: RegularGrid, bbox: tuple) -> RegularGrid:
    """Create a ``RegularGrid`` that spans an index-aligned bbox of the parent grid.


    The parent grid uses spacing ``grid.spacing`` and has total ``resolution``. The returned
    subgrid shares the same spacing; its extent is computed from the parent's extent and the
    inclusive index bounds ``bbox`` (in [Z, Y, X] order).


    Notes
    -----
    - This function **does not** modify the parent grid.
    - It assumes ``grid.extent`` is ordered as ``[xmin, xmax, ymin, ymax, zmin, zmax]`` and
    spacing is ``[dx, dy, dz]``. If your ``RegularGrid`` uses a different convention, adjust
    the extent calculation accordingly (the interpolation results written back still preserve
    the original storage layout).
    """
    kz0, kz1, ky0, ky1, kx0, kx1 = bbox


    dx, dy, dz = grid.spacing
    xmin, xmax, ymin, ymax, zmin, zmax = grid.extent


    # Convert inclusive index bounds to physical extents (upper bound is exclusive in space)
    sub_xmin = xmin + kx0 * dx
    sub_xmax = xmin + (kx1 + 1) * dx
    sub_ymin = ymin + ky0 * dy
    sub_ymax = ymin + (ky1 + 1) * dy
    sub_zmin = zmin + kz0 * dz
    sub_zmax = zmin + (kz1 + 1) * dz


    sub_extent = np.array([sub_xmin, sub_xmax, sub_ymin, sub_ymax, sub_zmin, sub_zmax], dtype=float)
    sub_resolution = np.array([kx1 - kx0 + 1, ky1 - ky0 + 1, kz1 - kz0 + 1], dtype=int)

    return RegularGrid(extent=sub_extent, resolution=sub_resolution)


def build_fault_frame(
    fault_surface_points_df: pd.DataFrame,
    fault_orientations_df: pd.DataFrame,
    fault_names: list,  # youngest to oldest
    grid: RegularGrid,
    colors: list = None,
) -> FaultFrame:
    """Build a :class:`FaultFrame` from ordered fault names, surface data, and a grid.

    Parameters
    ----------
    fault_surface_points_df : pd.DataFrame
        Columns: ``['X', 'Y', 'Z', 'formation']``.
    fault_orientations_df : pd.DataFrame
        Columns: ``['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'formation']``.
    fault_names : list
        Fault names ordered from youngest to oldest (input convention).
    grid : RegularGrid
        Model grid.
    colors : list, optional
        Hex colors for faults in the same order as ``fault_names``. Defaults to dark grey.

    Returns
    -------
    FaultFrame
        A fully configured fault frame with elements, colors, input data, and grid.
    """
    if colors is None:
        colors = ["#555555"] * len(fault_names)
    if len(colors) != len(fault_names):
        raise ValueError("Length of colors must match fault_names")

    fault_elements = []
    # Internally we build oldest -> youngest, preserving the original behavior
    for name, color in reversed(list(zip(fault_names, colors))):
        fault = FaultElement(name=name)
        fault.set_color(color)
        fault_elements.append(fault)

    fault_frame = FaultFrame(fault_elements=fault_elements)
    fault_frame.set_surface_points_df(fault_surface_points_df)
    fault_frame.set_orientations_df(fault_orientations_df)
    fault_frame.set_grid(grid)

    return fault_frame


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
    mapping_object: Dict[str, Tuple[str, ...]],
    extent: np.ndarray,
    resolution: np.ndarray,
    surface_points: pd.DataFrame,
    orientations: Optional[pd.DataFrame] = None,
    default_interpolation: InterpolationMethod = InterpolationMethod.ORDINARY_KRIGING,
) -> StructuralFrame:
    """Construct a :class:`StructuralFrame` from mapping, grid info, and data.

    Parameters
    ----------
    mapping_object : dict[str, tuple[str, ...]]
        Mapping of group name -> tuple of element names in *youngest to oldest* order.
    extent : np.ndarray
        Grid extent passed to :class:`RegularGrid`.
    resolution : np.ndarray
        Grid resolution passed to :class:`RegularGrid`.
    surface_points : pd.DataFrame
        Surface points with columns ``['X', 'Y', 'Z', 'formation']``.
    orientations : pd.DataFrame, optional
        Orientations with columns ``['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'formation']``.
    default_interpolation : InterpolationMethod, default ``ORDINARY_KRIGING``
        Interpolator assigned to each group (can be overridden later).

    Returns
    -------
    StructuralFrame
        Frame with groups/elements, colors assigned, grid and inputs attached.
    """
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
    group_objects: list[StructuralGroup] = []
    for group_name, element_names in mapping_object.items():
        elements = [StructuralElement(name=name) for name in element_names]
        group = StructuralGroup(
            name=group_name,
            interpolation_method=None,  # placeholder—kept as-is
            scalar_field=np.array([]),
            structural_elements=elements,
        )
        group.set_interpolation_method(default_interpolation)
        group_objects.append(group)

    # Generate colors AFTER groups exist
    color_map = generate_grouped_colors_per_element(group_objects)

    # Now assign colors to each element
    element_objects: Dict[str, StructuralElement] = {}
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


def assign_domain_ids_to_points(
    grid: RegularGrid, domain_map: np.ndarray, df: pd.DataFrame
) -> pd.DataFrame:
    """Assign a domain ID to each (X, Y, Z) point from ``domain_map``.

    Parameters
    ----------
    grid : RegularGrid
        The model grid providing ``xyz_to_indices``.
    domain_map : np.ndarray
        3D array with domain IDs, shaped ``[Z, Y, X]``.
    df : pd.DataFrame
        DataFrame with columns ``'X', 'Y', 'Z'``.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with a new integer column ``'domain_id'``.
        If the input is empty, the same DataFrame is returned with an empty column of dtype int.
    """
    if df.empty:
        df["domain_id"] = pd.Series(dtype=int)
        return df

    coords = df[["X", "Y", "Z"]].values
    indices = grid.xyz_to_indices(coords)  # Shape: [N, 3], order: [X, Y, Z]

    # Clamp to bounds (note reversed order for domain_map indexing [Z, Y, X])
    for dim in range(3):
        indices[:, dim] = np.clip(indices[:, dim], 0, domain_map.shape[2 - dim] - 1)

    # Reverse the indexing to [Z, Y, X]
    domain_ids = domain_map[indices[:, 2], indices[:, 1], indices[:, 0]]

    out = df.copy()
    out["domain_id"] = domain_ids
    return out
