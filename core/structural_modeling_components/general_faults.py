"""
Pipeline utilities for fault modeling and fault domaining.

This module provides:
- Fault interpolation (GemPy Universal CoKriging variant) to obtain scalar fields and isovalues
- Domain map generation by iteratively splitting existing domains using fault masks
- Cross-cut detection using overlapping isovalue bands
- Domain adjacency (domain pair) inference via a thin band around the fault surface

Axis-order convention notes
---------------------------
There is a mix of XYZ and ZYX conventions in this file (kept as-is).
When you see transposes like `.transpose(2, 1, 0)`, those are converting between
(Z, Y, X) and (X, Y, Z) representations. The code preserves the existing convention.
"""

from __future__ import annotations

from typing import Optional, TypeAlias, FrozenSet

import numpy as np
import numpy.typing as npt
import pandas as pd
import gempy as gp

from core.object_components import InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    RegularGrid,
)
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultFrame,
    FaultElement,
)

from core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction import marching_cubes

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
IntArray: TypeAlias = npt.NDArray[np.integer]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
Pairs: TypeAlias = FrozenSet[tuple[int, int]]


def check_fault_crosscuts_via_isovalue_bands(
    fault_frame: FaultFrame,
    thickness_world: float | None = None,
    voxels: float = 1.0,
    use_gradient: bool = True,
) -> None:
    """
    Detect cross-cutting faults by overlapping 'isovalue bands' around each fault's own scalar isovalue.

    For each fault i with scalar field φ_i and isovalue L_i (fault.scalar_value):
        band_i = |φ_i - L_i| <= tol_scalar_i

    If any voxel satisfies band_i & band_j for i!=j, the faults cross-cut.

    Parameters
    ----------
    fault_frame : FaultFrame
        FaultFrame containing fault elements and an assigned grid.
    thickness_world : float | None
        Desired half-thickness (in world units) of each isovalue band.
        If None, it is computed as `voxels * min(grid.spacing)`.
    voxels : float
        If `thickness_world` is None, uses this many voxels (based on min spacing) as the half-thickness.
    use_gradient : bool
        If True (recommended), convert world thickness to scalar tolerance per-fault using
        that fault's median gradient magnitude:
            tol_scalar_i = thickness_world * median(|∇φ_i|)
        If False, assumes φ is approximately a signed distance function and uses:
            tol_scalar_i = thickness_world

    Raises
    ------
    ValueError
        If any pair of faults' bands overlap (i.e., cross-cut is detected).
    """
    if fault_frame.grid is None:
        raise ValueError("FaultFrame grid must be set.")
    spacing = getattr(fault_frame.grid, "spacing", None)
    if spacing is None:
        raise ValueError("Grid.spacing must be defined to compute band thickness.")

    # Determine band thickness in world units
    if thickness_world is None:
        thickness_world = float(voxels) * float(np.min(spacing))

    # Collect faults that have scalar fields and an isovalue
    faults: list[tuple[str, np.ndarray, float]] = []
    for f in fault_frame.fault_elements:
        field = getattr(f, "scalar_field", None)
        level = getattr(f, "scalar_value", None)
        if field is None or level is None:
            continue
        if not isinstance(field, np.ndarray) or field.size == 0:
            continue
        faults.append((f.name, field, float(level)))

    if len(faults) < 2:
        return  # nothing to compare

    # Compute per-fault scalar tolerances from world thickness
    tol_scalar: list[float] = []
    for _, fld, _ in faults:
        if use_gradient:
            # gradient in scalar units per meter along each axis
            gx, gy, gz = np.gradient(fld, *spacing, edge_order=1)
            grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
            med = float(np.nanmedian(grad_mag)) if np.isfinite(grad_mag).any() else 0.0
            # guard against tiny gradients
            if med <= 1e-12:
                med = 1e-12
            tol_scalar.append(thickness_world * med)
        else:
            # assume φ is approx. signed distance
            tol_scalar.append(thickness_world)

    # Build boolean bands once
    bands: list[tuple[str, npt.NDArray[np.bool_]]] = []
    for (nm, fld, level), ts in zip(faults, tol_scalar):
        band = np.abs(fld - level) <= ts
        bands.append((nm, band))

    # Pairwise overlap tests
    for i in range(len(bands)):
        name_i, band_i = bands[i]
        if not band_i.any():
            continue
        for j in range(i + 1, len(bands)):
            name_j, band_j = bands[j]
            if not band_j.any():
                continue
            if np.any(band_i & band_j):
                raise ValueError(
                    f"❌ Fault '{name_i}' crosscuts fault '{name_j}' (isovalue-band overlap)."
                )


def interpolate_group_universal_cokriging_for_faults(
    element: FaultElement,
    grid: RegularGrid,
    fault_surface_points_df: pd.DataFrame,
    fault_orientations_points_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Interpolate a fault scalar field using GemPy and store the result on the FaultElement.

    Notes
    -----
    - This function sets:
        - element.scalar_value (from scalar_field_at_surface_points)
        - element.scalar_field (from scalar_field_matrix reshaped to grid.resolution)
        - element.domain_mask (boolean, positive side convention)
    - Axis-order is preserved as implemented (no refactor).
    """
    if fault_surface_points_df.empty:
        raise ValueError(f"No surface points provided for {element.name}")

    if fault_orientations_points_df is None or fault_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for {element.name}")

    # --- GemPy conversion ---
    surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
        x=fault_surface_points_df.X.to_numpy(),
        y=fault_surface_points_df.Y.to_numpy(),
        z=fault_surface_points_df.Z.to_numpy(),
        names=fault_surface_points_df.formation.to_numpy(),
        nugget=np.zeros(len(fault_surface_points_df)),
        name_id_map=None,
    )

    orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
        x=fault_orientations_points_df.X.to_numpy(),
        y=fault_orientations_points_df.Y.to_numpy(),
        z=fault_orientations_points_df.Z.to_numpy(),
        G_x=fault_orientations_points_df.G_x.to_numpy(),
        G_y=fault_orientations_points_df.G_y.to_numpy(),
        G_z=fault_orientations_points_df.G_z.to_numpy(),
        names=fault_orientations_points_df.formation.to_numpy(),
        nugget=np.zeros(len(fault_orientations_points_df)),
        name_id_map=surface_data.name_id_map,
    )

    gempy_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(
        surface_data, orientation_data
    )

    geo_model = gp.create_geomodel(
        project_name="random",
        extent=list(grid.extent),
        resolution=list(grid.resolution),
        structural_frame=gempy_structural_frame,
    )

    mapping = {"fault_group": [element.name]}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)

    gp.compute_model(geo_model)

    # Set scalar value at surface and field
    element.set_scalar_value(
        float(geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0][0])
    )

    # NOTE: kept exactly as-is: reshape uses tuple(grid.resolution)
    sf = geo_model.solutions.raw_arrays.scalar_field_matrix[0].reshape(
        tuple(grid.resolution)
    )
    element.set_scalar_field(sf)

    # Domain mask convention (positive side = True)
    element.set_domain_mask(element.scalar_field > element.scalar_value)


def compute_fault_domains(
    fault_frame: FaultFrame,
) -> None:
    """
    Interpolate all faults and generate a domain map across the model grid.

    This function relies on fault.domain_mask being set by the interpolator.

    Side effects
    ------------
    - Sets `fault_frame._domain_map`
    - Sets `fault_frame._domain_masks`
    - Sets per-fault separated_domains and domain_pairs
    - Extracts unmasked fault meshes and stores them on each FaultElement
    - Performs a cross-cut sanity check at the end

    Raises
    ------
    ValueError
        If required frame inputs are missing or interpolation fails.
    RuntimeError
        If any fault does not generate a valid split or yields no adjacency pairs.
    """
    if not fault_frame.grid:
        raise ValueError("Grid must be set before domain generation.")
    if fault_frame.fault_surface_points_df is None:
        raise ValueError("Fault surface points must be set.")

    res_xyz = tuple(fault_frame.grid.resolution)  # (X,Y,Z)
    res_zyx = (res_xyz[2], res_xyz[1], res_xyz[0])  # (Z,Y,X)

    # XYZ domain map internally (recommended)
    domain_map_xyz = np.zeros(res_xyz, dtype=int)

    domain_id_counter = 1
    temp_ids: list[int] = []

    for fault in reversed(fault_frame.fault_elements):  # youngest first
        name = fault.name
        points = fault_frame.get_surface_points_for_element(name)
        orientations = fault_frame.get_orientations_for_element(name)

        interpolate_group_universal_cokriging_for_faults(
            fault_frame.get_element_by_name(name),
            fault_frame.grid,
            fault_surface_points_df=points,
            fault_orientations_points_df=orientations,
        )

        mask = fault.get_domain_mask()
        if mask is None:
            raise ValueError(f"Interpolator did not set domain_mask for fault '{name}'")

        # Force mask to XYZ deterministically
        if mask.shape == res_xyz:
            fault_mask_xyz = mask
        elif mask.shape == res_zyx:
            fault_mask_xyz = mask.transpose(2, 1, 0)
        else:
            raise ValueError(f"Fault mask has unexpected shape {mask.shape}, expected {res_xyz} or {res_zyx}")

        new_domain_map = domain_map_xyz.copy()

        for existing_id in np.unique(domain_map_xyz):
            current_mask = domain_map_xyz == existing_id
            overlap = current_mask & fault_mask_xyz
            if np.any(overlap):
                new_id = 9999 + domain_id_counter
                new_domain_map[overlap] = new_id
                temp_ids.append(new_id)
                domain_id_counter += 1

        domain_map_xyz = new_domain_map

    # Remap IDs to consecutive
    unique_ids = np.unique(domain_map_xyz)
    remap = {int(old): int(new) for new, old in enumerate(unique_ids)}
    remapped_xyz = np.vectorize(remap.get)(domain_map_xyz)

    fault_frame._domain_map = remapped_xyz  # store XYZ (no .T)

    # Store domain masks ALSO in XYZ
    fault_frame._domain_masks = {int(uid): (remapped_xyz == uid) for uid in np.unique(remapped_xyz)}

    # --- ALSO store separated_domains (global side-sets), for reporting/sanity ---
    for fault in fault_frame.fault_elements:
        mask = fault.get_domain_mask()
        # ensure mask is XYZ
        if mask.shape != domain_map_xyz.shape:
            mask = mask.transpose(2, 1, 0)

        left_ids = set(np.unique(domain_map_xyz[mask]))
        right_ids = set(np.unique(domain_map_xyz[~mask]))

        if not left_ids or not right_ids:
            raise RuntimeError(
                f"Fault '{fault.name}' does not create a valid split (empty side)."
            )

        fault.set_separated_domains((frozenset(left_ids), frozenset(right_ids)))

    for fault in fault_frame.fault_elements:
        # Ensure scalar_field is XYZ for adjacency pair inference
        sf = fault.scalar_field
        if sf.shape != domain_map_xyz.shape:
            sf = sf.transpose(2, 1, 0)

        pairs = compute_domain_pairs_from_fault_band(
            domain_map_xyz=domain_map_xyz,
            scalar_field_xyz=sf,
            scalar_value=float(fault.scalar_value),
            spacing_xyz=fault_frame.grid.spacing,
            voxels=1.0,
            use_gradient=True,
        )

        if not pairs:
            raise RuntimeError(
                f"Fault '{fault.name}' produced no domain_pairs (check band tolerance / orientation)."
            )

        fault.set_domain_pairs(pairs)

    # Store per-domain masks (kept as-is: uses remapped_map in ZYX orientation)
    # fault_frame._domain_masks = {}
    # for uid in np.unique(remapped_map):
    #     fault_frame.domain_masks[int(uid)] = remapped_map == uid

    # Extract surfaces meshes for faults
    for _, fault in enumerate(reversed(fault_frame.fault_elements)):
        vertices, edges = marching_cubes(
            fault.scalar_field,
            [fault.scalar_value],
            fault_frame.grid.spacing,
            fault_frame.grid.extent,
        )
        fault.set_mesh("unmasked", vertices[0], edges[0])

    # After all faults are processed
    check_fault_crosscuts_via_isovalue_bands(fault_frame)


def build_fault_frame(
    input_data_fault_elements: InputData_FaultElements,
    grid: RegularGrid,
) -> FaultFrame:
    """
    Build a :class:`FaultFrame` from ordered fault names, surface data, and a grid.

    Parameters
    ----------
    input_data_fault_elements : InputData_FaultElements
        Input data for the fault elements, including names, surface points, and orientations.
    grid : RegularGrid
        Model grid.

    Returns
    -------
    FaultFrame
        Fully configured fault frame with elements, colors, input data, and grid.
    """
    # Collect input data
    fault_names = input_data_fault_elements.fault_names
    fault_surface_points_df = input_data_fault_elements.fault_surface_points.copy()
    fault_orientations_df = input_data_fault_elements.fault_orientations.copy()

    # Assign default gray colors if none provided
    colors = ["#555555"] * len(fault_names)

    fault_elements: list[FaultElement] = []
    for name, color in reversed(list(zip(fault_names, colors))):
        fault = FaultElement(name=name)
        fault.set_color(color)
        fault_elements.append(fault)

    fault_frame = FaultFrame(fault_elements=fault_elements)
    fault_frame.set_surface_points_df(fault_surface_points_df)
    fault_frame.set_orientations_df(fault_orientations_df)
    fault_frame.set_grid(grid)

    return fault_frame


def compute_domain_pairs_from_fault_band(
    domain_map_xyz: np.ndarray,
    scalar_field_xyz: np.ndarray,
    scalar_value: float,
    spacing_xyz: npt.ArrayLike,
    *,
    voxels: float = 1.0,
    use_gradient: bool = True,
) -> frozenset[tuple[int, int]]:
    """
    Compute adjacent domain-id pairs across a fault by sampling a thin band around the fault surface.

    All arrays must be XYZ. Returns pairs as sorted (a, b) with a < b.

    Parameters
    ----------
    domain_map_xyz : np.ndarray
        Domain IDs in XYZ.
    scalar_field_xyz : np.ndarray
        Fault scalar field in XYZ.
    scalar_value : float
        Isovalue for the fault surface.
    spacing_xyz : np.ndarray
        Grid spacing (dx, dy, dz).
    voxels : float, default 1.0
        Half-thickness of the sampling band in voxels (converted via min spacing).
    use_gradient : bool, default True
        If True, scale scalar tolerance by median gradient magnitude (more robust).

    Returns
    -------
    frozenset[tuple[int, int]]
        Adjacent domain pairs across the fault surface.
    """
    thickness_world = float(voxels) * float(np.min(spacing_xyz))

    if use_gradient:
        gx, gy, gz = np.gradient(scalar_field_xyz, *spacing_xyz, edge_order=1)
        grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        med = float(np.nanmedian(grad_mag)) if np.isfinite(grad_mag).any() else 1e-12
        tol_scalar = thickness_world * max(med, 1e-12)
    else:
        tol_scalar = thickness_world

    band = np.abs(scalar_field_xyz - float(scalar_value)) <= tol_scalar

    # Two sides near the surface
    pos = band & (scalar_field_xyz > scalar_value)
    neg = band & (scalar_field_xyz <= scalar_value)

    pos_ids = np.unique(domain_map_xyz[pos])
    neg_ids = np.unique(domain_map_xyz[neg])

    pairs: set[tuple[int, int]] = set()
    for a in neg_ids:
        for b in pos_ids:
            ia, ib = int(a), int(b)
            if ia != ib:
                pairs.add((ia, ib) if ia < ib else (ib, ia))

    return frozenset(pairs)
