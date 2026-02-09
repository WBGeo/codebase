"""
Pipeline utilities for fault modeling and fault domaining.
"""
import numpy as np
import pandas as pd
import gempy as gp

from typing import Optional, Tuple, FrozenSet

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structuralmodeling_components.structural_objects.structural_objects import FaultFrame, FaultElement

from core.utility.surface_mesh_extraction import marching_cubes_new


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

    If any voxel satisfies band_i & band_j for i!=j, the faults crosscut.

    Parameters
    ----------
    thickness_world : float | None
        Desired half-thickness (in world units, e.g. meters) of each isovalue band.
        If None, it is computed as `voxels * min(grid.spacing)`.
    voxels : float
        If `thickness_world` is None, use this many voxels (based on min spacing) as the half-thickness.
    use_gradient : bool
        If True (recommended), convert the world thickness to scalar tolerance per-fault using
        that fault's median gradient magnitude: tol_scalar_i = thickness_world * median(|∇φ_i|).
        If False, assumes φ is approximately a signed distance function and uses tol_scalar_i = thickness_world.

    Raises
    ------
    ValueError
        If any pair of faults' bands overlap (i.e., cross-cut is detected).
    """

    if fault_frame._grid is None:
        raise ValueError("FaultFrame grid must be set.")
    spacing = getattr(fault_frame._grid, "spacing", None)
    if spacing is None:
        raise ValueError("Grid.spacing must be defined to compute band thickness.")

    # Determine band thickness in world units (meters)
    if thickness_world is None:
        thickness_world = float(voxels) * float(np.min(spacing))

    # Collect faults that have scalar fields and an isovalue
    faults = []
    for f in fault_frame._fault_elements:
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
    tol_scalar = []
    for nm, fld, _ in faults:
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
    bands = []
    for (nm, fld, level), ts in zip(faults, tol_scalar):
        band = np.abs(fld - level) <= ts
        bands.append((nm, band))

    # Pairwise overlap test
    for i in range(len(bands)):
        name_i, band_i = bands[i]
        if not band_i.any():
            continue
        for j in range(i + 1, len(bands)):
            name_j, band_j = bands[j]
            if not band_j.any():
                continue
            if np.any(band_i & band_j):
                raise ValueError(f"❌ Fault '{name_i}' crosscuts fault '{name_j}' (isovalue-band overlap).")


def interpolate_group_universal_cokriging_for_faults(
        element: FaultElement,
        grid,
        fault_surface_points_df: pd.DataFrame,
        fault_orientations_points_df: Optional[pd.DataFrame] = None,
) -> None:
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
        extent=grid.extent,
        resolution=grid.resolution,
        structural_frame=gempy_structural_frame,
    )

    mapping = {"fault_group": [element.name]}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)

    gp.compute_model(geo_model)

    # Set scalar value at surface and field
    element.set_scalar_value(
        float(geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0][0])
    )

    # NOTE: transpose if that’s how your marching/plotting expects it
    sf = geo_model.solutions.raw_arrays.scalar_field_matrix[0].reshape(tuple(grid.resolution))
    element.set_scalar_field(sf)

    # Domain mask convention (positive side = True)
    element.set_domain_mask(element.scalar_field > element.scalar_value)


def compute_fault_domains(
        fault_frame: FaultFrame,
) -> None:
    """
    Interpolates all faults and generates a domain map across the model grid.
    Relies on fault.domain_mask being set by the interpolator_func.
    """
    if not fault_frame._grid:
        raise ValueError("Grid must be set before domain generation.")
    if fault_frame._fault_surface_points_df is None:
        raise ValueError("Fault surface points must be set.")

    # Initialize single-domain model
    domain_map = np.zeros((fault_frame._grid.resolution[2],  # Z
                           fault_frame._grid.resolution[1],  # Y
                           fault_frame._grid.resolution[0]),  # X
                          dtype=int)

    domain_id_counter = 1

    temp_ids = []  # Track temporary domain IDs before remapping

    # Interpolate faults from youngest to oldest
    for i, fault in enumerate(reversed(fault_frame._fault_elements)):  # Youngest first
        name = fault.name

        # Extract surface point/orientation input_data for this fault
        points = fault_frame.get_surface_points_for_element(name)
        orientations = fault_frame.get_orientations_for_element(name)

        if points.empty:
            raise ValueError(f"❌ No surface points found for fault '{name}'.")

        # Run interpolation (sets scalar field, scalar value, mask internally)
        interpolate_group_universal_cokriging_for_faults(fault_frame.get_element_by_name(name),
                                                              fault_frame._grid,
                                                              fault_surface_points_df=points,
                                                              fault_orientations_points_df=orientations)

        if fault.get_domain_mask() is None:
            raise ValueError(f"❌ Interpolator did not set domain_mask for fault '{name}'.")

        fault_mask = fault.get_domain_mask()

        # Security to ensure matching resolution
        if fault_mask.shape != domain_map.shape:
            # assume fault mask is (z, y, x) and grid is (x, y, z)
            fault_mask = fault_mask.transpose(2, 1, 0)

        new_domain_map = domain_map.copy()

        # For each existing domain, split it if affected by this fault
        for existing_id in np.unique(domain_map):
            current_mask = domain_map == existing_id
            overlap = current_mask & fault_mask

            if np.any(overlap):
                new_id = 9999 + domain_id_counter

                # Assign a temporary large ID
                new_domain_map[overlap] = new_id
                temp_ids.append(9999 + domain_id_counter)
                domain_id_counter += 1

        domain_map = new_domain_map

    # Remap domain IDs to consecutive values starting from 0
    unique_ids = np.unique(domain_map)
    remap = {old: new for new, old in enumerate(unique_ids)}
    remapped_map = np.vectorize(remap.get)(domain_map)
    fault_frame._domain_map = remapped_map.T

    # remapped_map is currently ZYX in your code; convert to XYZ once
    domain_map_xyz = remapped_map.transpose(2, 1, 0)

    # --- ALSO store separated_domains (global side-sets), for reporting/sanity ---
    for fault in fault_frame.fault_elements:
        mask = fault.get_domain_mask()
        # ensure mask is XYZ
        if mask.shape != domain_map_xyz.shape:
            mask = mask.transpose(2, 1, 0)

        left_ids = set(np.unique(domain_map_xyz[mask]))
        right_ids = set(np.unique(domain_map_xyz[~mask]))

        if not left_ids or not right_ids:
            raise RuntimeError(f"Fault '{fault.name}' does not create a valid split (empty side).")

        fault.set_separated_domains((frozenset(left_ids), frozenset(right_ids)))

    for fault in fault_frame.fault_elements:
        # Ensure scalar_field is XYZ (you said you standardized to XYZ later;
        # here in this function it looks like it's already in the "native" orientation you use)
        sf = fault.scalar_field
        if sf.shape != domain_map_xyz.shape:
            # if sf is ZYX, convert to XYZ
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
            raise RuntimeError(f"Fault '{fault.name}' produced no domain_pairs (check band tolerance / orientation).")

        fault.set_domain_pairs(pairs)  # <-- you'll add this setter on the FaultElement

    #  Store per-domain masks
    fault_frame._domain_masks = {}
    for uid in np.unique(remapped_map):
        fault_frame._domain_masks[uid] = remapped_map == uid

    # Extract surfaces meshes for faults
    for i, fault in enumerate(reversed(fault_frame.fault_elements)):
        vertices, edges = marching_cubes_new(fault.scalar_field,
                                             [fault.scalar_value],
                                             fault_frame._grid.spacing,
                                             fault_frame._grid.extent)

        fault.set_mesh("unmasked", vertices[0], edges[0])

        # fault.set_vertices(vertices[0])
        # fault.set_edges(edges[0])

    # 🔎 After all faults are processed
    check_fault_crosscuts_via_isovalue_bands(fault_frame)


def build_fault_frame(
    fault_surface_points_df: pd.DataFrame,
    fault_orientations_df: pd.DataFrame,
    fault_names: list,  # youngest to oldest
    grid: RegularGrid,
) -> FaultFrame:
    """Build a :class:`FaultFrame` from ordered fault names, surface input_data, and a grid.

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

    Returns
    -------
    FaultFrame
        A fully configured fault frame with elements, colors, input input_data, and grid.
    """
    # Assign default gray colors if none provided
    colors = ["#555555"] * len(fault_names)

    fault_elements = []

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
    spacing_xyz: np.ndarray,
    *,
    voxels: float = 1.0,
    use_gradient: bool = True,
) -> frozenset[tuple[int, int]]:
    """
    Compute adjacent domain-id pairs across a fault by sampling a thin band around the fault surface.
    All arrays must be XYZ.
    Returns pairs as sorted (a,b) with a<b.
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

    # two sides near the surface
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
