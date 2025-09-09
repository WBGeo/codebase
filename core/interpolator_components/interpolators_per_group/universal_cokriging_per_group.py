import numpy as np
import pandas as pd
import gempy as gp
from typing import Dict, Tuple

def interpolate_group_universal_cokriging(
    *,
    group,                       # StructuralGroup
    grid,                        # RegularGrid (has extent, resolution)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Universal Co-Kriging (via GemPy) for a single structural group (pure function).

    Returns:
        scalar_field : np.ndarray with shape == tuple(grid.resolution)  (nx, ny, nz)
        scalar_values_by_element : Dict[str, float]  (element_name -> scalar value at surface points)

    Notes:
        - Does NOT mutate `group`.
        - Assumes `group.structural_elements` are ordered youngest -> oldest (as in your framework).
        - We build the GemPy StructuralFrame from the provided dataframes, map a single series
          (this group's name) to its surfaces (element names), compute the model, and then
          fetch the scalar field and per-surface scalar values.
    """
    # ---- validation ----
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")
    if group_orientations_points_df is None or group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group '{group.name}'")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")
    for col in ("X", "Y", "Z", "G_x", "G_y", "G_z", "formation"):
        if col not in group_orientations_points_df.columns:
            raise ValueError(f"Orientations for '{group.name}' missing column '{col}'")

    # Ensure the formations in the DFs belong to this group
    group_elem_names = [e.name for e in group.structural_elements]
    unknown_sp = set(group_surface_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_sp:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown_sp)}"
        )
    unknown_ori = set(group_orientations_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_ori:
        raise ValueError(
            f"Orientations for group '{group.name}' contain formations not in the group: {sorted(unknown_ori)}"
        )

    # ---- build GemPy data tables ----
    surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
        x=group_surface_points_df["X"].to_numpy(),
        y=group_surface_points_df["Y"].to_numpy(),
        z=group_surface_points_df["Z"].to_numpy(),
        names=group_surface_points_df["formation"].to_numpy(),
        nugget=np.zeros(len(group_surface_points_df)),
        name_id_map=None,
    )

    orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
        x=group_orientations_points_df["X"].to_numpy(),
        y=group_orientations_points_df["Y"].to_numpy(),
        z=group_orientations_points_df["Z"].to_numpy(),
        G_x=group_orientations_points_df["G_x"].to_numpy(),
        G_y=group_orientations_points_df["G_y"].to_numpy(),
        G_z=group_orientations_points_df["G_z"].to_numpy(),
        names=group_orientations_points_df["formation"].to_numpy(),
        nugget=np.zeros(len(group_orientations_points_df)),
        name_id_map=surface_data.name_id_map,
    )

    gp_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(
        surface_data, orientation_data
    )

    # ---- create model ----
    geo_model = gp.create_geomodel(
        project_name="uk_group",
        extent=np.asarray(grid.extent, dtype=float),
        resolution=np.asarray(grid.resolution, dtype=int),
        structural_frame=gp_structural_frame,
    )

    # Map one series (group) to its surfaces (elements)
    series_to_surfaces = {group.name: group_elem_names}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=series_to_surfaces)

    # ---- compute ----
    gp.compute_model(geo_model)

    # ---- fetch results ----
    # scalar values per surface (GemPy returns one per surface in the order of surfaces in the series)
    # Indexing [0] is the first series; then [i] for each surface in that series
    raw_scalar_vals = geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0]
    if len(raw_scalar_vals) != len(group_elem_names):
        raise RuntimeError(
            f"GemPy returned {len(raw_scalar_vals)} scalar values but group '{group.name}' has "
            f"{len(group_elem_names)} elements."
        )
    scalar_values_by_element = {
        name: float(val) for name, val in zip(group_elem_names, raw_scalar_vals)
    }

    # The scalar field (flattened) for this series is typically in scalar_field_matrix[0]
    raw_scalar_field = geo_model.solutions.raw_arrays.scalar_field_matrix[0]
    scalar_field = np.asarray(raw_scalar_field, dtype=float).reshape(tuple(grid.resolution)).T

    return scalar_field, scalar_values_by_element
