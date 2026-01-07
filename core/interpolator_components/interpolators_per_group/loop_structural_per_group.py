import numpy as np
import pandas as pd
from typing import Dict, Tuple
from LoopStructural import GeologicalModel

def interpolate_group_loop_structural(
    *,
    group,                        # StructuralGroup
    grid,                         # RegularGrid (has extent, resolution, grid_coordinates)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    LoopStructural interpolation for a single structural group (pure function).

    Returns:
        scalar_field : np.ndarray shaped to tuple(grid.resolution)
        scalar_values_by_element : Dict[str, float] (element_name -> scalar value)
    """
    # --- validation ---
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

    # Ensure all formations in the DFs belong to this group
    group_elem_names = [e.name for e in group.structural_elements]
    unknown_sp = set(group_surface_points_df["formation"].unique()) - set(group_elem_names)
    unknown_ori = set(group_orientations_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_sp:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown_sp)}"
        )
    if unknown_ori:
        raise ValueError(
            f"Orientations for group '{group.name}' contain formations not in the group: {sorted(unknown_ori)}"
        )

    # --- per-element scalar values (oldest=1 ... youngest=n) ---
    names_old_to_young = [e.name for e in reversed(group.structural_elements)]
    scalar_values_by_element: Dict[str, float] = {
        name: float(i) for i, name in enumerate(names_old_to_young, start=1)
    }

    # Map formation name -> scalar ('val') for LoopStructural
    formation_to_scalar = scalar_values_by_element

    # --- build LoopStructural input tables ---
    # Surface points table
    surface_points = group_surface_points_df.copy()
    surface_points["feature_name"] = group.name
    surface_points["val"] = surface_points["formation"].map(formation_to_scalar)
    # LS expects columns: X, Y, Z, val, feature_name, gx, gy, gz
    surface_points = surface_points[["X", "Y", "Z", "val", "feature_name"]]
    surface_points["gx"] = np.nan
    surface_points["gy"] = np.nan
    surface_points["gz"] = np.nan

    # Orientations table
    orientations = group_orientations_points_df.copy()
    orientations["feature_name"] = group.name
    orientations["val"] = orientations["formation"].map(formation_to_scalar)
    orientations = orientations.rename(columns={"G_x": "gx", "G_y": "gy", "G_z": "gz"})
    orientations = orientations[["X", "Y", "Z", "val", "feature_name", "gx", "gy", "gz"]]

    # Combined table
    data_combined = pd.concat([surface_points, orientations], ignore_index=True)

    # --- build LoopStructural model ---
    # GeologicalModel(min_bounds, max_bounds)
    model = GeologicalModel(grid.extent[::2], grid.extent[1::2])
    model.set_model_data(data_combined)

    # Stratigraphic column: keep your original order (group.structural_elements)
    stratigraphic_column = {group.name: {}}
    for i, rock in enumerate(group.structural_elements):
        stratigraphic_column[group.name][rock.name] = {"min": i, "max": i + 1, "id": i}
    model.set_stratigraphic_column(stratigraphic_column)

    # Interpolator params
    params = group.get_interpolation_params()  # LoopStructuralParams (expects .interpolator_type)

    # Create the foliation / feature
    _ = model.create_and_add_foliation(
        group.name,
        interpolatortype=params.interpolator_type,  # "FDI" or "PLI"
        nelements=1e4,
        buffer=0,
        solver="cg",
        damp=True,
    )

    # Evaluate on grid
    regular_grid = grid.grid_coordinates  # (N, 3)
    sf_flat = model.evaluate_feature_value(group.name, regular_grid, scale=True)
    # Match your previous implementation: reshape then transpose
    scalar_field = np.asarray(sf_flat).reshape(tuple(grid.resolution)).T

    return scalar_field, scalar_values_by_element
