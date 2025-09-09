import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from pykrige.ok3d import OrdinaryKriging3D


def interpolate_group_ordinary_kriging(
    *,
    group,  # StructuralGroup
    grid,   # RegularGrid with grid.gridx, grid.gridy, grid.gridz
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Ordinary Kriging for a single structural group (pure function).

    Returns:
        scalar_field: np.ndarray with shape == grid.resolution
        scalar_values_by_element: Dict[str, float]  (element_name -> scalar value)

    Notes:
        - Assigns strictly increasing scalar values: oldest = 1, youngest = n
        - Uses params from group.get_interpolation_params() (OrdinaryKrigingParams)
        - Does NOT mutate `group`
    """
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # 1) scalar values oldest->youngest = 1..n (your groups are youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }

    # Guard: all formations in DF must belong to this group
    unknown = set(group_surface_points_df["formation"].unique()) - set(scalar_values_by_element.keys())
    if unknown:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown)}"
        )

    # 2) map formations -> scalar values
    vals = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )

    # 3) coordinates
    x = group_surface_points_df["X"].to_numpy()
    y = group_surface_points_df["Y"].to_numpy()
    z = group_surface_points_df["Z"].to_numpy()

    # 4) parameters
    params = group.get_interpolation_params()  # OrdinaryKrigingParams

    # 5) fit & evaluate on grid
    ok3d = OrdinaryKriging3D(
        x, y, z,
        vals,
        variogram_model=params.variogram_model,
        variogram_parameters=[params.sill, params.range, params.nugget],
        anisotropy_scaling_z=params.anisotropy_scaling_z,
    )

    scalar_field, _ = ok3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        n_closest_points=params.neighbors
    )

    return scalar_field, scalar_values_by_element
