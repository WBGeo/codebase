
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.interpolate import RBFInterpolator
import warnings

def interpolate_group_radial_basis_function(
    *,
    group,                       # StructuralGroup
    grid,                        # RegularGrid with grid.resolution and either grid.grid_coordinates or (grid.gridx, grid.gridy, grid.gridz)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Radial Basis Function interpolation for a single structural group (pure).

    Returns:
        scalar_field : np.ndarray, shape == grid.resolution (nx, ny, nz)
        scalar_values_by_element : Dict[str, float] (element_name -> scalar value)

    Notes:
        - Assigns strictly increasing scalar values: oldest = 1, youngest = n
        - Uses params from group.get_interpolation_params() (your RBFParams)
        - Does NOT mutate `group`
    """
    # Check if any surface points are provided
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    # Check if orientations are provided and war that they will not be used
    if group_orientations_points_df is not None and not group_orientations_points_df.empty:
        warnings.warn(f"Orientations provided for group '{group.name}' will not be used in RBF interpolation", UserWarning)

    # Check if there are at least two distinct elements in the group
    if len(group.structural_elements) < 2:
        raise ValueError(f"Group '{group.name}' must contain at least two structural elements for RBF interpolation")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # 1) scalar values oldest->youngest = 1..n (your groups list is youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }

    # 2) map formations -> scalar values
    vals = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )

    # 3) coordinates of control points
    coords = group_surface_points_df[["X", "Y", "Z"]].to_numpy()

    # 4) parameters
    params = group.get_interpolation_params()  # should be your RBFParams instance

    # 5) fit RBF
    rbfi = RBFInterpolator(
        coords,
        vals,
        kernel=params.kernel,
        smoothing=params.smoothing,
        epsilon=params.epsilon,
        neighbors=params.neighbors,  # None => all points
    )

    # 6) evaluate on grid
    # Prefer a precomputed (N,3) array if your grid exposes it; otherwise build from axes.
    if hasattr(grid, "grid_coordinates") and grid.grid_coordinates is not None:
        grid_points = grid.grid_coordinates
    else:
        gx, gy, gz = np.meshgrid(grid.gridx, grid.gridy, grid.gridz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    scalar_flat = rbfi(grid_points)
    scalar_field = scalar_flat.reshape(tuple(grid.resolution)).T  # (nx, ny, nz)

    return scalar_field, scalar_values_by_element
