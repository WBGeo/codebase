
import pandas as pd
import numpy as np
from typing import Optional
from scipy.interpolate import RBFInterpolator
from core.structural_objects.objects import StructuralGroup, StructuralElement, StructuralFrame


#%%

def interpolate_group_radial_basis_function(
        group: StructuralGroup,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        grid
) -> None:

    # 1. Assign strictly increasing scalar values: oldest = 1, youngest = n
    for i, elem in enumerate(reversed(group.structural_elements), start=1):
        elem.set_scalar_value(float(i))

    if group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group {group.name}")

    # 2. Map formation (element name) to scalar_value
    formation_to_scalar = {elem.name: elem.scalar_value for elem in group.structural_elements}
    scalar_values = group_surface_points_df['formation'].map(formation_to_scalar).values.astype(float)

    # 3. Extract coordinates
    x = group_surface_points_df['X'].values
    y = group_surface_points_df['Y'].values
    z = group_surface_points_df['Z'].values

    # Get interpolation parameters from group
    params = group.get_interpolation_params()

    # Create RBF interpolator
    rbfi = RBFInterpolator(np.stack((x,y,z), axis=1),
                           scalar_values, kernel=params.kernel,
                           smoothing=params.smoothing,
                           neighbors=params.neighbors,  # Use None to use all points
                           epsilon=params.epsilon)

    # Interpolate the function on the grid
    rbf_res = rbfi(grid.grid_coordinates)

    # Reshape the result to resolution
    rbf_res = rbf_res.reshape(grid.resolution)

    # Store original scalar fields
    group.set_scalar_field(rbf_res.T)