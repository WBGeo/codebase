
import pandas as pd
from typing import Optional
from pykrige.ok3d import OrdinaryKriging3D
from core.structural_objects.objects import StructuralGroup

def interpolate_group_ordinary_kriging(
        group: StructuralGroup,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        grid,
        variogram_model: str,
        variogram_parameters: list,
        anisotropy_scaling_z: float,
        neighbors: Optional[int] = None,
) -> None:
    """
    Perform Ordinary Kriging interpolation for a single structural group.

    Args:
        group: StructuralGroup instance to interpolate.
        group_surface_points_df: DataFrame with columns ['X', 'Y', 'Z', 'formation'] filtered for this group.
        grid: Grid object containing gridx, gridy, gridz arrays for interpolation.
        variogram_model: Variogram model name, e.g. 'spherical'.
        variogram_parameters: List of variogram parameters [sill, range, nugget].
        anisotropy_scaling_z: Scaling factor for z-direction anisotropy.
        neighbors: Number of closest points to use in kriging.
    """
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

    # 4. Perform Ordinary Kriging
    ok3d = OrdinaryKriging3D(
        x, y, z,
        scalar_values,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )

    k3d1, ss3d = ok3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        n_closest_points=neighbors
    )

    # 5. Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    group.set_scalar_field(k3d1)