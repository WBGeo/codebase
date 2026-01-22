
import numpy as np
from concepts.archive.objects import StructuralGroup, IDWParams
from scipy.spatial import cKDTree
import pandas as pd


def interpolate_group_idw(
    group: StructuralGroup,
    group_surface_points_df: pd.DataFrame,
    grid
) -> None:
    """
    Perform IDW interpolation for a single structural group with optional anisotropy.

    Args:
        group: StructuralGroup instance to interpolate.
        group_surface_points_df: DataFrame with ['X', 'Y', 'Z', 'formation'] filtered for this group.
        grid: Grid object with gridx, gridy, gridz.
    """
    # Assign scalar values
    for i, elem in enumerate(reversed(group.structural_elements), start=1):
        elem.set_scalar_value(float(i))


    if group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group {group.name}")

    # TODO: This is a very simplified way to handle buffer points.
    # Add buffer points
    buffer_scalars = np.array([0,len(group.structural_elements)+1])
    # buffer coords based on min and max of the group surface points
    buffer_coords = np.array([group_surface_points_df[['X', 'Y', 'Z']].min().values - 100,
                              group_surface_points_df[['X', 'Y', 'Z']].min().values + 100])


    formation_to_scalar = {elem.name: elem.scalar_value for elem in group.structural_elements}
    scalar_values = group_surface_points_df['formation'].map(formation_to_scalar).values.astype(float)

    # Extract and scale coordinates
    coords = group_surface_points_df[["X", "Y", "Z"]].values

    # TODO: This buffering does not really work as intended.
    coords = np.vstack([coords, buffer_coords])  # Add buffer points
    scalar_values = np.hstack([scalar_values, buffer_scalars])  # Add buffer scalars

    params: IDWParams = group.get_interpolation_params()
    scale_x, scale_y, scale_z = params.anisotropy_scaling
    scaled_coords = coords / np.array([scale_x, scale_y, scale_z])

    # Prepare grid points and scale them
    gx, gy, gz = grid.gridx, grid.gridy, grid.gridz
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_points = np.vstack([Gx.ravel(), Gy.ravel(), Gz.ravel()]).T
    scaled_grid_points = grid_points / np.array([scale_x, scale_y, scale_z])

    # set neighbors to all if None
    if params.neighbors is None:
        n_neighbors = len(scaled_coords)
    else:
        n_neighbors = params.neighbors

    # Build KDTree with scaled input
    tree = cKDTree(scaled_coords)
    distances, indices = tree.query(scaled_grid_points, k=n_neighbors)

    distances = np.where(distances == 0, 1e-12, distances)  # avoid div by 0
    weights = 1 / distances**params.power
    weights /= weights.sum(axis=1)[:, np.newaxis]

    interpolated = np.sum(weights * scalar_values[indices], axis=1)
    scalar_field = interpolated.reshape(Gx.shape)

    print(scalar_field.shape)
    print(scalar_field.min(), scalar_field.max())

    # plot section of scalar field
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(scalar_field[:, scalar_field.shape[1] // 2, :].T, cmap='viridis', origin='lower')
    plt.colorbar(label='Scalar Value')
    plt.title(f'Scalar Field for Group: {group.name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


    group.set_scalar_field(scalar_field)
