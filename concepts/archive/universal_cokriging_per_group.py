import pandas as pd
import numpy as np
import gempy as gp
from concepts.archive.objects import StructuralGroup


def interpolate_group_universal_cokriging(
        group: StructuralGroup,
        grid,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        group_orientations_points_df=pd.DataFrame,  # Only orientations relevant to this group
) -> None:
    """
    Perform Ordinary Kriging interpolation for a single structural group.

    Args:
        group: StructuralGroup instance to interpolate.
        grid: Grid object containing gridx, gridy, gridz arrays for interpolation.
        group_surface_points_df: DataFrame with columns ['X', 'Y', 'Z', 'formation'] filtered for this group.
        group_orientations_points_df:
    """

    if group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group {group.name}")

    if group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group {group.name}")

    # 3. Convert to gempy
    surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
        x=group_surface_points_df.X.to_numpy(),
        y=group_surface_points_df.Y.to_numpy(),
        z=group_surface_points_df.Z.to_numpy(),
        names=group_surface_points_df.formation.to_numpy(),
        nugget=np.zeros(len(group_surface_points_df)),
        name_id_map=None)

    orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
        x=group_orientations_points_df.X.to_numpy(),
        y=group_orientations_points_df.Y.to_numpy(),
        z=group_orientations_points_df.Z.to_numpy(),
        G_x=group_orientations_points_df.G_x.to_numpy(),
        G_y=group_orientations_points_df.G_y.to_numpy(),
        G_z=group_orientations_points_df.G_z.to_numpy(),
        names=group_orientations_points_df.formation.to_numpy(),
        nugget=np.zeros(len(group_orientations_points_df)),
        name_id_map=surface_data.name_id_map)

    gempy_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(surface_data, orientation_data)

    # Create a GeoModel instance
    geo_model = gp.create_geomodel(
        project_name="random",
        extent=grid.extent,
        resolution=grid.resolution,
        structural_frame=gempy_structural_frame
    )

    mapping = {group.name: [element.name for element in group.structural_elements]}

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object=mapping
    )

    # TODO: This completely ignores faults

    # Get parameters for Universal CoKriging
    params = group.get_interpolation_params()

    # 4. Perform Universal CoKriging
    # Compute the geological model
    gp.compute_model(geo_model)

    for i, elem in enumerate(group.structural_elements):
        elem.set_scalar_value(geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0][i])

    print(geo_model.solutions.raw_arrays.scalar_field_matrix[0].shape)

    # 5. Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    group.set_scalar_field(geo_model.solutions.raw_arrays.scalar_field_matrix[0].reshape(grid.resolution).T)
