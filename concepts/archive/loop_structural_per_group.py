
import pandas as pd
import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from core.structural_objects.objects import StructuralGroup
from LoopStructural import GeologicalModel

def interpolate_group_loop_structural(
        group: StructuralGroup,# Only points relevant to this group
        grid,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        group_orientations_points_df=pd.DataFrame,  # Only orientations relevant to this group
) -> None:

    if group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group {group.name}")

    if group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group {group.name}")

    # 1. Assign strictly increasing scalar values: oldest = 1, youngest = n
    for i, elem in enumerate(reversed(group.structural_elements), start=1):
        elem.set_scalar_value(float(i))

    # Convert surface_points to loopstructural input DataFrame
    surface_points_temp = group_surface_points_df.copy()
    orientations_temp = group_orientations_points_df.copy()

    # Map from formation name to scalar value
    formation_to_scalar = {
        element.name: element.scalar_value
        for element in group.structural_elements
    }

    # Add 'feature_name' and 'val' columns
    surface_points_temp["feature_name"] = group.name
    surface_points_temp["val"] = surface_points_temp["formation"].map(formation_to_scalar)

    # Reorder columns if necessary
    surface_points_temp = surface_points_temp[['X', 'Y', 'Z', 'val', 'feature_name']]
    surface_points_temp['gx'] = np.nan
    surface_points_temp['gy'] = np.nan
    surface_points_temp['gz'] = np.nan

    # TODO: Location of these orientations seems to matter (a lot) for a loopstructural model
    # Add 'feature_name' and 'val' columns
    orientations_temp["feature_name"] = group.name
    orientations_temp["val"] = orientations_temp["formation"].map(formation_to_scalar)

    orientations_temp = orientations_temp.rename(columns={'G_x': 'gx', 'G_y': 'gy', 'G_z': 'gz'})
    orientations_temp = orientations_temp[['X', 'Y', 'Z', 'val', 'feature_name', 'gx', 'gy', 'gz']]

    # Create final combined df for loopstructural
    data_combined = pd.concat([surface_points_temp, orientations_temp], ignore_index=True)

    # 4. Perform Loop Structural Interpolation

    # Create a GeologicalModel instance
    model = GeologicalModel(grid.extent[::2], grid.extent[1::2])
    model.set_model_data(data_combined)

    # Set stratigraphic column
    stratigraphic_column = {}
    stratigraphic_column[group.name] = {}
    for i, rock in enumerate(group.structural_elements):
        stratigraphic_column[group.name][rock.name] = {"min": i, "max": i + 1, "id": i}

    model.set_stratigraphic_column(stratigraphic_column)

    # features = [input_data.mapping_object.keys()]
    strat_features = []

    # Get the parameters from the group
    params = group.get_interpolation_params()

    strat = model.create_and_add_foliation(
            group.name,
            interpolatortype=params.interpolator_type,  # try changing this to 'PLI'
            nelements=1e4,  # try changing between 1e3 and 5e4
            buffer=0,
            solver="cg",
            # npw=1,
            # gpw=100000,
            # regularisation=1,
            damp=True,
        )

    strat_features.append(strat)

    # Set grid # TODO: Check if this is necessary
    regular_grid = grid.grid_coordinates

    sf = model.evaluate_feature_value(group.name, regular_grid, scale=True)
    sf = sf.reshape(grid.resolution).T

    # 5. Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    group.set_scalar_field(sf)









