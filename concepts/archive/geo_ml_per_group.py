
import pandas as pd
from typing import Optional
from pykrige.ok3d import OrdinaryKriging3D
from core.structural_objects.objects import StructuralGroup

import geoml

import geoml.kernels as kr
import geoml.transform as tr
import geoml.latent as gl
# import geoml.likelihood as lk

#%%

def interpolate_group_geoml(
        group: StructuralGroup,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        grid,
) -> None:
    """
    Perform Ordinary Kriging interpolation for a single structural group.

    Args:
        group: StructuralGroup instance to interpolate.
        group_surface_points_df: DataFrame with columns ['X', 'Y', 'Z', 'formation'] filtered for this group.
        grid: Grid object containing gridx, gridy, gridz arrays for interpolation.
    """
    # TODO: transfer to GeoML

    # Create a 3D grid for interpolation in GeoML format
    x3d = geoml.data.Grid3D(start=[grid.extent[0], grid.extent[2], grid.extent[4]],
                            n=grid.resolution,
                            end=[grid.extent[1], grid.extent[3], grid.extent[5]])

    # 1. Assign strictly increasing scalar values: oldest = 1, youngest = n
    for i, elem in enumerate(reversed(group.structural_elements), start=1):
        elem.set_scalar_value(float(i))

    # 2. Map formation (element name) to scalar_value
    formation_to_scalar = {elem.name: elem.scalar_value for elem in group.structural_elements}
    print(formation_to_scalar)
    print(len(group.structural_elements))

    # create a new column in the DataFrame for scalar values
    group_surface_points_df['scalar_value'] = group_surface_points_df['formation'].map(formation_to_scalar).astype(float)

    # How to actually get the input data in the right format
    input_data = geoml.data.PointData(group_surface_points_df, ["X", "Y", "Z"])

    input_data.add_continuous_variable("scalar_value",
                                        measurements=group_surface_points_df['scalar_value'].values,
                                        probabilities=(0.25,0.5, 0.75))


    # TODO: Train the thingy

    net_input = gl.BasicInput(
        inducing_points=input_data)#,
        #transform=tr.Anisotropy3D(5000, 1, 0.01))


    net_output = gl.BasicGP(net_input, size=len(group.structural_elements), kernel=kr.Gaussian())

    model = geoml.models.VGPNetwork(
            data=input_data,
            variables=["scalar_value"],
            likelihoods=geoml.likelihood.CategoricalGaussianIndicator(
                n_components=len(group.structural_elements), # Number of elements in the group (len(group.structural_elements))
                sharpness=5  # to boost accuracy in training data
            ),
            latent_network=net_output,
            options=geoml.models.GPOptions(jitter=1e-6))

    model.train_full(max_iter=500)

    # # TODO: Predict
    # All arguments represent the [x, y, z] directions:
    model.predict(x3d)

    # TODO: Where is stuff stored?

    scalar_field = x3d.variables['scalar_value'].quantiles[0.5].values # mean scalar field
    print(scalar_field.shape)



    # group.set_scalar_field(????) # Does this need to be transformed properly

    # 1. Assign strictly increasing scalar values: oldest = 1, youngest = n
    # for i, elem in enumerate(reversed(group.structural_elements), start=1):
    #     elem.set_scalar_value(float(i))



    # TODO: This is show Italo plots the meshes
    # The boundaries are defined by the 0 level of each indicator in the model output.
    contours = {}
    for form in form_vals.values():
        contours[form] = ara_grid.variables["Formation"].components[form] \
            .indicator_predicted.get_contour(0.001).as_pyvista()

    pl = pv.Plotter(notebook=True, window_size=[1800, 1500])
    pl.add_mesh(holes_pv.tube(radius=100), cmap=cm.roma,
                scalars='Formation',  # annotations=form_vals
                )
    for form in form_vals.values():
        pl.add_mesh(contours[form], color='gray')
    pl.set_scale(zscale=vertical_exaggeration)
    pl.show_grid()
    pl.show(jupyter_backend="static", return_viewer=True)











    # TODO: Old OK code

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
    params: OrdinaryKrigingParams = group.get_interpolation_params()

    # 4. Perform Ordinary Kriging
    ok3d = OrdinaryKriging3D(
        x, y, z,
        scalar_values,
        variogram_model=params.variogram_model,
        variogram_parameters=[params.sill, params.range, params.nugget],
        anisotropy_scaling_z=params.anisotropy_scaling_z
    )

    k3d1, ss3d = ok3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        n_closest_points=params.neighbors
    )

    # 5. Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    group.set_scalar_field(k3d1)