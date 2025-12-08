# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from core.object_components import InputData

from core.interpolator_components.interpolators_per_group import general

from core.visualization_components_new import visualize_structural_frame, plot_structural_slice
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series
# Component 1: Input data
data_test = InputData(name='Model_12',
                      extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                      resolution=np.array([100, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model12_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model12_orientations_df.csv"),
                      mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                      )

#%%

frame = general.build_structural_frame(data_test.mapping_object,
                                        data_test.extent,
                                        data_test.resolution,
                                        data_test.surface_points,
                                        data_test.orientations)
frame.summary()

#%%

# Plot a slice of the structural model
plot_structural_slice(frame, axis='y', index=10, show_scalar_contours=True)

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

frame["Strat_Series1"].set_interpolation_method("Loop Structural")
frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

frame.detailed_report()

#%%

frame, block = general.combined_interpolator(frame)

#%%

# Plot a slice of the structural model
plot_structural_slice(frame, lith_block=block, axis='y', index=0, show_scalar_contours=True)


#%%

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_surface_meshes=True, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

# plot section of scalar field for group "Strat_Series1"
import matplotlib.pyplot as plt
group = frame["Strat_Series1"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series1"]["rock3"].scalar_value, frame["Strat_Series1"]["rock4"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

group = frame["Strat_Series2"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series2"]["rock1"].scalar_value, frame["Strat_Series2"]["rock2"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')


plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    wells=[(100,100,980,100,100,600)],
    sources=[(300,100,900)],
    centers=[(100,100,100)],
    axes=[(2000,0,0)],
    radii=[30],
    extra_planes=[],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=10,
    DISTANCE_THRESHOLD = 50,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 120,
    z_threshold = 10
)

## (commented out to avoid file creation while running the example)
# mesh_test.export_vtm('file.vtm')
# mesh_ex=mesh_test.export_exodus("filename.exo")
# mesh_vtu=mesh_test.export_vtu("filename.vtu")

