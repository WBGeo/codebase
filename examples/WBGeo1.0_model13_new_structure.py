# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.object_components import InputData
from core.grids.grid_classes import RegularGrid

from core.interpolator_components.interpolators_per_group import general

from core.visualization_components import plot_2d, plot_3d


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series
# Component 1: Input data
data_test = InputData(name='Model_13',
                      extent=np.array([0, 1000, 0, 500, 0, 1000]),
                      resolution=np.array([50, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model13_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model13_orientations_df.csv"),
                      mapping_object={
                          "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
                          "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
                          "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1')},
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

grid = RegularGrid(data_test.extent, data_test.resolution)

#%%

frame = general.build_structural_frame(data_test.mapping_object, data_test.surface_points,data_test.orientations)
frame.summary()

#%%

frame["Shallow_Strat"].set_interpolation_method("Loop Structural")
frame["Medium_Strat"].set_interpolation_method("Universal Co-Kriging")
frame["Deep_Strat"].set_interpolation_method("GeoINR")

# frame["Shallow_Strat"].set_interpolation_method("Ordinary Kriging")
# frame["Medium_Strat"].set_interpolation_method("Radial Basis Function")
# frame["Deep_Strat"].set_interpolation_method("Ordinary Kriging")

# frame.pretty_print()

frame.summary()

#%%

frame["Shallow_Strat"].configure_interpolation_params(interpolator_type='PLI',)
frame["Medium_Strat"].configure_interpolation_params()
frame["Deep_Strat"].configure_interpolation_params(beta=10)

#%%

frame.detailed_report()

#%%

# TODO: When executed twice throws error when meshing
frame, block = general.combined_interpolator(frame, grid)

#%%


import matplotlib.pyplot as plt
# plot slice of block
plt.imshow(block[:, 0, :], cmap='viridis', origin='lower')
plt.colorbar()
plt.title("Lithology Block Section")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

#%%

frame.pretty_print()

#%%

# plot surface meshes
import pyvista as pv

pv.global_theme.allow_empty_mesh = True

# Create a PyVista plotter
plotter = pv.Plotter(notebook=False)

# loop over all elements from all groups in frame
for group in frame.structural_groups:
    for element in group.structural_elements:
        plotter.add_mesh(pv.PolyData(element.vertices['masked'],
                            np.insert(element.edges["masked"], 0, 3, axis=1).ravel()),
                            color=element.color, label=element.name)

plotter.add_legend(size=(0.13, 0.13), loc='lower right', face='circle')

# Set the bounds and grid of the plotter
plotter.show_bounds(bounds=data_test.extent,
                        location="furthest",
                        grid=True)

# Set the camera position
plotter.camera.view_angle = 30.0
plotter.camera.azimuth = 25.0
plotter.camera.elevation = -15.0

plotter.show()

# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100,100,100,100,100,500, 300,100,500,300,100,300), (500,500,500,500,500,900)],
    num_sources=2,
    sources=[(100,300,500), (400,600,700)],
    num_shafts=2,
    centers=[(200,500,400), (100,200,700)],
    axes=[(1000,0,0), (1000,0,0)],
    radii=[30, 20],
    num_planes=2,
    extra_planes=[(0,0,400,1000,0,400,1000,1000,400,0,1000,400), (0,0,600,1000,0,600,1000,1000,600,0,1000,600)],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)


mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")


