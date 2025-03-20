# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.universal_cokriging_cleaned import universal_cokriging_interpolator_with_cleaning_and_alternative_meshes
from core.visualization_components import plot_2d, plot_3d

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 10: 2 faults, 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model 10',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([50, 50, 50]),
                      mapping_object={
                          "Fault_Series2": ('fault2'),
                          "Strat_Series2": ('rock3'),
                          "Fault_Series1": ('fault1'),
                          "Strat_Series1": ('rock2', 'rock1')},
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model10_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model10_orientations_df.csv"),
                      faults=[True, False, True, False]
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
results_test = universal_cokriging_interpolator_with_cleaning_and_alternative_meshes(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

# Calculate gradients at the surface mesh vertices
from core.utility import surface_mesh_gradients

points_list, vectors_list = surface_mesh_gradients.get_surface_mesh_gradients(results_test)


#%%

# Plotting the gradient vector field

import pyvista as pv

colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
        '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
        '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

for i in range(len(points_list)):
    pdata = pv.PolyData(points_list[i])
    pdata["vectors"] = vectors_list[i]  # Add vector field

    # Create arrow glyphs
    arrows = pdata.glyph(orient="vectors", scale="vectors", factor=50)

    # Plot the arrows
    plotter.add_mesh(arrows, color=colors[i])
    plotter.add_mesh(
                    pv.PolyData(results_test.surface_meshes_vertices[i],
                                np.insert(results_test.surface_meshes_edges[i], 0, 3, axis=1).ravel()),
                    color=colors[i])
plotter.show()

#%%

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

unit=1

pdata = pv.PolyData(points_list[unit])
pdata["vectors"] = vectors_list[unit]  # Add vector field

# Create arrow glyphs
arrows = pdata.glyph(orient="vectors", scale="vectors", factor=50)

# Plot the arrows
plotter.add_mesh(arrows, color=colors[unit])
plotter.add_mesh(
                pv.PolyData(results_test.surface_meshes_vertices[unit],
                            np.insert(results_test.surface_meshes_edges[unit], 0, 3, axis=1).ravel()),
                color=colors[unit], style="wireframe")
plotter.show()

#%%

# Count the number of NaN values
nan_count = np.sum(np.isnan(vectors_list[0]))

print(f"Number of NaN values: {nan_count}")
