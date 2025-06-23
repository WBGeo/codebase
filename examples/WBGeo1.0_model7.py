# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 7: 1 fault, 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model_7_UCK',
                      extent=np.array([0, 2500, 0, 1000, 0, 1000]),
                      resolution=np.array([125, 50, 50]),
                      mapping_object={
                          "Fault_Series": ('fault'),
                          "Strat_Series1": ('rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model7_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model7_orientations_df.csv"),
                      faults=[True, False, False]
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = universal_cokriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True,
        direction="y", slice_int=12)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="masked")

#%%

# Calculate gradients at the surface mesh vertices
from core.utility import surface_mesh_gradients

points_list, vectors_list = surface_mesh_gradients.get_surface_mesh_gradients(results_test, mesh_type="unmasked")

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
                    pv.PolyData(results_test.surface_meshes_vertices[1][i],
                                np.insert(results_test.surface_meshes_edges[1][i], 0, 3, axis=1).ravel()),
                    color=colors[i], style="surface")
plotter.show()

#%%

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

unit=2

pdata = pv.PolyData(points_list[unit])
pdata["vectors"] = vectors_list[unit]  # Add vector field

# Create arrow glyphs
arrows = pdata.glyph(orient="vectors", scale="vectors", factor=50)

# Plot the arrows
plotter.add_mesh(arrows, color=colors[unit])
plotter.add_mesh(
                pv.PolyData(results_test.surface_meshes_vertices[1][unit],
                            np.insert(results_test.surface_meshes_edges[1][unit], 0, 3, axis=1).ravel()),
                color=colors[unit], style="wireframe")
plotter.show()

# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100,100,100,100,100,500, 300,100,500,300,100,300), (500,500,500,500,500,900)],
    num_sources=2,
    sources=[(100,300,500), (400,600,700)],
    num_shafts=0,
    centers=[],
    axes=[],
    radii=[],
    num_planes=0,
    extra_planes=[],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=2,
    DISTANCE_THRESHOLD = 80,
    PROJECTION_THRESHOLD = 80,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)


mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")


