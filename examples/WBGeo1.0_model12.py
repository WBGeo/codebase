# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./codebase')

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components import export_mesh_moose
from core.meshing_components.meshing_moose import create_mesh_moose

#%%

cwd = os.getcwd() + "/codebase"

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model_12_RBF',
                      extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                      resolution=np.array([125, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model12_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model12_orientations_df.csv"),
                      mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
# results_test = ordinary_kriging_interpolator(data_test, var_range=500)
results_test = rbf_interpolator(data_test, kernel='cubic', epsilon=1)
# results_test, combined_scalar_field = ordinary_kriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

# 4: Meshing for Process Simulation
# export_mesh_moose.export_data_to_moose(results_test)

#%%

# 5: Create mesh
mesh_test = create_mesh_moose(results_test, name="Model_12_OK")

#%%

# 5.5: Plot the mesh in 3D
plot_mesh_3d(mesh_test, data_test)


#%%

# TODO: Alternative plotting like Elisa - probably requires combined scalar field
import matplotlib.pyplot as plt

levels = [0.99999, 1.99999, 2.99999, 3.99999]

fig, ax = plt.subplots()
# Reshape lithology block to resolution
plot_block = combined_scalar_field.reshape(results_test.resolution)
plot_block = plot_block

# cut slice in the middle of the model
image = plot_block[:, int(np.rint(results_test.resolution[1] / 2)), :].T

# Create a discrete color map for the lithology block

cs = ax.imshow(image, origin='lower', zorder=-100,
          extent=(float(results_test.extent[0]), float(results_test.extent[1]),
                  float(results_test.extent[4]), float(results_test.extent[5])))

ax.contour(image, 0, levels=levels, colors=["green", "yellow", "red", "black"], linestyles='solid', origin='lower',
          extent=(float(results_test.extent[0]), float(results_test.extent[1]),
                  float(results_test.extent[4]), float(results_test.extent[5])))

plt.colorbar(cs)

plt.show()

#%%

import pyvista as pv
from skimage import measure

#%%

# Approach to directly do MC on the scalar field values, not the lith IDs
# Does not improve the result as far as I can see
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd']

dx = (data_test.extent[1] - data_test.extent[0]) / data_test.resolution[0]
dy = (data_test.extent[3] - data_test.extent[2]) / data_test.resolution[1]
dz = (data_test.extent[5] - data_test.extent[4]) / data_test.resolution[2]

levels = [0.9999, 1.9999, 2.9999, 3.9999]
formations = data_test.surface_points['formation'].unique()

# Extract the surface from the block
mc_vertices = []
mc_edges = []
for i in levels:
    verts, faces, _, _ = measure.marching_cubes(combined_scalar_field, i,
                                                spacing=(dx,dy,dz))
    mc_vertices.append(verts)
    mc_edges.append(faces)

# Create a PyVista plotter
plotter = pv.Plotter()

for i in range(len(formations)):
    plotter.add_mesh(
        pv.PolyData(mc_vertices[i],
                    np.insert(mc_edges[i], 0, 3, axis=1).ravel()),
    color=colors[i])

plotter.show()