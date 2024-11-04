# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components import export_mesh_moose

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model Weisweiler Mini: 1 stratigraphic series

# Component 1: Input data
data_test = InputData(name='WeisweilerMini',
                      extent=np.array([5623500, 5640000, 32304500, 32305500, -3000, 500]),
                      resolution=np.array([165, 50, 70]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/modelWeisweilerMini_surface_points_df2.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/modelWeisweilerMini_orientations_df2.csv"),
                      mapping_object={
                          "Strat_Series1": ('BreitgangFM','KrebsTraufeFM', 'WilhelmineFM',
                                'ObererKohlenkalkGP','MittlererKohlenkalkGP', 'CondrozGP')},
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
# results_test = ordinary_kriging_interpolator(data_test, var_range=10000)
results_test = rbf_interpolator(data_test, kernel='cubic', epsilon=0.00000001)


#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

#%%

# 4: Export the results to MOOSE
export_mesh_moose.export_data_to_moose(results_test)

#%%

import pyvista as pv
from skimage import measure

#%%

np.unique(results_test.lith_block)


#%%

# Approach to directly do MC on the scalar field values, not the lith IDs
# Does not improve the result as far as I can see
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd']

dx = (data_test.extent[1] - data_test.extent[0]) / data_test.resolution[0]
dy = (data_test.extent[3] - data_test.extent[2]) / data_test.resolution[1]
dz = (data_test.extent[5] - data_test.extent[4]) / data_test.resolution[2]

levels = np.unique(results_test.lith_block)-0.01
levels = levels[1:-1]
print(levels)
formations = data_test.surface_points['formation'].unique()

# Extract the surface from the block
mc_vertices = []
mc_edges = []
for i in levels:
    verts, faces, _, _ = measure.marching_cubes(results_test.lith_block.reshape(results_test.resolution), i,
                                                spacing=(dx,dy,dz))
    mc_vertices.append(verts)
    mc_edges.append(faces)

#%%

# Create a PyVista plotter
plotter = pv.Plotter()

for i in range(len(formations)):
    plotter.add_mesh(
        pv.PolyData(mc_vertices[i],
                    np.insert(mc_edges[i], 0, 3, axis=1).ravel()),
    color=colors[i])

plotter.show()

