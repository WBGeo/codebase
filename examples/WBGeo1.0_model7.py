# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components import export_mesh_moose
from core.meshing_components.meshing_moose import create_mesh_moose

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
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

# 4: Meshing for Process Simulation
# export_mesh_moose.export_data_to_moose(results_test)

#%%

# 5: Create mesh
mesh_test = create_mesh_moose(results_test, name='Model_7_UCK')

#%%

# 5.5: Plot the mesh in 3D
plot_mesh_3d(mesh_test, data_test)

