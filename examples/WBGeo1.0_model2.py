# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.geo_inr import geo_inr_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.loopstructural import loop_structural_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.mesh_generation.mesh_data import create_mesh_data

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 2: no faults, no unconformities, 2 stratigraphic series

# Component 1: input data
data_test = InputData(name='Model 2',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([20, 20, 20]),
                      mapping_object={"Strat_Series": ('rock2', 'rock1')},
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model2_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model2_orientations_df.csv"),
                      )

#%%

# Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
results_test = ordinary_kriging_interpolator(data_test)
# results_test = geo_inr_interpolator(data_test)
# results_test = loop_structural_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

# 4: Meshing for Process Simulation
mesh_test = create_mesh_data(
    geomodel_result=results_test,
    refinement_data=[10,10,10],
    z_threshold=0.1,
    tolerance=1
)

#%%

# 4.5: Plot the meshing result (only 3D at current state)
plot_mesh_3d(mesh_test, data_test)







