# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData, GeomodelResults
from core.interpolator_components import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 2: no faults, no unconformities, 2 stratigraphic series

# Component 1: input data
data_test = InputData(name='Model 2',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([20, 20, 20]),
                      surface_points=pd.read_csv(
                          cwd+"/examples/data/model2_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model2_orientations_df.csv"),
                      mapping_object={"Strat_Series": ('rock2', 'rock1')}
                      )

#%%

# Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = universal_cokriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)
