# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 7: 1 fault, 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model 7',
                      extent=np.array([0, 2500, 0, 1000, 0, 1000]),
                      resolution=np.array([50, 20, 20]),
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
