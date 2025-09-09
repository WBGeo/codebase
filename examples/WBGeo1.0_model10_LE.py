# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys

from core.liquidEarth.le_push_data import push_geosolution_to_le
from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 10: 2 faults, 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model 10',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([125, 50, 50]),
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

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = universal_cokriging_interpolator(data_test)

link= push_geosolution_to_le(results_test, "wbgeo_data_test", "model10","PUT_TOKEN_HERE")

#%%
print(link)