# Importing necessary libraries
import pandas as pd
import os

import core.structural_modeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults

#%%

cwd = os.getcwd()

# TODO: This model contains cross-cutting faults that are not yet handled in WBGeo1.0

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(80, 40, 40)  # Example resolution
)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_4',
                                             mapping_object={
                                              "Strat_Series1": ('rock4', 'rock3', 'rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model4_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model4_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements,
                                       grid
                                       )
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)


#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_4',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model4_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model4_orientations_df.csv"))

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    fault_surface_points_df=data_faults.fault_surface_points,
    fault_orientations_df=data_faults.fault_orientations,
    fault_names=["fault1", "fault2"],
    colors=["#A9A9A9", "#696969"],
    grid=grid
)

fault_frame.detailed_report()

#%%

# Compute fault domains
general_faults.compute_fault_domains(fault_frame)

#%%

# Plot fault domains (2D and 3D possible)
fault_frame.plot_fault_domain_section(axis='y', index=12)
plot_fault_model_3D(fault_frame)



