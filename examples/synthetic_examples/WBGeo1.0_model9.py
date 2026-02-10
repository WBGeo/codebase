# Importing necessary libraries
import pandas as pd
import os

import core.structuralmodeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_model_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 1500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(75, 50, 50)  # Example resolution
)

#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_9',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model9_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model9_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2', 'fault3', 'fault4', 'fault5']
                                      )

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
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

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_9',
                                             mapping_object={
                                                "A": ('rock1', 'rock2'),
                                                "B": ('rock3', 'rock4'),
                                                "C": ('rock5', 'rock6'),
                                                "D": ('rock7', 'rock8'),
                                                },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model9_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model9_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid = grid,
                                       fault_frame=fault_frame)
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# set fault activity verbose
frame.set_fault_activity_by_group(fault_name="fault1", group_name="C")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="D")
frame.set_fault_activity_by_group(fault_name="fault3", group_name="B")
frame.set_fault_activity_by_group(fault_name="fault4", group_name="C")
frame.set_fault_activity_by_group(fault_name="fault5", group_name="D")

frame.fault_activity_verbose

#%%

# Set interpolation methods for each stratigraphic series

# UK
frame["A"].set_interpolation_method("Universal Kriging")
frame["B"].set_interpolation_method("Universal Kriging")
# frame["C"].set_interpolation_method("Universal Kriging") # This one causes problems with UK
frame["D"].set_interpolation_method("Universal Kriging")

# frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)


#%%

plot_structural_model_2D(frame=structural_model_result.structural_frame,
                         axis='y',
                         show_input_data=True,
                         index=25)

#%%

plot_structural_model_3D(frame=structural_model_result.structural_frame,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
frame.plot_age_mask_section(group_nr=2, axis='y', index=12)

#%%





