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
    extent=(0, 2000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 50, 50)  # Example resolution
)

#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_8',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model8_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model8_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
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
data_elements = InputData_StructuralElements(name='Model_8',
                                             mapping_object={
                                                "Top": ('rock6', 'rock5'),
                                                "Mid": ('rock4', 'rock3'),
                                                "Bot": ('rock2', 'rock1'),
                                                },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model8_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model8_orientations_df.csv")
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
frame.set_fault_activity_by_group(fault_name="fault1", group_name="Mid")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="Bot")

frame.fault_activity_verbose

#%%

# Set interpolation methods for each stratigraphic series

# UCK
# frame["Top"].set_interpolation_method("Universal Co-Kriging")
# frame["Mid"].set_interpolation_method("Universal Co-Kriging")
# frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# UCK
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Mid"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")

# OK --> dont have working parameters here
frame["Top"].set_interpolation_method("Ordinary Kriging")
frame["Mid"].set_interpolation_method("Ordinary Kriging")
frame["Bot"].set_interpolation_method("Ordinary Kriging")

# GeoINR
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Mid"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")

# LoopStructural
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Mid"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")
# frame["Top"].configure_interpolation_params(interpolator_type="FDI")
# frame["Mid"].configure_interpolation_params(interpolator_type="FDI")
# frame["Bot"].configure_interpolation_params(interpolator_type="FDI")

frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)


#%%

plot_structural_model_2D(frame=result.structural_frame,
                         axis='y',
                         show_input_data=True,
                         index=25)

#%%

plot_structural_model_3D(frame=result.structural_frame,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
frame.plot_age_mask_section(group_nr=2, axis='y', index=12)

#%%





