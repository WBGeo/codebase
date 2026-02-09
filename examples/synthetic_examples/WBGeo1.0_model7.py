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
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(50, 50, 50)  # Example resolution
)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_7',
                                             mapping_object={
                                                 "Top": ('UnitD', 'UnitC'),
                                                 "Bot": ('UnitB', 'UnitA')
                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model7_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model7_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements.mapping_object,
                                       grid,
                                       data_elements.surface_points,
                                       data_elements.orientations)
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_7',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model7_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model7_orientations_df.csv"))

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    fault_surface_points_df=data_faults.fault_surface_points,
    fault_orientations_df=data_faults.fault_orientations,
    fault_names=["FaultA", "FaultB", "FaultC"],
    grid=grid
)

fault_frame.detailed_report()

#%%

# TODO: Order is reversed here comapred to naming

# set crazy colors
fault_frame.fault_elements[0].set_color("#FF0000") # red
fault_frame.fault_elements[1].set_color("#0000FF") # blue
fault_frame.fault_elements[2].set_color("#000000") # black

#%%

# check color
print(fault_frame.fault_elements[1].name, fault_frame.fault_elements[1].color)

#%%

# Compute fault domains
general_faults.compute_fault_domains(fault_frame)

#%%

print(fault_frame.fault_elements[1].name, fault_frame.fault_elements[1].get_separated_domains())

#%%

# Plot fault domains (2D and 3D possible)
fault_frame.plot_fault_domain_section(axis='y', index=12)
plot_fault_model_3D(fault_frame)

#%%

# Set interpolation methods for each stratigraphic series

# UCK
frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# OK
# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
# frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
# frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

# RBF
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")
# frame["Top"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)
# frame["Bot"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)

# GeoINR
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")

# Loop Structural
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")


frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    fault_frame=fault_frame,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                         fault_frame=fault_frame,
                         axis='y',
                         show_input_data=True,
                         index=0)

#%%

plot_structural_model_3D(frame=frame,
                         fault_frame=fault_frame,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=