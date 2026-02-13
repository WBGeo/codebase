# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_model_2D, plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 9: 5 faults, 3 unconformities, 4 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(75, 50, 50)  # Example resolution
)


#%%

# Create input data for the fault elements
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

# Plot the fault input input_data (2D and 3D possible)
plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%%

# Compute fault model result
fault_model_result = general_faults.compute_fault_domains(fault_frame)

#%%

# Plot the fault model results (2D and 3D possible)
plot_fault_model_2D(fault_model_result.fault_frame)
plot_fault_model_3D(fault_model_result.fault_frame)


#%%

# Create input data for the structural elements
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

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_model_results=fault_model_result
                                       )

frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

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

# Set another interpolation method per group
# frame["A"].set_interpolation_method("GeoINR")
# frame["B"].set_interpolation_method("GeoINR")
# frame["C"].set_interpolation_method("GeoINR")
# frame["D"].set_interpolation_method("GeoINR")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["A"].configure_interpolation_params()
# frame["B"].configure_interpolation_params()
# frame["C"].configure_interpolation_params()
# frame["D"].configure_interpolation_params()

# frame.detailed_report()

#%%

# Compute structural model result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# Plot the results (2D and 3D possible)
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=3, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=2, axis='y', index=12)

#%%

# Optional: Compute gradients at the surface mesh vertices
# from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
#
# gradients_dict, gradients_faults_dict = surface_mesh_gradients.get_surface_mesh_gradients(structural_model_result,
#                                                                                           mesh_type="unmasked")
# surface_mesh_gradients.plot_surface_mesh_gradients(structural_model_result,
#                                                    gradients_dict,
#                                                    gradients_faults_dict,
#                                                    mesh_type="unmasked")

#%%

# TODO: Meshing needs to be adapted to work with the new Structural Modeling output

# Meshing for Process Simulation
# mesh_test = create_structured_mesh_data(
#     geomodel_result=results_test,
#     refinement_data=[10,10,10],
#     z_threshold=0.1,
#     tolerance=1
# )


