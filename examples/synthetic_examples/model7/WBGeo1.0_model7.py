# Simulate the nodesapi at runtime to also check that (de)serialization is set up correctly
from examples.pydantic_nodesapi_simulation import register_as_test_nodes_api
register_as_test_nodes_api()

# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D, plot_fault_input_data_3D, plot_input_data_3D)

from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 7: no faults, 1 unconformity, 2 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(50, 50, 50)  # Example resolution
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_7',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model7/input_data/geological_data/model7_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model7/input_data/geological_data/model7_orientations_df.csv"),
                                      fault_names=["FaultA", "FaultB", "FaultC"])

#%%

plot_fault_input_data_3D(data_faults)

#%%

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

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_7',
                                             mapping_object={
                                                 "Top": ('UnitD', 'UnitC'),
                                                 "Bot": ('UnitB', 'UnitA')
                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model7/input_data/geological_data/model7_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model7/input_data/geological_data/model7_orientations_df.csv")
                                             )

#%%

plot_input_data_3D(data_elements)

#%%

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

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
# frame["Top"].set_interpolation_method("Universal Co-Kriging")
# frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Top"].configure_interpolation_params()
# frame["Bot"].configure_interpolation_params()

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
# frame.plot_scalar_field_section(group_nr=0, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)

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

# Meshing for Process simulation_components
mesh_test = create_unstructured_mesh_data(
     geomodel_result=structural_model_result,
     tolerance=50,
     mesh_size=30,
     curve_mesh_size=5,
    DISTANCE_THRESHOLD = 40,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
     z_threshold = 20
 )
