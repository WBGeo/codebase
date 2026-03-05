# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

surface_points = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_surface_points_downsampled.csv")
orientations = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_orientations_downsampled.csv")

surface_points_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_faults_surface_points_downsampled.csv")
orientations_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_faults_orientations_downsampled.csv")

print(len(surface_points), len(orientations))
print(len(surface_points_faults), len(orientations_faults))


#%%

margin = 100  # Add a margin of 100 units around the data
grid = RegularGrid(
    extent=(surface_points["X"].min()-margin, surface_points["X"].max()+margin,
            surface_points["Y"].min()-margin, surface_points["Y"].max()+margin,
            surface_points["Z"].min()-margin, surface_points["Z"].max()+margin),
    resolution=(50, 50, 50)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='GSB',
                                             mapping_object={
                                                "01": ('01_top_hannover'),
                                                "02": ('02_top_dethlingen'),
                                                "03": ('03_top_ebs'),
                                                "04": ('04_top_rockel'),
                                                "05": ('05_top_havel'),
                                                "06": ('06_top_vulkanit'),
                                                "07": ('07_top_karbon')
                                             },
                                             surface_points=surface_points,
                                             orientations=orientations
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid
                                       )

frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series
frame["01"].set_interpolation_method("Universal Co-Kriging")
frame["02"].set_interpolation_method("Universal Co-Kriging")
frame["03"].set_interpolation_method("Universal Co-Kriging")
frame["04"].set_interpolation_method("Universal Co-Kriging")
frame["05"].set_interpolation_method("Universal Co-Kriging")
frame["06"].set_interpolation_method("Universal Co-Kriging")
frame["07"].set_interpolation_method("Universal Co-Kriging")

frame.detailed_report()

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

