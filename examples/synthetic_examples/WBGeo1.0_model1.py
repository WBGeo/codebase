# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(25, 25, 25)  # Example resolution
)

#%%
# WORKFLOW Model 1: no faults, no unconformities, 2 stratigraphic series

# Component 1: input input_data
data_elements = InputData_StructuralElements(name='Model_1',
                                             mapping_object={"Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model1_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model1_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements.mapping_object,
                                       grid,
                                       data_elements.surface_points,
                                       data_elements.orientations)
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y')
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Set interpolation methods for each stratigraphic series
frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")

# Set interpolation parameters if needed
frame["Strat_Series1"].configure_interpolation_params(range=1000, anisotropy_scaling_z=0.5, variogram_model="spherical")

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    fault_frame=None,
    extract_meshes=True,
    verbose=True,
)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_structural_model_2D(frame, axis='y')
plot_structural_model_3D(frame, show_surface_meshes=True)

#%%

# 4: Meshing for Process Simulation
# mesh_test = create_structured_mesh_data(
#     geomodel_result=results_test,
#     refinement_data=[10,10,10],
#     z_threshold=0.1,
#     tolerance=1
# )
