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

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 50, 50)  # Example resolution
)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_5',
                                             mapping_object={
                                                 "Strat_Series2": ('rock4', 'rock3'),
                                                 "Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model5_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model5_orientations_df.csv")
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

# Set interpolation methods for each stratigraphic series

# UCK
frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")
frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

# OK
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")
# frame["Strat_Series2"].set_interpolation_method("Ordinary Kriging")
# # Set interpolation parameters if needed
# frame["Strat_Series1"].configure_interpolation_params(range=500, variogram_model="gaussian", anisotropy_scaling_z=0.3)
# frame["Strat_Series2"].configure_interpolation_params(range=500, variogram_model="gaussian", anisotropy_scaling_z=0.3)

# RBF
# frame["Strat_Series1"].set_interpolation_method("Radial Basis Function")
# frame["Strat_Series2"].set_interpolation_method("Radial Basis Function")
# # Set interpolation parameters if needed
# frame["Strat_Series1"].configure_interpolation_params(kernel="linear", epsilon=1)
# frame["Strat_Series2"].configure_interpolation_params(kernel="linear", epsilon=1)

# GeoINR
# frame["Strat_Series1"].set_interpolation_method("GeoINR")
# frame["Strat_Series2"].set_interpolation_method("GeoINR")

# Loop
# frame["Strat_Series1"].set_interpolation_method("Loop Structural")
# frame["Strat_Series2"].set_interpolation_method("Loop Structural")

frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                         fault_frame=None,
                         axis='y',
                         show_input_data=True,
                         index=0)

#%%

plot_structural_model_3D(frame=frame,
                         fault_frame=None,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)


#%%


#%%

# TODO: Adapt meshing to new structural modeling components

# 4: Meshing for Process Simulation
# mesh_test = create_structured_mesh_data(
#    geomodel_result=results_test,
#    refinement_data=[25, 21, 16, 5, 6],
#    z_threshold=0.1,
#    tolerance=1
# )

# Generate mesh
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    tolerance=50,
    mesh_size=10,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10,
    extent=[],
    buffer_dist=0,
    smooth =3
)

#%%


