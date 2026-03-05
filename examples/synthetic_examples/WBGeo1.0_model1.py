# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D)

from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d


#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 1: no faults, no unconformities, 2 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(25, 25, 25)  # Example resolution
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_1',
                                             mapping_object={"Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model1_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model1_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid
                                       )

frame.detailed_report()


#%%

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Default interpolation method is RBF, but we can set it to something else if we want

# # UK
# frame["Strat_Series1"].set_interpolation_method("Universal Kriging")
#
# # UCK
# frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")
#
# # OK
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")
#
# # Loop
# frame["Strat_Series1"].set_interpolation_method("Loop Structural")
#
# # GeoINR
# frame["Strat_Series1"].set_interpolation_method("GeoINR")


# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()

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

# Explicit Structured meshing
mesh_result = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=(10,10,10),
    z_threshold=0.1,
    tolerance=1
)

# Explicit Unstructured meshing
# mesh_result = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     z_threshold=0.1,
#     tolerance=1
# )

#%%

# Plot the meshing results
plot_mesh_3d(mesh_result, structural_model_result, show_plotter=True)
