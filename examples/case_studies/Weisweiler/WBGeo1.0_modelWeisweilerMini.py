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

# Create a grid for the model
grid = RegularGrid(
    extent=(5623500, 5640000, 32304500, 32305500, -3000, 500),  # Example grid extent
    resolution=(250, 20, 125)  # Example resolution
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_9',
                                             mapping_object={
                                                "Strat_Series1":
                                                    ('BreitgangFM',
                                                     'KrebsTraufeFM',
                                                     'WilhelmineFM',
                                                     'ObererKohlenkalkGP',
                                                     'MittlererKohlenkalkGP',
                                                     'CondrozGP')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/case_studies/Weisweiler/input_data/modelWeisweilerMini_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/case_studies/Weisweiler/input_data/modelWeisweilerMini_orientations_df.csv")
                                             )

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid)

frame.detailed_report()


#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")

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






