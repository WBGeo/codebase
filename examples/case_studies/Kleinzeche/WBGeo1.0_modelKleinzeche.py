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

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 90, 0, 10, 55, 125),  # Example grid extent
    resolution=(180, 10, 140)  # Example resolution
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Kleinzeche',
                                      fault_surface_points=pd.read_csv(
                                        cwd + "/examples/input_data/modelKleinzeche_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                        cwd + "/examples/input_data/modelKleinzeche_orientations_df.csv"),
                                      fault_names=['Stoerung']
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
general_faults.compute_fault_domains(fault_frame)

#%%

# Plot the fault model results (2D and 3D possible)
plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Kleinzeche',
                                             mapping_object={"Strat_Series3": ('Schottertragschicht'),
                                                             "Strat_Series2": ('Schlufstein_locker'),
                                                             "Strat_Series1": ('Schlufstein_tonig_C',
                                                                               'Schlufstein_sandig_C',
                                                                               'Schlufstein_tonig_B',
                                                                               'Schlufstein_sandig_B',
                                                                               'Schlufstein_tonig_A',
                                                                               'Schlufstein_sandig_A')
                                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/modelKleinzeche_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/modelKleinzeche_orientations_df.csv"))

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_frame=fault_frame
                                       )

frame.structural_groups[0].structural_elements[0].set_color('#959595')
frame.structural_groups[1].structural_elements[0].set_color('#00c401')
frame.structural_groups[2].structural_elements[0].set_color('#828701')
frame.structural_groups[2].structural_elements[1].set_color('#ec7a10')
frame.structural_groups[2].structural_elements[2].set_color('#828701')
frame.structural_groups[2].structural_elements[3].set_color('#ec7a10')
frame.structural_groups[2].structural_elements[4].set_color('#828701')
frame.structural_groups[2].structural_elements[5].set_color('#ec7a10')

frame.detailed_report()

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
frame["Strat_Series3"].set_interpolation_method("Universal Co-Kriging")
frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")
frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")

# set fault activity verbose
frame.set_fault_activity_by_group(fault_name="Stoerung", group_name="Strat_Series2")


frame.fault_activity_verbose


#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

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




