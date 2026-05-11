# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_FaultElements
from core.structural_modeling_components import general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_fault_model_2D, plot_fault_model_3D, plot_fault_input_data_3D)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 4: 2 faults, no unconformities, 1 stratigraphic group

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),
    resolution=(100, 50, 50)
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_4',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model4/input_data/geological_data/model4_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model4/input_data/geological_data/model4_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
                                      )

#%%

# Plot fault input data without model context
plot_fault_input_data_3D(data_faults)

#%%

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid
)

fault_frame.detailed_report()

#%%

# Plot the fault input data (2D and 3D possible)
plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%%

# Compute fault model result
general_faults.compute_fault_domains(fault_frame)

#%%

# This model is expected to fail at compute_fault_domains above.
# The two faults in this model cross-cut each other, which the fault domain algorithm
# cannot handle: cross-cutting faults create grid cells whose domain membership is
# contradictory (claimed by more than one fault side simultaneously), producing an
# inconsistent domain map that the downstream interpolation cannot resolve.
# Non-intersecting faults are required.
