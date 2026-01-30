# Importing necessary libraries
import numpy as np
import pandas as pd
import os

import core.structuralmodeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

surface_points = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_surface_points_downsampled.csv")
orientations = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_orientations_downsampled.csv")

surface_points_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_faults_surface_points_downsampled.csv")
orientations_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_faults_orientations_downsampled.csv")

#%%

grid = RegularGrid(
    extent=(surface_points["X"].min(), surface_points["X"].max(),
            surface_points["Y"].min(), surface_points["Y"].max(),
            surface_points["Z"].min(), surface_points["Z"].max()),
    resolution=(50, 50, 50)
)

#%%

# Create a StructuralFrame
frame = general.build_structural_frame({
                                        "Main": ('01_top_hannover',
                                                 # '02_top_dethlingen',
                                                 '03_top_ebs',
                                                 # '04_top_rockel',
                                                 '05_top_havel',
                                                 # '06_top_vulkanit',
                                                 '07_top_karbon')
                                       },
                                        grid=grid,
                                        surface_points=surface_points,
                                        orientations=orientations)

# frame = general.build_structural_frame({
#                                         "01": ('01_top_hannover'),
#                                         "02": ('02_top_dethlingen'),
#                                         "03": ('03_top_ebs'),
#                                         "04": ('04_top_rockel'),
#                                         "05": ('05_top_havel'),
#                                         "06": ('06_top_vulkanit'),
#                                         "07": ('07_top_karbon')
#                                        },
#                                         grid,
#                                         df)

frame.detailed_report()

#%%

plot_structural_model_3D(frame=frame, show_surface_meshes=False)

#%%

# UCK
frame["Main"].set_interpolation_method("Universal Co-Kriging")

# RBF
# frame["Main"].set_interpolation_method("Radial Basis Function")
# frame["01"].set_interpolation_method("Radial Basis Function")
# frame["02"].set_interpolation_method("Radial Basis Function")
# frame["03"].set_interpolation_method("Radial Basis Function")
# frame["04"].set_interpolation_method("Radial Basis Function")
# frame["05"].set_interpolation_method("Radial Basis Function")
# frame["06"].set_interpolation_method("Radial Basis Function")
# frame["07"].set_interpolation_method("Radial Basis Function")


#%%

general.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_3D(frame=frame, show_surface_meshes=True)

 #%%

 # TODO: Does not work because cross cutting, maybe still allow visualization?

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_GSB',
                                      fault_surface_points=surface_points_faults,
                                      fault_orientations=orientations_faults)

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    fault_surface_points_df=data_faults.fault_surface_points,
    fault_orientations_df=data_faults.fault_orientations,
    fault_names=['29', 'F21n', 'F27', 'F28', 'F29', 'F9'],
    colors=["#A9A9A9"]*6,
    grid=grid
)

fault_frame.detailed_report()

#%%

# Compute fault domains
general_faults.compute_fault_domains(fault_frame)

#%%

# Plot fault domains (2D and 3D possible)
fault_frame.plot_fault_domain_section(axis='y', index=12)
plot_fault_frame_3D(fault_frame)

