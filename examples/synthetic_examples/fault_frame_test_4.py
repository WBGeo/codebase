import pandas as pd
import os

import core.structuralmodeling_components.general_faults
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

# load the csv
df1 = pd.read_csv(cwd + "/examples/input_data/model2_surface_points_df.csv")
fault_surface_points_df = df1[df1["formation"] == "fault"]
structural_surface_points_df = df1[df1["formation"] != "fault"]

df2 = pd.read_csv(cwd + "/examples/input_data/model2_orientations_df.csv")
fault_orientations_df = df2[df2["formation"] == "fault"]
structural_orientations_df = df2[df2["formation"] != "fault"]

new_row = structural_orientations_df.iloc[-1].copy()
new_row["X"] = 2000
structural_orientations_df.loc[len(structural_orientations_df)] = new_row

new_row = structural_orientations_df.iloc[0].copy()
new_row["X"] = 2300
structural_orientations_df.loc[len(structural_orientations_df)] = new_row

# Add another element on top
# Select rows where formation == "rock3" and make a copy
rock3_copy = structural_surface_points_df[
    structural_surface_points_df["formation"] == "rock3"
].copy()

# Modify the copied rows
rock3_copy["formation"] = "rock4"
rock3_copy["Z"] = rock3_copy["Z"] + 50

# Append to the original dataframe
structural_surface_points_df = pd.concat(
    [structural_surface_points_df, rock3_copy],
    ignore_index=True
)

#%%

grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(62, 25, 50)  # Example resolution
)

#%%

grid.extent

#%%

fault_frame = core.structuralmodeling_components.general_faults.build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=["fault"],
    colors=["#A9A9A9"],
    grid=grid
)

#%%

fault_frame.detailed_report()
general_faults.compute_fault_domains(fault_frame)

#%%

fault_frame.plot_fault_domain_section(axis='y', index=12)

#%%

# Plot the fault meshes using pyvista
plot_fault_frame_3D(fault_frame)


#%%

# Create a StructuralFrame
frame = general.build_structural_frame({"Top": ('rock4', 'rock3'), "Bot": ('rock2', 'rock1')},
                                               grid,
                                               structural_surface_points_df,
                                               structural_orientations_df)
frame.detailed_report()

#%%

plot_structural_model_3D(frame=frame, fault_frame=fault_frame, show_surface_meshes=False)


#%%

# UCK
# frame["Top"].set_interpolation_method("Universal Co-Kriging")
# frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# OK
frame["Top"].set_interpolation_method("Ordinary Kriging")
frame["Bot"].set_interpolation_method("Ordinary Kriging")
frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=1, variogram_model="spherical")

# RBF
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")
# frame["Top"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)
# frame["Bot"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)

# LOOP
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")

# GEOINR
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")


#%%

frame.detailed_report()

#%%

general.compute_structural_model(
    frame,
    fault_frame=fault_frame,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                        fault_frame=fault_frame,
                        axis='y',
                        show_input_data=True,
                        index=0)

#%%

plot_structural_model_3D(frame=frame,
                        fault_frame=fault_frame,
                        mesh_type="masked",
                        show_orientations=True)

#%%

frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)

#%%

frame.plot_age_mask_section(group_nr=0, axis='y', index=12)





