import numpy as np
import pandas as pd
import os

from core.grids.grid_classes import RegularGrid
from core.visualization_components_new import visualize_fault_frame

from core.interpolator_components.interpolators_per_group import general_updated

#%%

cwd = os.getcwd()

#%%

# load the csv
df1 = pd.read_csv(cwd + "/examples/data/model7_surface_points_df.csv")
fault_surface_points_df = df1[df1["formation"] == "fault"]
structural_surface_points_df = df1[df1["formation"] != "fault"]

df2 = pd.read_csv(cwd + "/examples/data/model7_orientations_df.csv")
fault_orientations_df = df2[df2["formation"] == "fault"]
structural_orientations_df = df2[df2["formation"] != "fault"]

structural_orientations_df

#%%

new_row = structural_orientations_df.iloc[-1].copy()
new_row["X"] = 2000
structural_orientations_df = pd.concat([structural_orientations_df, new_row.to_frame().T], ignore_index=True)

new_row = structural_orientations_df.iloc[0].copy()
new_row["X"] = 2300
structural_orientations_df = pd.concat([structural_orientations_df, new_row.to_frame().T], ignore_index=True)

structural_orientations_df


#%%

grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(50, 50, 50)  # Example resolution
)

fault_frame = general_updated.build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=["fault"],
    colors=["#A9A9A9"],
    grid=grid
)

#%%

grid.spacing

#%%
fault_frame.detailed_report()

#%%

# Compute result for fault frame
fault_frame.generate_fault_domains()

#%%

# Plot the fault meshes using pyvista
visualize_fault_frame(fault_frame)

#%%

# plot slice of domain map
import matplotlib.pyplot as plt
plt.imshow(fault_frame.domain_map[:, 25, :].T, origin='lower', cmap='tab20')
plt.title('Fault Domain Map Slice at Y=25')
plt.xlabel('X Index')
plt.ylabel('Z Index')
plt.colorbar(label='Domain ID')
plt.show()


#%%

# Create a StructuralFrame
frame = general_updated.build_structural_frame({"Top": ('rock3'), "Bot": ('rock2', 'rock1')},
                                               np.array([0, 2500, 0, 1000, 0, 1000]),
                                               np.array([50, 50, 50]),
                                               structural_surface_points_df,
                                               structural_orientations_df)
frame.detailed_report()


#%%

# TODO: Works after transposing in function
frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# TODO: Kind of works, needs high range
# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
# frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
# frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

# TODO: Works after transposing in function
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")
# frame["Top"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)
# frame["Bot"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)

# TODO: Works like a charm
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")

# TODO: Works after transposing in function
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")


#%%

frame.detailed_report()

#%%

# TODO: Problem A: Missing data for top layer on both sides of fault block -
#  Solution: Add more data points
# TODO: Problem B: Need to units in top group for OK and RBF to work well
#  Solution: Add another element on top

general_updated.combined_interpolator_with_domains(
    frame,
    fault_frame=fault_frame,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

from core.visualization_components_new_new import plot_structural_slice_with_faults

#%%

plot_structural_slice_with_faults(frame=frame, fault_frame=fault_frame, axis='y', index=25)

#%%

from core.visualization_components_new_new import visualize_structural_frame_with_faults

#%%

visualize_structural_frame_with_faults(frame=frame, fault_frame=fault_frame, mesh_type="masked")

#%%
