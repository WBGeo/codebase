import numpy as np
import pandas as pd
import os

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.visualization_components_new import visualize_fault_frame

from core.structuralmodeling_components.interpolators_per_group import general_updated

#%%

cwd = os.getcwd()

#%%

# load the csv
df1 = pd.read_csv(cwd + "/examples/input_data/model7_surface_points_df.csv")
fault_surface_points_df = df1[df1["formation"] == "fault"]
structural_surface_points_df = df1[df1["formation"] != "fault"]

df2 = pd.read_csv(cwd + "/examples/input_data/model7_orientations_df.csv")
fault_orientations_df = df2[df2["formation"] == "fault"]
structural_orientations_df = df2[df2["formation"] != "fault"]

structural_orientations_df.dtypes

#%%

new_row = structural_orientations_df.iloc[-1].copy()
new_row["X"] = 2000
structural_orientations_df.loc[len(structural_orientations_df)] = new_row

new_row = structural_orientations_df.iloc[0].copy()
new_row["X"] = 2300
structural_orientations_df.loc[len(structural_orientations_df)] = new_row

structural_orientations_df


#%%

grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(125, 50, 50)  # Example resolution
)

fault_frame = general_updated.build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=["fault"],
    colors=["#A9A9A9"],
    grid=grid
)



#%%
fault_frame.detailed_report()

#%%

# Compute result for fault frame
fault_frame.compute_fault_domains()

#%%

# Plot the fault meshes using pyvista
visualize_fault_frame(fault_frame)

#%%

# plot slice of domain map
import matplotlib.pyplot as plt
plt.imshow(fault_frame.domain_map[:, 25, :], origin='lower', cmap='tab20')
plt.title('Fault Domain Map Slice at Y=25')
plt.xlabel('X Index')
plt.ylabel('Z Index')
plt.show()


#%%

# Create a StructuralFrame
frame = general_updated.build_structural_frame({"Top": ('rock3'), "Bot": ('rock2', 'rock1')},
                                               np.array([0, 2500, 0, 1000, 0, 1000]),
                                               np.array([125, 50, 50]),
                                               structural_surface_points_df,
                                               structural_orientations_df)
frame.detailed_report()


#%%

# UCK
frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# OK
# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
# frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
# frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

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

# TODO: Problem A: Missing input_data for top layer on both sides of fault block -
#  Solution: Add more input_data points
# TODO: Problem B: Need two units in top group for OK and RBF to work well
#  Solution: Add another element on top
# TODO: Problem C: For Gempy the saclar values are nor consistent among domains

general_updated.compute_structural_model(
    frame,
    fault_frame=fault_frame,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)



#%%

from core.visualization_components import plot_structural_slice_with_faults
# TODO: Something wrong with input input_data here (arrows)
plot_structural_slice_with_faults(frame=frame,
                                  fault_frame=fault_frame,
                                  lith_block=frame.get_LithBlock(),
                                  axis='y',
                                  show_input_data=False,
                                  index=0)

#%%

from core.visualization_components import visualize_structural_frame_with_faults
# TODO: Something wrong with input input_data here (arrows)
visualize_structural_frame_with_faults(frame=frame,
                                       fault_frame=fault_frame,
                                       mesh_type="masked",
                                       show_orientations=False)

#%%

frame.structural_groups[1].scalar_field

# plot slice of scalar field
import matplotlib.pyplot as plt
plt.imshow(frame.structural_groups[1].scalar_field[:, 25, :].T, origin='lower', cmap='viridis')
plt.title('Scalar Field Slice at Y=25 for Group Bot')
plt.xlabel('X Index')
plt.ylabel('Z Index')
plt.colorbar(label='Scalar Value')
plt.show()