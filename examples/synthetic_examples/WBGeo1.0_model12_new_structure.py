# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData_StructuralElements

from core.structuralmodeling_components.interpolators_per_group import general

from core.visualization_components_new import visualize_structural_frame, plot_structural_slice


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series
# Component 1: Input input_data
data_test = InputData_StructuralElements(name='Model_12',
                                         extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                                         resolution=np.array([100, 50, 50]),
                                         surface_points=pd.read_csv(
                          cwd + "/examples/input_data/model12_surface_points_df.csv"),
                                         orientations=pd.read_csv(
                          cwd + "/examples/input_data/model12_orientations_df.csv"),
                                         mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                                         )

#%%

frame = general.build_structural_frame(data_test.mapping_object,
                                        data_test.extent,
                                        data_test.resolution,
                                        data_test.surface_points,
                                        data_test.orientations)



#%%

# Plot a slice of the structural model
plot_structural_slice(frame, axis='y', index=10, show_scalar_contours=True)

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

frame.detailed_report()


#%%

frame["Strat_Series1"].set_interpolation_method("Loop Structural")
frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

#frame["Strat_Series1"].set_interpolation_method("GeoML")
#frame["Strat_Series2"].set_interpolation_method("GeoML")

frame.detailed_report()

#%%

from core.structuralmodeling_components.interpolators_per_group import general_updated

general_updated.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)



#%%

# Plot a slice of the structural model
plot_structural_slice(frame, lith_block=block, axis='y', index=0, show_scalar_contours=True)


#%%

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_surface_meshes=True, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

# plot section of scalar field for group "Strat_Series1"
import matplotlib.pyplot as plt
group = frame["Strat_Series1"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series1"]["rock3"].scalar_value, frame["Strat_Series1"]["rock4"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

group = frame["Strat_Series2"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series2"]["rock1"].scalar_value, frame["Strat_Series2"]["rock2"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

# %%

frame.structural_groups[0].structural_elements[0].edges
