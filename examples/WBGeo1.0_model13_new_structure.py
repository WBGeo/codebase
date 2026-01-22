# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData

from core.structuralmodeling_components.interpolators_per_group import general
from core.visualization_components_new import visualize_structural_frame, plot_structural_slice


#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series
# Component 1: Input data
data_test = InputData(name='Model_13',
                      extent=np.array([0, 1000, 0, 500, 0, 1000]),
                      resolution=np.array([50, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model13_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model13_orientations_df.csv"),
                      mapping_object={
                          "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
                          "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
                          "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1')},
                      )

#%%

frame = general.build_structural_frame(data_test.mapping_object,
                                        data_test.extent,
                                        data_test.resolution,
                                        data_test.surface_points,
                                        data_test.orientations)
frame.summary()

#%%

# Plot a slice of the structural model
plot_structural_slice(frame, axis='y', index=10, show_scalar_contours=True)

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

# frame["Shallow_Strat"].set_interpolation_method("Loop Structural")
frame["Shallow_Strat"].set_interpolation_method("Ordinary Kriging")
frame["Medium_Strat"].set_interpolation_method("Universal Co-Kriging")
frame["Deep_Strat"].set_interpolation_method("GeoINR")

# frame["Deep_Strat"].set_interpolation_method("Ordinary Kriging")

# frame.pretty_print()

frame.summary()

#%%

# frame["Shallow_Strat"].configure_interpolation_params(interpolator_type="FDI")
frame["Shallow_Strat"].configure_interpolation_params(range=1000, anisotropy_scaling_z=0.3)
# # frame["Deep_Strat"].configure_interpolation_params(range=1000, anisotropy_scaling_z=0.3)

#%%

frame.detailed_report()

#%%

# Compute solution
frame, block = general.combined_interpolator(frame)


#%%

# Plot a slice of the structural model
plot_structural_slice(frame, lith_block=block, axis='y', index=0, show_scalar_contours=True)


#%%

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_surface_meshes=True, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

#%%

frame.structural_groups[0]

#%%

# plot section of scalar field
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(frame.structural_groups[0]._scalar_field[:, frame.structural_groups[0]._scalar_field.shape[1] // 2, :].T, cmap='viridis', origin='lower')
plt.colorbar(label='Scalar Value')
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.show()


