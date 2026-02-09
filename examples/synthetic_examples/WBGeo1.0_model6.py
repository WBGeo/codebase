# Importing necessary libraries
import pandas as pd
import os

import core.structuralmodeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_model_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 1000, 0, 500, 0, 1000),  # Example grid extent
    resolution=(100, 50, 100)  # Example resolution
)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_6',
                                             mapping_object={
                                              "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
                                              "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
                                              "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1')
                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model6_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model6_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements.mapping_object,
                                       grid,
                                       data_elements.surface_points,
                                       data_elements.orientations)
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Set interpolation methods for each stratigraphic series

# UCK
# frame["Shallow_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Medium_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Deep_Strat"].set_interpolation_method("Universal Co-Kriging")

# OK
# frame["Shallow_Strat"].set_interpolation_method("Ordinary Kriging")
# frame["Medium_Strat"].set_interpolation_method("Ordinary Kriging")
# frame["Deep_Strat"].set_interpolation_method("Ordinary Kriging")
# # # Set interpolation parameters if needed
# frame["Shallow_Strat"].configure_interpolation_params(range=1500, variogram_model="gaussian", anisotropy_scaling_z=0.3)
# frame["Medium_Strat"].configure_interpolation_params(range=1500, variogram_model="gaussian", anisotropy_scaling_z=0.3)
# frame["Deep_Strat"].configure_interpolation_params(range=1500, variogram_model="gaussian", anisotropy_scaling_z=0.3)

# RBF
frame["Shallow_Strat"].set_interpolation_method("Radial Basis Function")
frame["Medium_Strat"].set_interpolation_method("Radial Basis Function")
frame["Deep_Strat"].set_interpolation_method("Radial Basis Function")

# # Set interpolation parameters if needed
frame["Shallow_Strat"].configure_interpolation_params(kernel='multiquadric', epsilon=0.0001)
frame["Medium_Strat"].configure_interpolation_params(kernel='multiquadric', epsilon=0.0001)
frame["Deep_Strat"].configure_interpolation_params(kernel='multiquadric', epsilon=0.0001)

# GeoINR
# frame["Shallow_Strat"].set_interpolation_method("GeoINR")
# frame["Medium_Strat"].set_interpolation_method("GeoINR")
# frame["Deep_Strat"].set_interpolation_method("GeoINR")

# Loop
# frame["Shallow_Strat"].set_interpolation_method("Loop Structural")
# frame["Medium_Strat"].set_interpolation_method("Loop Structural")
# frame["Deep_Strat"].set_interpolation_method("Loop Structural")

frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                         fault_frame=None,
                         axis='y',
                         show_input_data=True,
                         index=0)

#%%

plot_structural_model_3D(frame=frame,
                         fault_frame=None,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)

#%%

# TODO: Adapt to new structure

mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=0,
    wells=[],
    num_sources=0,
    sources=[],
    num_shafts=0,
    centers=[],
    axes=[],
    radii=[],
    num_planes=0,
    extra_planes=[],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=2,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10,
    extent=[],
    buffer_dist=20,
    smooth =2
)


mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")



#%%

# 4.5: Plot the meshing result (only 3D at current state)
plot_mesh_3d(mesh_test, data_test)
