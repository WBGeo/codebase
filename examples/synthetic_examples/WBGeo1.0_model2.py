# Importing necessary libraries
import pandas as pd
import os

import core.structural_modeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(125, 50, 50)  # Example resolution
)

#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_2',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model2_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model2_orientations_df.csv"))

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    fault_surface_points_df=data_faults.fault_surface_points,
    fault_orientations_df=data_faults.fault_orientations,
    fault_names=["fault"],
    grid=grid
)

fault_frame.detailed_report()


#%%

# Compute fault domains
general_faults.compute_fault_domains(fault_frame)

#%%

plot_fault_model_3D(fault_frame)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_2',
                                             mapping_object={
                                                 "Strat_Series2": ('rock4', 'rock3'),
                                                 "Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model2_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model2_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements.mapping_object,
                                       grid,
                                       data_elements.surface_points,
                                       data_elements.orientations,
                                       fault_frame=fault_frame)

frame.detailed_report()

#%%

frame.fault_activity_verbose


#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)


#%%

# Plot fault frame sections and 3D
fault_frame.plot_fault_domain_section(axis='y', index=12)
plot_fault_model_3D(fault_frame)


#%%

# Set interpolation methods for each stratigraphic series

# UCK
# frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")
# frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

# OK TODO: Find good parameters
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")
# frame["Strat_Series2"].set_interpolation_method("Ordinary Kriging")
# # Set interpolation parameters if needed
# frame["Strat_Series1"].configure_interpolation_params(range=500, variogram_model="gaussian", anisotropy_scaling_z=0.1)
# frame["Strat_Series2"].configure_interpolation_params(range=500, variogram_model="gaussian", anisotropy_scaling_z=0.3)

# RBF TODO: Find good parameters
frame["Strat_Series1"].set_interpolation_method("Radial Basis Function")
frame["Strat_Series2"].set_interpolation_method("Radial Basis Function")


# GeoINR
# frame["Strat_Series1"].set_interpolation_method("GeoINR")
# frame["Strat_Series2"].set_interpolation_method("GeoINR")

# Loop
# frame["Strat_Series1"].set_interpolation_method("Loop Structural")
# frame["Strat_Series2"].set_interpolation_method("Loop Structural")

frame.detailed_report()


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)


#%%

frame.plot_scalar_field_section(group_nr=0, axis='y', index=12)
frame.structural_groups[1].structural_elements[1].scalar_value

#%%

plot_structural_model_2D(frame=frame,
                         axis='y',
                         show_input_data=True,
                         index=0)

#%%

plot_structural_model_3D(frame=frame,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)

#%%
# TODO: Update to new structure
# Calculate gradients at the surface mesh vertices
from core.utility import surface_mesh_gradients

points_list, vectors_list = surface_mesh_gradients.get_surface_mesh_gradients(results_test, mesh_type="unmasked")

#%%

# Plotting the gradient vector field

import pyvista as pv

colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
          '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
          '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

for i in range(len(points_list)):
    pdata = pv.PolyData(points_list[i])
    pdata["vectors"] = vectors_list[i]  # Add vector field

    # Create arrow glyphs
    arrows = pdata.glyph(orient="vectors", scale="vectors", factor=50)

    # Plot the arrows
    plotter.add_mesh(arrows, color=colors[i])
    plotter.add_mesh(
        pv.PolyData(results_test.surface_meshes_vertices[1][i],
                    np.insert(results_test.surface_meshes_edges[1][i], 0, 3, axis=1).ravel()),
        color=colors[i], style="surface")
plotter.show()

#%%

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

unit = 2

pdata = pv.PolyData(points_list[unit])
pdata["vectors"] = vectors_list[unit]  # Add vector field

# Create arrow glyphs
arrows = pdata.glyph(orient="vectors", scale="vectors", factor=50)

# Plot the arrows
plotter.add_mesh(arrows, color=colors[unit])
plotter.add_mesh(
    pv.PolyData(results_test.surface_meshes_vertices[1][unit],
                np.insert(results_test.surface_meshes_edges[1][unit], 0, 3, axis=1).ravel()),
    color=colors[unit], style="wireframe")
plotter.show()

# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    data_test=data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100, 100, 100, 100, 100, 500, 300, 100, 500, 300, 100, 300), (500, 500, 500, 500, 500, 900)],
    num_sources=2,
    sources=[(100, 300, 500), (400, 600, 700)],
    num_shafts=0,
    centers=[],
    axes=[],
    radii=[],
    num_planes=0,
    extra_planes=[],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=2,
    DISTANCE_THRESHOLD=80,
    PROJECTION_THRESHOLD=80,
    EXTRUSION_FACTOR=80,
    z_threshold=10
)

mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex = mesh_test.export_exodus("filename.exo")
mesh_vtu = mesh_test.export_vtu("filename.vtu")
