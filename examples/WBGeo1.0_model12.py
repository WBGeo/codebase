# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from scipy.interpolate import griddata
import meshio

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.interpolator_components.geo_inr import geo_inr_interpolator
from core.interpolator_components.loopstructural import loop_structural_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components import export_mesh_moose
from core.meshing_components.meshing_moose import create_mesh_moose
from core.meshing_components.grid_generator import create_surface_grid
from core.meshing_components.grid_generator import sort_points_by_x_y
from core.meshing_components.grid_generator import sort_surfaces_by_z
from core.meshing_components.grid_generator import store_points_in_array
from core.meshing_components.store_grid_data import create_surfaces_with_grids_for_bottom_and_top
from core.meshing_components.store_grid_data import read_refinement_file
from core.meshing_components.store_grid_data import create_intermediate_layers
from core.meshing_components.node_element_generator import adjust_z_values
from core.meshing_components.node_element_generator import create_hexahedral_elements_with_nodes
from core.meshing_components.exodus_mesh import Exos_inputs
from core.meshing_components.plots import plot_points_with_ids
from core.meshing_components.plots import plot_surfaces
#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model_12',
                      extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                      resolution=np.array([125, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model12_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model12_orientations_df.csv"),
                      mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
# results_test = ordinary_kriging_interpolator(data_test, var_range=500)
results_test = rbf_interpolator(data_test, kernel='cubic', epsilon=1)
# results_test = geo_inr_interpolator(data_test)
#results_test = loop_structural_interpolator(data_test)
# results_test, combined_scalar_field = ordinary_kriging_interpolator(data_test)


#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

# 4: Meshing for Process Simulation
# export_mesh_moose.export_data_to_moose(results_test)

#%%

# 5: Create mesh
mesh_test = create_mesh_moose(results_test, name="Model_12_OK")

#%%

# 5.5: Plot the mesh in 3D
plot_mesh_3d(mesh_test, data_test)

############################### Create Explicit mesh ####################################################
# Extract extent values
extent = results_test.extent
min_x, max_x, min_y, max_y, min_z, max_z = extent

# Create the interpolated surfaces dictionary
interpolated_surfaces, n_gx, n_gy = create_surface_grid(results_test)

# Convert the dictionary values (interpolated grids) to pandas DataFrames
dataframes = {
    surface_key: pd.DataFrame(grid, columns=["X", "Y", "Z"])
    for surface_key, grid in interpolated_surfaces.items()
}

# Convert the dictionary values (interpolated grids) to a list of DataFrames
dataframes_list = [pd.DataFrame(grid, columns=["X", "Y", "Z"]) for grid in interpolated_surfaces.values()]

# Sort the dataframes based on first x and  then y components
sorted_dataframes = sort_points_by_x_y(dataframes_list)

# Sort the surfaces based on z component
sorted_surfaces = sort_surfaces_by_z(sorted_dataframes)

# Store the data in an array
output_array=store_points_in_array(sorted_surfaces)
output_array = np.array(output_array[0])


# Create bottom and top surfaces
bottom_top_surfaces = create_surfaces_with_grids_for_bottom_and_top(min_x, max_x, min_y, max_y, min_z, max_z, n_gx, n_gy)

# Read refinement file
refinement_data = read_refinement_file('refine.txt', len(dataframes))

# Create points between surfaces
updated_output = create_intermediate_layers(bottom_top_surfaces, output_array, refinement_data, n_gx, n_gy)

# Plot all surfaces
plot_surfaces(bottom_top_surfaces, n_gx, n_gy)
plot_points_with_ids(updated_output, n_gx, n_gy)

# check if points are within a given distance tolerance and move one of them in the Z direction 
# by a specified z_threshold when the condition is met
adjusted_array= adjust_z_values(updated_output, n_gx, n_gy, z_threshold = 0.1, tolerance = 1)

# Create elements and assign the surface ids to them
elements, nodes = create_hexahedral_elements_with_nodes(adjusted_array, n_gx, n_gy)

# Initialize the Exos class
exo_in = Exos_inputs(nodes, elements)

# Get surface IDs of all nodes
print(exo_in.surface_id(), 'surface_id')

# Access boundary nodes directly from the Bound object
print(exo_in.bound.front(), 'front')
print(exo_in.bound.top(), 'top')
print(exo_in.bound.top_coords(), 'top')
print(exo_in.bound.left_surface_id(), 'left')


# Get the unique surface IDs
unique_surface_ids = np.unique(exo_in.elements_array[:, -1])

# Initialize a dictionary to store elements for each surface ID
elements_by_surface_id = {}

# Loop through each unique surface ID and save elements
for surface_id in unique_surface_ids:
    elements_n = exo_in.get_elements_by_surface_id(surface_id)
    elements_by_surface_id[int(surface_id)] = np.array(elements_n)

# Initialize the `cells` list dynamically
cells = [("hexahedron", elements.tolist()) for elements in elements_by_surface_id.values()]

# Exclude the first and last columns
formatted_nodes = nodes[:, 1:-1]

# Convert the resulting array to floating-point format
formatted_nodes = np.array(formatted_nodes.astype(float))

# Create a dictionary to store the boundary values
boundary_dict = {
    'front': exo_in.bound.front(),
    'top': exo_in.bound.top(),
    'left': exo_in.bound.left(),
    'right': exo_in.bound.right(),
    'bottom': exo_in.bound.bottom(),
    'back': exo_in.bound.back()
}

# Create the meshio.Mesh object
mesh = meshio.Mesh(
    points=formatted_nodes,
    cells=cells,
    point_sets=boundary_dict
)
# Write the mesh to an Exodus file
output_filename = "model_WG12.exo"
mesh.write(output_filename, file_format="exodus")

print(f"Exodus file '{output_filename}' created successfully!")


