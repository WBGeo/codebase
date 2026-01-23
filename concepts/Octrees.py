#%%

import gempy as gp
import gempy_viewer as gpv
import pyvista as pv
import numpy as np

#%%

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/input_data/input_data/jan_models/"
# Create a GeoModel instance
geo_model = gp.create_geomodel(
    project_name='fold',
    extent=[0, 1000, 0, 1000, 0, 1000],
    refinement=7,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data + "model2_orientations.csv",
        path_to_surface_points=path_to_data + "model2_surface_points.csv"
    )
)
# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={"Strat_Series": ('rock2', 'rock1')}
)

# Compute the geological model
gp.compute_model(geo_model)

#%%

# dc_vertices_transformed = [model_instance.input_transform.apply_inverse(mesh.vertices) for mesh in
#                                model_instance.solutions.dc_meshes]
# dc_edges = [mesh.edges for mesh in model_instance.solutions.dc_meshes]

#%%

# TODO: No edges from this output as far as I can see
geo_model.solutions.octrees_output[0]

#%%

# Create object containing back transformed octree grids per level
array_list = []

for i in range(len(geo_model.solutions.octrees_output) - 1):
    array_list.append(
        geo_model.input_transform.apply_inverse(geo_model.solutions.octrees_output[i].grid_corners.values))

octree_levels = np.array(array_list, dtype=object)

#%%

colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
          '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd']

level_show = 6

plotter = pv.Plotter()

for level in range(level_show):
    plotter.add_mesh(pv.PolyData(octree_levels[level]),
                     render_points_as_spheres=True,
                     point_size=5,
                     color=colors[level])

# Set the bounds and grid of the plotter
plotter.show_bounds(bounds=geo_model.grid.extent,
                    location="furthest",
                    grid=True)

# Display the interactive plot
plotter.show()

#%%

gpv.plot_2d(geo_model)
