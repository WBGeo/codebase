# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.utility import surface_mesh_gradients
import pyvista as pv

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model Weisweiler Mini: 1 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Kleinzeche',
                      extent=np.array([0, 90, 0, 10, 55, 125]),
                      resolution=np.array([180, 20, 140]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/modelKleinzeche_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/modelKleinzeche_orientations_df.csv"),
                      mapping_object={"Strat_Series3": ('Schottertragschicht'),
                                        "Fault_Series1": ('Stoerung'),
                                        "Strat_Series2": ('Schlufstein_locker'),
                                        "Strat_Series1": ('Schlufstein_tonig_C',
                                                        'Schlufstein_sandig_C',
                                                        'Schlufstein_tonig_B',
                                                        'Schlufstein_sandig_B',
                                                        'Schlufstein_tonig_A',
                                                        'Schlufstein_sandig_A')},
                      faults=[False, True, False, False]
                      )


#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = universal_cokriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True, show_data=True,
        colors = ['#959595', '#000000', '#00c401', '#828701', '#ec7a10', '#828701',
                  '#ec7a10','#828701', '#ec7a10', "#a619e7"])

#%%

# TODO: For some reason the meshes are shifted in z-direction, this is a workaround
for j in range(len(results_test.surface_meshes_vertices)):
    for i in range(len(results_test.surface_meshes_vertices[j])):
        results_test.surface_meshes_vertices[j][i][:,2]= results_test.surface_meshes_vertices[j][i][:,2] - 55

#%%
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="unmasked",
        colors = ['#959595', '#000000', '#00c401', '#828701', '#ec7a10', '#828701',
                  '#ec7a10','#828701', '#ec7a10', "#a619e7"])

#%%

points_list, vectors_list = surface_mesh_gradients.get_surface_mesh_gradients(results_test, mesh_type="unmasked")



# Stack all arrays vertically
all_points = np.vstack(points_list)

# Get overall min and max per column (x, y, z)
min_vals = np.min(all_points, axis=0)
max_vals = np.max(all_points, axis=0)
print(min_vals, max_vals)

colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
        '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
        '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()
print('0')
for i in range(len(points_list)):
    pdata = pv.PolyData(points_list[i])
    pdata["vectors"] = vectors_list[i]  # Add vector field

    # Create arrow glyphs
    arrows = pdata.glyph(orient="vectors", scale="vectors", factor=5)

    # Plot the arrows
    plotter.add_mesh(arrows, color=colors[i])
    plotter.add_mesh(
                    pv.PolyData(results_test.surface_meshes_vertices[1][i],
                                np.insert(results_test.surface_meshes_edges[1][i], 0, 3, axis=1).ravel()),
                    color=colors[i])
plotter.show()
#%%
print('1')
# Create a PyVista dataset
# Plot the arrows
plotter = pv.Plotter()

unit=1

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



# Generate mesh
mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    wells=[(20,2,124.5,20,2,100,40,2,100)],
    sources=[(70,2,80)],
    centers=[(0,4,70)],
    axes=[(100,0,0)],
    radii=[3],
    extra_planes=[],
    tolerance=0.01,
    mesh_size=0.5,
    curve_mesh_size=0.5,
    DISTANCE_THRESHOLD = 3,
    PROJECTION_THRESHOLD = 3.5,
    EXTRUSION_FACTOR = 5.5,
    z_threshold = 0.1,
    extent=[1, 89, 1, 9, 60, 124.5]
)

#%%
mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")


