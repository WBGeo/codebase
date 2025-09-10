# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 10: 2 faults, 1 unconformity, 2 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model 10',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([125, 50, 50]),
                      mapping_object={
                          "Fault_Series2": ('fault2'),
                          "Strat_Series2": ('rock3'),
                          "Fault_Series1": ('fault1'),
                          "Strat_Series1": ('rock2', 'rock1')},
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model10_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model10_orientations_df.csv"),
                      faults=[True, False, True, False]
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
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)

#%%

from core.visualization_components import inspect_gradients
inspect_gradients(results_test)

#%%
from core.visualization_components import inspect_gradients_wf
inspect_gradients_wf(results_test)

#%%

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



mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100,100,100,100,100,500), (500,500,500,500,500,900)],
    num_sources=2,
    sources=[(900,300,900), (400,600,700)],
    num_shafts=1,
    centers=[(200,500,800)],
    axes=[(1000,0,0)],
    radii=[20],
    num_planes=1,
    extra_planes=[(0,0,400,1000,0,400,1000,1000,400,0,1000,400)],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=10,
    DISTANCE_THRESHOLD = 40,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)

mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")
mesh_aba=mesh_test.export_abaqus("filename.inp")
mesh_feflow=mesh_test.export_feflow("filename.fem")


#%%
from dotenv import load_dotenv
load_dotenv(cwd + "/.env")
print(results_test.name)
from core.liquidEarth.le_push_data import push_geosolution_to_le
res = push_geosolution_to_le(geosolution=results_test, space_name='WBGeo: Demo')
print(res)

