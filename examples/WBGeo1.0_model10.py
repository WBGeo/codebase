# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from core.object_components import InputData, GeomodelResults
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from dotenv import load_dotenv
from core.liquidEarth.le_push_data import push_geosolution_to_le
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

mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    wells=[(100,100,100,100,100,500), (500,500,500,500,500,900)],
    sources=[(900,300,900), (400,600,700)],
    centers=[(200,500,800)],
    axes=[(1000,0,0)],
    radii=[20],
    extra_planes=[(0,0,400,1000,0,400,1000,1000,400,0,1000,400)],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=10,
    DISTANCE_THRESHOLD = 40,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)

#%%

# 4.5: Plot the mesh (2D and 3D possible) - Should be an option of the mesh component
plot_mesh_3d(mesh_test, data_test, style="surface")

#%%

# Bonus: Export mesh to VTM, Exodus, and VTU formats
# mesh_test.export_vtm('file.vtm')
# mesh_ex=mesh_test.export_exodus("filename.exo")
# mesh_vtu=mesh_test.export_vtu("filename.vtu")
# mesh_aba=mesh_test.export_abaqus("filename.inp")
# mesh_feflow=mesh_test.export_feflow("filename.fem")


#%%

# Bonus: liquid earth
load_dotenv(cwd + "/.env") # load .env file
res = push_geosolution_to_le(geosolution=results_test, space_name='WBGeo: Demo')
print(res)

