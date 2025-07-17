# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 2: no faults, no unconformities, 2 stratigraphic series

# Component 1: input data
data_test = InputData(name='Model 2',
                      extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                      resolution=np.array([50, 50, 50]),
                      mapping_object={"Strat_Series": ('rock2', 'rock1')},
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model2_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model2_orientations_df.csv"),
                      faults=[False]
                      )

#%%

# Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
results_test = ordinary_kriging_interpolator(data_test)
# results_test = rbf_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="masked")

#%%

# 4: Meshing for Process Simulation
# mesh_test = create_structured_mesh_data(
#     geomodel_result=results_test,
#     refinement_data=[10,10,10],
#     z_threshold=0.1,
#     tolerance=1
# )




#%%




# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100,100,100,100,100,500, 300,100,500,300,100,300), (500,500,500,500,500,900)],
    num_sources=2,
    sources=[(100,300,500), (400,600,700)],
    num_shafts=1,
    centers=[(0,0,700)],
    axes=[(1000,0,0)],
    radii=[20],
    num_planes=1,
    extra_planes=[(0, 0, 100, 1000, 0,100, 1000,1000,100, 0,1000,100)],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=2,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)


mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")




