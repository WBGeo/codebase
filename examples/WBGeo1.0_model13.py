# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.geo_inr import geo_inr_interpolator
from core.interpolator_components.loopstructural import loop_structural_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model 13: 2 unconformity, 3 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Model_13',
                      extent=np.array([0, 1000, 0, 500, 0, 1000]),
                      resolution=np.array([100, 50, 100]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model13_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd+"/examples/data/model13_orientations_df.csv"),
                      mapping_object={
                          "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
                          "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
                          "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1')},
                      faults=[False, False,False]
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible)
plot_2d(data_test)
plot_3d(data_test)

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
results_test = ordinary_kriging_interpolator(data_test, var_range=1500)
# results_test = rbf_interpolator(data_test, kernel='multiquadric', epsilon=0.0001)
# results_test = geo_inr_interpolator(data_test, beta=1)
# results_test = loop_structural_interpolator(data_test, interpolator_type="PLI")


#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True, direction="y", slice_int=0)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="masked")


#%%

# 4: Meshing for Process Simulation
# TODO: Does not work properly as layers dont extend to the full extent of the model
# mesh_test = create_structured_mesh_data(
#     geomodel_result=results_test,
#     refinement_data=[10,10,10,10,10,10,10,10,10,10,10],
#     z_threshold=0.1,
#     tolerance=1
# )

# TODO: Mesh creation fails for unstructured mesh creation
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
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

#%%

# 4.5: Plot the meshing result (only 3D at current state)
plot_mesh_3d(mesh_test, data_test)

#%%

# Bonus: Export mesh to VTM, Exodus, and VTU formats
# mesh_test.export_vtm('file.vtm')
# mesh_ex=mesh_test.export_exodus("filename.exo")
# mesh_vtu=mesh_test.export_vtu("filename.vtu")

