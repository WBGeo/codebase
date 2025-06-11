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
# WORKFLOW Model 12: 1 unconformity, 2 stratigraphic series
# Component 1: Input data
data_test = InputData(name='Model_12',
                      extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                      resolution=np.array([100, 50, 50]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/model12_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/model12_orientations_df.csv"),
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
# results_test = universal_cokriging_interpolator(data_test) # TODO: Missing basement when generating unstructured mesh
# results_test = ordinary_kriging_interpolator(data_test, var_range=500)
# results_test = rbf_interpolator(data_test, kernel='cubic', epsilon=1)
# results_test = geo_inr_interpolator(data_test)
results_test = loop_structural_interpolator(data_test, interpolator_type="FDI")

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True, direction="y")
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="masked", show_plotter=True)

#%%

# 4: Meshing for Process Simulation
# mesh_test = create_structured_mesh_data(
#    geomodel_result=results_test,
#    refinement_data=[25, 21, 16, 5, 6],
#    z_threshold=0.1,
#    tolerance=1
# )

mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    mesh_size=20
)

#%%

# 4.5: Plot the meshing result (only 3D at current state)
plot_mesh_3d(mesh_test, data_test, style="surface")


#%%

# mesh_test.export_exodus(cwd + '/examples/meshes/Model_structured.exo')
#
# mesh_test.export_vtm(cwd + '/examples/meshes/Model_structured.vtm')
#
# mesh_test.export_vtk(cwd + '/examples/meshes/Model_structured.vtk')

