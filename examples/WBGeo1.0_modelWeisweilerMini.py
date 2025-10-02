# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.interpolator_components.ordinary_kriging import ordinary_kriging_interpolator
from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.interpolator_components.geo_inr import geo_inr_interpolator
from core.interpolator_components.loopstructural import loop_structural_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model Weisweiler Mini: 1 stratigraphic series

# Component 1: Input data
data_test = InputData(name='WeisweilerMini',
                      extent=np.array([5623500, 5640000, 32304500, 32305500, -3000, 500]),
                      resolution=np.array([250, 20, 125]),
                      surface_points=pd.read_csv(
                          cwd + "/examples/data/modelWeisweilerMini_surface_points_df.csv"),
                      orientations=pd.read_csv(
                          cwd + "/examples/data/modelWeisweilerMini_orientations_df.csv"),
                      mapping_object={
                          "Strat_Series1": ('BreitgangFM','KrebsTraufeFM', 'WilhelmineFM',
                                'ObererKohlenkalkGP','MittlererKohlenkalkGP', 'CondrozGP')},
                      faults=[False]
                      )

#%%

# 1.5: Plot the input data (2D and 3D possible) - Should be an option of the input data component
plot_2d(data_test)
plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
# results_test = universal_cokriging_interpolator(data_test)
results_test = ordinary_kriging_interpolator(data_test, var_range=11000, anisotropy_scaling_z=0.3)
# results_test = rbf_interpolator(data_test, kernel='multiquadric', epsilon=0.00013)
# results_test = geo_inr_interpolator(data_test, beta=5) # TODO: Find reasonable parameters for INR
# results_test = loop_structural_interpolator(data_test, interpolator_type="FDI")

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True, show_data=False)
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True)


#%%

# 4: Meshing for Process Simulation
# TODO: Throws memory error on my machine
mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    tolerance=300,
    mesh_size=50,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10,
    extent=[5623500, 5640000, 32304500, 32305500, -3000, 450]
)

#%%

# 4.5: Plot the meshing result (only 3D at current state)
plot_mesh_3d(mesh_test, data_test)

#%%

# Bonus: Create a mesh with wells
mesh_test = create_unstructured_mesh_data(
    geomodel_result=results_test,
    wells=[(5624000,32305000,-3000,15624000,32305000,-2000)],
    sources=[],
    centers=[],
    axes=[],
    radii=[],
    extra_planes=[],
    tolerance=300,
    mesh_size=50,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10,
    extent=[5623500, 5640000, 32304500, 32305500, -3000, 450]
)

#%%

# Bonus: Export mesh to VTM, Exodus, and VTU formats
# mesh_test.export_vtm('file.vtm')
# mesh_ex=mesh_test.export_exodus("filename.exo")
# mesh_vtu=mesh_test.export_vtu("filename.vtu")






