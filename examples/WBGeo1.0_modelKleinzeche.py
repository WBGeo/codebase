# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.object_components import InputData
from core.interpolator_components.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d, plot_mesh_3d

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model Weisweiler Mini: 1 stratigraphic series

# Component 1: Input data
data_test = InputData(name='Kleinzeche',
                      extent=np.array([0, 90, 0, 10, 55, 125]),
                      resolution=np.array([180, 10, 140]),
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
# plot_3d(data_test)


#%%

# Component 2 --> Component 3: Interpolation to geomodel result
results_test = universal_cokriging_interpolator(data_test)

#%%

# 3.5: Plot the results (2D and 3D possible) - Should be an option of the results component
plot_2d(input_data=data_test, geomodel_results=results_test, show_results=True, show_data=True,
        colors = ['#959595', '#000000', '#00c401', '#828701', '#ec7a10', '#828701',
                  '#ec7a10','#828701', '#ec7a10', "#a619e7"])

#%%
# TODO: Meshes shifted in z-direction, why?
plot_3d(input_data=data_test, geomodel_results=results_test, show_results=True, surface_type="unmasked",
        colors = ['#959595', '#000000', '#00c401', '#828701', '#ec7a10', '#828701',
                  '#ec7a10','#828701', '#ec7a10', "#a619e7"])




# Geberate mesh
mesh_test = create_unstructured_mesh_data(
    data_test= data_test,
    geomodel_result=results_test,
    num_wells=2,
    wells=[(100,100,100,100,100,500, 300,100,500,300,100,300), (500,500,500,500,500,900)],
    num_sources=2,
    sources=[(100,300,500), (400,600,700)],
    num_shafts=2,
    centers=[(200,500,400), (100,200,700)],
    axes=[(1000,0,0), (1000,0,0)],
    radii=[30, 20],
    num_planes=2,
    extra_planes=[(0,0,400,1000,0,400,1000,1000,400,0,1000,400), (0,0,600,1000,0,600,1000,1000,600,0,1000,600)],
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD = 60,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 80,
    z_threshold = 10
)


mesh_test.export_vtm('file.vtm')
print('doneeeeee')
mesh_ex=mesh_test.export_exodus("filename.exo")
mesh_vtu=mesh_test.export_vtu("filename.vtu")


