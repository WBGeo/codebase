# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.object_components import InputData_StructuralElements
from concepts.archive.universal_cokriging import universal_cokriging_interpolator
from core.visualization_components import plot_2d, plot_3d

#%%

cwd = os.getcwd()

#%%
# WORKFLOW Model Weisweiler Mini: 1 stratigraphic series

# Component 1: Input input_data
data_test = InputData_StructuralElements(name='Kleinzeche',
                                         extent=np.array([0, 90, 0, 10, 55, 125]),
                                         resolution=np.array([180, 10, 140]),
                                         surface_points=pd.read_csv(
                          cwd + "/examples/input_data/modelKleinzeche_surface_points_df.csv"),
                                         orientations=pd.read_csv(
                          cwd + "/examples/input_data/modelKleinzeche_orientations_df.csv"),
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

# 1.5: Plot the input input_data (2D and 3D possible) - Should be an option of the input input_data component
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



