# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.loading_components.geo_input_data import load_mapping


from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 6: no faults, 2 unconformities, 3 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 100, 100)  # Example resolution
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_6',
                                             mapping_object=load_mapping(cwd + "/examples/synthetic_examples/model6/input_data/geological_data/model6_mapping.json"),
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model6/input_data/geological_data/model6_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model6/input_data/geological_data/model6_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid
                                       )

frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
# frame["Shallow_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Medium_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Deep_Strat"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Shallow_Strat"].set_interpolation_parameters()
# frame["Medium_Strat"].set_interpolation_parameters()
# frame["Deep_Strat"].set_interpolation_parameters()

# frame.detailed_report()

#%%

# Compute structural model result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# Plot the results (2D and 3D possible)
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)


#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=0, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)

#%%

# Optional: Compute gradients at the surface mesh vertices
# from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
#
# gradients_dict, gradients_faults_dict = surface_mesh_gradients.get_surface_mesh_gradients(structural_model_result,
#                                                                                           mesh_type="unmasked")
# surface_mesh_gradients.plot_surface_mesh_gradients(structural_model_result,
#                                                    gradients_dict,
#                                                    gradients_faults_dict,
#                                                    mesh_type="unmasked")

#%%

###########################################################################################################################
#                                              Meshing
###########################################################################################################################

#################################
# ImplicitStructured meshing.   #
#################################
mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)
#################################
# Explicit Unstructured meshing #
mesh_unstr = create_unstructured_mesh_data(
     geomodel_result=structural_model_result,
     tolerance=10,
     mesh_size=10,
     curve_mesh_size=5,
     extent=(0,1000,0,1000,20,980)
 )


#########################
# Export mesh to exodus #
#########################

# Implicit structured mesh
#buf = export_mesh_results_to_exodus(mesh_implicit)
#with open("filename_implic2.exo", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_exodus(mesh_unstr)
#with open("filename.exo", "wb") as f:
#    f.write(buf.getvalue())

