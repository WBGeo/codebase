# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.mesh_format.exudos.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.mesh_format.vtu.VTU_format import export_mesh_results_to_vtu
from core.meshing_components.mesh_format.vtk.VTK_format import export_mesh_results_to_vtk
from core.meshing_components.mesh_format.feflow.Feflow_format import export_mesh_results_to_feflow
from core.meshing_components.mesh_format.gmsh.GMSH_format import export_mesh_results_to_gmsh
from core.meshing_components.mesh_format.stl.STL_format import export_mesh_results_to_stl
from core.meshing_components.mesh_format.vtm.VTM_format import export_mesh_results_to_vtm
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 8: 2 faults, 2 unconformities, 3 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 50, 50)  # Example resolution
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_9',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
                                      )

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid
)

fault_frame.detailed_report()

#%%

# Plot the fault input input_data (2D and 3D possible)
plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%%

# Compute fault model result
fault_model_result = general_faults.compute_fault_domains(fault_frame)

#%%

# Plot the fault model results (2D and 3D possible)
plot_fault_model_2D(fault_model_result.fault_frame)
plot_fault_model_3D(fault_model_result.fault_frame)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_8',
                                             mapping_object={
                                                "Top": ('rock6', 'rock5'),
                                                "Mid": ('rock4', 'rock3'),
                                                "Bot": ('rock2', 'rock1'),
                                                },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_orientations_df.csv")
                                             )

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_model_results=fault_model_result
                                       )

frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# set fault activity verbose
frame.set_fault_activity_by_group(fault_name="fault1", group_name="Mid")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="Bot")

frame.fault_activity_verbose

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
# frame["Top"].set_interpolation_method("Universal Co-Kriging")
# frame["Mid"].set_interpolation_method("Universal Co-Kriging")
# frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Top"].configure_interpolation_params()
# frame["Mid"].configure_interpolation_params()
# frame["Bot"].configure_interpolation_params()

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

# TODO: Meshing needs to be adapted to work with the new Structural Modeling output

mesh_unstr = create_unstructured_mesh_data(
     geomodel_result=structural_model_result,
     tolerance=50,
     mesh_size=30,
     curve_mesh_size=5,
     DISTANCE_THRESHOLD = 40,
     PROJECTION_THRESHOLD = 60,
     EXTRUSION_FACTOR = 80,
     z_threshold = 10,
     extent=(20,1980,20,980,20,980)

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
