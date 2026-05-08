# Importing necessary libraries
import pandas as pd
import os
# from dotenv import load_dotenv
# load_dotenv()

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D, plot_fault_model_2D,
    plot_fault_input_data_3D, plot_input_data_3D)

from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
    load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.mesh_format.vtu.VTU_format import export_mesh_results_to_vtu
from core.meshing_components.mesh_format.vtk.VTK_format import export_mesh_results_to_vtk
from core.meshing_components.mesh_format.feflow.Feflow_format import export_mesh_results_to_feflow
from core.meshing_components.mesh_format.gmsh.GMSH_format import export_mesh_results_to_gmsh
from core.meshing_components.mesh_format.stl.STL_format import export_mesh_results_to_stl
from core.meshing_components.mesh_format.vtm.VTM_format import export_mesh_results_to_vtm
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
from core.meshing_components.mesh_format.abaqus.Abaqus_format import export_mesh_results_to_abaqus

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 9: 5 faults, 3 unconformities, 4 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1500, 0, 1000, 0, 1000),
    resolution=(75, 50, 50)
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_9',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model9/input_data/geological_data/model9_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model9/input_data/geological_data/model9_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2', 'fault3', 'fault4', 'fault5']
                                      )

#%%

plot_fault_input_data_3D(data_faults)

#%%

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid
)

fault_frame.detailed_report()

#%%

# Plot the fault input data (2D and 3D possible)
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
data_elements = InputData_StructuralElements(name='Model_9',
                                             mapping_object={
                                                "A": ('rock1', 'rock2'),
                                                "B": ('rock3', 'rock4'),
                                                "C": ('rock5', 'rock6'),
                                                "D": ('rock7', 'rock8'),
                                                },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model9/input_data/geological_data/model9_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model9/input_data/geological_data/model9_orientations_df.csv")
                                             )

#%%

plot_input_data_3D(data_elements)

#%%

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_model_results=fault_model_result
                                       )

frame.detailed_report()

#%%

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set fault activity per stratigraphic group
frame.set_fault_activity_by_group(fault_name="fault1", group_name="C")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="D")
frame.set_fault_activity_by_group(fault_name="fault3", group_name="B")
frame.set_fault_activity_by_group(fault_name="fault4", group_name="C")
frame.set_fault_activity_by_group(fault_name="fault5", group_name="D")

frame.fault_activity_verbose

#%%

# Set another interpolation method per group
# frame["A"].set_interpolation_method("GeoINR")
# frame["B"].set_interpolation_method("GeoINR")
# frame["C"].set_interpolation_method("GeoINR")
# frame["D"].set_interpolation_method("GeoINR")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["A"].configure_interpolation_params()
# frame["B"].configure_interpolation_params()
# frame["C"].configure_interpolation_params()
# frame["D"].configure_interpolation_params()

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

# Optional: Plotting age masks and scalar fields
# frame.plot_scalar_field_section(group_nr=3, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=2, axis='y', index=12)

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

# Compute 3D meshes based on the structural model result using different meshing approaches.
# Meshing parameters not yet verified for this model.

# Implicit structured mesh
# mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
# mesh_unstructured = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     tolerance=50,
#     mesh_size=30,
#     curve_mesh_size=5,
#     DISTANCE_THRESHOLD=40,
#     PROJECTION_THRESHOLD=60,
#     EXTRUSION_FACTOR=80,
#     z_threshold=2
# )

# Explicit structured mesh
# mesh_explicit_structured = create_structured_mesh_data(geomodel_result=structural_model_result,
#                                                        refinement_data=[10, 10, 10])

#%%

# Plot the meshing results
# plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
# plot_mesh_3d(mesh_unstructured, structural_model_result, show_plotter=True)
# plot_mesh_3d(mesh_explicit_structured, structural_model_result, show_plotter=True)

#%%

# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# buf = export_mesh_results_to_exodus(mesh_unstructured)
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())
