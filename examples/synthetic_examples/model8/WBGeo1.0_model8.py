# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Importing necessary libraries
import pandas as pd
import os

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
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)

#%%

cwd = os.getcwd()


# ==============================================================================
# MODEL 8 - SCRIPT MAP
# ==============================================================================
# SECTION 1: Structural geological model
# SECTION 2: Mesh generation and mesh utilities
#
# Each section starts and ends with a large banner so the different sections are
# visually separated while keeping the existing Model 8 workflow unchanged.
# ==============================================================================

#%%

# ==============================================================================
# SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
# Build the two-fault model, three stratigraphic groups with two unconformities, fault activity, and the structural model result.
# ------------------------------------------------------------------------------

# WORKFLOW Synthetic Model 8: 2 faults, 2 unconformities, 3 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),
    resolution=(100, 50, 50)
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_8',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
                                      )

#%%

# Plot fault input data without model context
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

#%%

# Plot input data without model context
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
frame.set_fault_activity_by_group(fault_name="fault1", group_name="Mid")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="Bot")

frame.fault_activity_verbose

#%%

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

# Optional: Plotting age masks and scalar fields
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


#%%

# ------------------------------------------------------------------------------
# END SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================

#%%

# ==============================================================================
# SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================
# Build and visualize the supported mesh representation for Model 8 and document current meshing limitations.
# ------------------------------------------------------------------------------

# Compute 3D meshes based on the structural model result using different meshing approaches.
# Meshing parameters not yet verified for this model.

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
# # NOTE: Currently, the unstructured mesh does not support models
# with faults where different layers on both sides of the fault lie on nearly same surface.

# Explicit structured mesh
# NOTE: currently structured mesh does not support models with faults

#%%

# Plot the meshing results
#plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)

#%%


# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured)
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())




# ------------------------------------------------------------------------------
# END SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================
