"""Synthetic model 8: 2 faults, 2 unconformities, 3 stratigraphic groups.

Run it cell by cell (#%%) from the repository root; each main section starts with a
banner, each sub-section with a numbered cell title.

Sections (same order as the Workbench components)
  1. Structural modeling: fault model, input data, structural frame, structural model
  2. Meshing: the meshes this model supports, plots, export and engineering objects (optional)
"""

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
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, load_wells_from_csv
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)

cwd = os.getcwd()

# ==============================================================================
#%% 1. Structural modeling
# ==============================================================================

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),
    resolution=(100, 50, 50)
)

#%% 1.1 Fault input data

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_8',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model8/input_data/geological_data/model8_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
                                      )

# Plot fault input data without model context
plot_fault_input_data_3D(data_faults)

#%% 1.2 Fault frame

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid
)

fault_frame.detailed_report()

# Plot the fault input data (2D and 3D possible)
plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%% 1.3 Fault model

# Compute fault model result
fault_model_result = general_faults.compute_fault_domains(fault_frame)

# Plot the fault model results (2D and 3D possible)
plot_fault_model_2D(fault_model_result.fault_frame)
plot_fault_model_3D(fault_model_result.fault_frame)

#%% 1.4 Structural input data

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

# Plot input data without model context
plot_input_data_3D(data_elements)

#%% 1.5 Structural frame

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_model_results=fault_model_result
                                       )

frame.detailed_report()

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%% 1.6 Fault activity per group

# Set fault activity per stratigraphic group
frame.set_fault_activity_by_group(fault_name="fault1", group_name="Mid")
frame.set_fault_activity_by_group(fault_name="fault2", group_name="Bot")

frame.fault_activity_verbose

#%% 1.7 Interpolation settings (optional)

# Set another interpolation method per group
# frame["Top"].set_interpolation_method("Universal Co-Kriging")
# frame["Mid"].set_interpolation_method("Universal Co-Kriging")
# frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Top"].configure_interpolation_params()
# frame["Mid"].configure_interpolation_params()
# frame["Bot"].configure_interpolation_params()

# frame.detailed_report()

#%% 1.8 Structural model

# Compute structural model result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

# Plot the results (2D and 3D possible)
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

# Optional: Plotting age masks and scalar fields
# frame.plot_scalar_field_section(group_nr=0, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)

#%% 1.9 Surface gradients (optional)

# Optional: Compute gradients at the surface mesh vertices
# from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
#
# gradients_dict, gradients_faults_dict = surface_mesh_gradients.get_surface_mesh_gradients(structural_model_result,
#                                                                                           mesh_type="unmasked")
# surface_mesh_gradients.plot_surface_mesh_gradients(structural_model_result,
#                                                    gradients_dict,
#                                                    gradients_faults_dict,
#                                                    mesh_type="unmasked")

# ==============================================================================
#%% 2. Meshing
# ==============================================================================

# Compute 3D meshes based on the structural model result using different meshing approaches

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=50,
    mesh_size=30,
    curve_mesh_size=10,
)

# Explicit structured mesh
# Not available here: the structured mesh doesn't support models with faults yet.

#%% 2.1 Plot the meshes

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)

#%% 2.2 Export a mesh (optional)

# Optional: Export a mesh, here the unstructured mesh to Exodus.
# The other exporters work the same way (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# Only the Exodus exporter needs the mesh type ('imp', 'str' or 'unstr'), and some formats
# support only some mesh types; see the meshing manual.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured, type='unstr')
# with open("model8_unstructured_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#%% 2.3 Engineering objects (optional)

# Optional: Add engineering objects to the unstructured mesh (only works for unstructured meshes).
# This model's wells, sources, shafts, planes and ellipses are in input_data/engineering_objects/.
# Models 1 and 2 show the full workflow, including all refinement options around the objects.
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_wells.csv")
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     wells=wells,
#     tolerance=50,
#     mesh_size=30,
#     curve_mesh_size=10,
# )
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)
