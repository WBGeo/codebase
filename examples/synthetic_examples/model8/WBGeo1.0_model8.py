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
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement,EllipseRefinement,
   LinearSourceRefinement,FunctionSourceRefinement, TriangulationRefinement, FaultRefinement)
#%%

cwd = os.getcwd()

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

# Compute 3D meshes based on the structural model result using different meshing approaches.
# Meshing parameters not yet verified for this model.

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
#mesh_explicit_unstructured = create_unstructured_mesh_data(
#    geomodel_result=structural_model_result,
#    tolerance=50,
#    mesh_size=20,
#    curve_mesh_size=5,
#    DISTANCE_THRESHOLD = 40,
#    PROJECTION_THRESHOLD = 60,
#    EXTRUSION_FACTOR = 80,
#    z_threshold = 10,
# )

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


# Optional: Example of how to include objects (only works for unstructured mesh)

# # Load engineering objects
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model8/input_data/engineering_objects/model_8_ellipses.csv")
# csv_files=(cwd + "/examples/synthetic_examples/Model8/input_data/Engineering_objects/seismic_plane_new_offset_0.csv",
#           cwd + "/examples/synthetic_examples/Model8/input_data/Engineering_objects/seismic_plane_new_offset_1.csv")
# triangulations= load_triangulations_planes_from_csv(csv_files)

# Refinement of mesh
# refinement = Refinement()

# Linear refinement near wells
# refinement.wells=LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# Function-based refinement near sources
# refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")

# Refinment around triangulated surfaces
# refinement.triangulation = TriangulationRefinement(
#    hmin=8.0,
#    hmax=90.0,
#    d1=50.0,
#    d2=100.0,
#    enabled=True
# )

# NOTE: Wells and sources support both linear and function-based refinement:
#   - LinearWellRefinement / LinearSourceRefinement
#   - FunctionWellRefinement / FunctionSourceRefinement
#
# Ellipse and fault refinement follow the same configuration pattern as
# TriangulationRefinement, using hmin/hmax for mesh sizes and d1/d2 for
# distance-based refinement control.

# # Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     extra_planes=planes,
#     tolerance=50,
#     mesh_size=70,
#     curve_mesh_size=8,
#     DISTANCE_THRESHOLD=40,
#     PROJECTION_THRESHOLD=60,
#     EXTRUSION_FACTOR=80,
#     z_threshold=10,
#     gmsh_flag= True,  # to save original gmsh configuration (defaut is False)
#     mapping_litho='manual', # it can be 'manual', 'auto' or 'none' (default: auto)
#     merge_file = cwd + "/examples/synthetic_examples/model8/input_data/block_groups.csv", # Only if mapping_litho='manual'
#     refinement=refinement,
# )
# save the mesh
# buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type='unstr')
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())
#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)


