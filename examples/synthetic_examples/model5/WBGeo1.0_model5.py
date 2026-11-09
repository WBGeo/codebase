# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Importing necessary libraries
import pandas as pd
import os
from core.simulation_components.simulation_packages.golem.simulation_run import run_golem_with_image

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D, plot_structural_model_3D, plot_input_data_3D)

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
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials)
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results
from core.simulation_components.output_format.exodus.unified_format_exodus import load_exodus_results


from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,ExtraPlaneRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement, LinearShaftRefinement, FunctionShaftRefinement, GeneralRefinement)
#%%

cwd = os.getcwd()


# ==============================================================================
# MODEL 5 - SCRIPT MAP
# ==============================================================================
# SECTION 1: Structural geological model
# SECTION 2: Mesh generation and mesh utilities
#
# Each section starts and ends with a large banner so the different sections are
# visually separated while keeping the existing Model 5 workflow unchanged.
# ==============================================================================

#%%

# ==============================================================================
# SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
# Build the unconformable stratigraphy, structural frame, and structural model result.
# ------------------------------------------------------------------------------

# WORKFLOW Synthetic Model 5: no faults, 1 unconformity, 2 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),
    resolution=(100, 50, 50)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_5',
                                             mapping_object={
                                                 "Strat_Series2": ('rock4', 'rock3'),
                                                 "Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model5/input_data/geological_data/model5_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model5/input_data/geological_data/model5_orientations_df.csv")
                                             )

#%%

# Plot input data without model context
plot_input_data_3D(data_elements)

#%%

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       )

frame.detailed_report()

#%%

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Set another interpolation method per group
# frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")
# frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()
# frame["Strat_Series2"].configure_interpolation_params()

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
# Build and visualize the implicit structured, explicit unstructured, and explicit structured meshes.
# ------------------------------------------------------------------------------

# Compute 3D meshes based on the structural model result using different meshing approaches.

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)
buf = export_mesh_results_to_exodus(mesh_implicit_structured, type='imp')
with open("filename_example_imp__mesh.exo", "wb") as f:
    f.write(buf.getvalue())
# Explicit unstructured mesh
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel=structural_model_result,
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=10,
    DISTANCE_THRESHOLD=50,
    PROJECTION_THRESHOLD=60,
    EXTRUSION_FACTOR=120,
    z_threshold=10,
)

# Explicit structured mesh
mesh_explicit_structured = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[25, 21, 16, 5, 6],
    z_threshold=0.1,
    tolerance=1,
)

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_structured, structural_model_result, show_plotter=True)
#%%


# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured, type='unstr')
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())

#%%

# Optional: Example of how to include objects (only works for unstructured mesh)

# # Load engineering objects
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_ellipses.csv")


# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
# refinement = Refinement()
# refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.shafts = LinearShaftRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
# refinement.ellipses = EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
# refinement.extra_planes = ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0)
# refinement.general = [GeneralRefinement(coordinates=[(500.0, 400.0, 0.0),],SizeMin=1.0,DistMax=70.0,),
#    GeneralRefinement(coordinates=[(0.0, 0.0, 0.0),(1000.0, 0.0, 0.0),],expression="2 + 85*(1 - exp(-DIST/100))",),]

# Available refinement types per object:
#
#   wells / sources / shafts:             LinearWellRefinement, FunctionWellRefinement
#                                         LinearSourceRefinement, FunctionSourceRefinement
#                                         LinearShaftRefinement, FunctionShaftRefinement
#
#       Linear refinement:                SizeMin and SizeMax = element sizes
#                                         DistMin and DistMax = transition distances
#
#       Function refinement:              expression = a function using the DIST variable
#
#   faults / ellipses / triangulations:   FaultRefinement, EllipseRefinement, TriangulationRefinement
#   extra planes:                         ExtraPlaneRefinement
#
#                                         SizeMin and SizeMax = element sizes
#                                         DistMin and DistMax = transition distances
#
#   general point / line refinement:      GeneralRefinement
#
#                                         One coordinate = point refinement
#                                         Two coordinates = line-segment refinement
#
#       Linear refinement:                SizeMin = minimum element size at the point or line
#                                         DistMax = distance over which the element size
#                                         increases linearly from SizeMin to mesh_size
#
#       Function refinement:              expression = a function using the DIST variable



# The same configuration can be created in one Refinement instance like below:
#refinement = Refinement(
#    wells=LinearWellRefinement( SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0,),
#    shafts=LinearShaftRefinement( SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0,),
#    sources=FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))",),
#    ellipses=EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0,),
#    extra_planes=ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0,),
#    general=[
#        GeneralRefinement(coordinates=[(500.0, 400.0, 0.0),], SizeMin=4.0, DistMax=300.0,),
#        GeneralRefinement(coordinates=[(0.0, 0.0, 0.0),(1000.0, 0.0, 0.0),], expression="2 + 85*(1 - exp(-DIST/100))",),],)


# # Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel=structural_model_result,
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     extra_planes=planes,
#     tolerance=50,
#     mesh_size=20,
#     curve_mesh_size=10,
#     DISTANCE_THRESHOLD=50,
#     PROJECTION_THRESHOLD=60,
#     EXTRUSION_FACTOR=120,
#     z_threshold=10,
#     gmsh_flag= True,   # to save original gmsh configuration (defaut is False)
#     mapping_litho='automatic_centers', # it can be 'manual', 'automatic_centers', 'automatic_corners' or 'none' (default: auto)
#     refinement=refinement,
# )
#
# save the mesh
# buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type='unstr')
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())

#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%
# ------------------------------------------------------------------------------
# END SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================
