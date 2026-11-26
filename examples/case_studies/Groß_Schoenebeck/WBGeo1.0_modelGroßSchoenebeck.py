from pathlib import Path
# Importing necessary libraries
import pandas as pd
import numpy as np
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)
from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
    load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv,
    load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv)
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemCustomBuilder,
)

from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials, plot_variable_time_series)
#%%

cwd = os.getcwd()


# ==============================================================================
# GROSS SCHOENEBECK CASE STUDY - SCRIPT MAP
# ==============================================================================
# SECTION 1: Structural geological model
# SECTION 2: Mesh generation and mesh utilities
# SECTION 3: Groß Schönebeck GOLEM simulation
#     ├── Steady-state simulation
#     └── Transient simulation

#
# Each section starts and ends with a large banner so the workflow is easier to
# navigate while keeping the existing Groß Schönebeck model logic unchanged.
# ==============================================================================

#%%

# ==============================================================================
# SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
# Load the Groß Schönebeck geological data, build the seven stratigraphic series, configure interpolation, and compute the structural model result.
# ------------------------------------------------------------------------------

# WORKFLOW Groß Schoenebeck Case Study: 7-layer structural model

#%%

# Load input data
surface_points = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_surface_pointsdownsampled_full.csv")
orientations = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_orientations_downsampled_full.csv")

surface_points_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_faults_surface_points_downsampled.csv")
orientations_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_faults_orientations_downsampled.csv")

print(len(surface_points), len(orientations))
print(len(surface_points_faults), len(orientations_faults))

#%%

# Create a grid for the model
margin = 100  # Add a margin of 100 units around the data
grid = RegularGrid(
    extent=(surface_points["X"].min()-margin, surface_points["X"].max()+margin,
            surface_points["Y"].min()-margin, surface_points["Y"].max()+margin,
            surface_points["Z"].min()-margin, surface_points["Z"].max()+margin),
    resolution=(50, 50, 100)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='GSB',
                                             mapping_object={
                                                "01": ('01_top_hannover'),
                                                "02": ('02_top_dethlingen'),
                                                "03": ('03_top_ebs'),
                                                "04": ('04_top_rockel'),
                                                "05": ('05_top_havel'),
                                                "06": ('06_top_vulkanit'),
                                                "07": ('07_top_karbon')
                                             },
                                             surface_points=surface_points,
                                             orientations=orientations
                                             )

#%%

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid
                                       )

frame.detailed_report()

#%%

# Plot the input data (2D and 3D possible)
#plot_structural_model_2D(frame)
#plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series
frame["01"].set_interpolation_method("Universal Co-Kriging")
frame["02"].set_interpolation_method("Universal Co-Kriging")
frame["03"].set_interpolation_method("Universal Co-Kriging")
frame["04"].set_interpolation_method("Universal Co-Kriging")
frame["05"].set_interpolation_method("Universal Co-Kriging")
frame["06"].set_interpolation_method("Universal Co-Kriging")
frame["07"].set_interpolation_method("Universal Co-Kriging")

frame.detailed_report()

#%%

# Compute structural model result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# Plot the results (2D and 3D possible)
#plot_structural_model_2D(structural_model_result.structural_frame)
#plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

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
# Build the explicit structured and implicit structured meshes, retain the optional unstructured-mesh example.
# ------------------------------------------------------------------------------

# Compute 3D meshes based on the structural model result using different meshing approaches.

# Explicit structured mesh
mesh_str = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[10, 10, 5, 5, 6, 7, 9, 10],
    mesh_division=(100, 100),
    z_threshold=0.1,
    tolerance=0.5
)

# Implicit structured mesh
mesh_implicit = create_implicit_structured_mesh(geomodel_result=structural_model_result)


# Explicit unstructured mesh
#mesh_explicit_unstr = create_unstructured_mesh_data(
#    geomodel_result=structural_model_result,
#    tolerance=500,
#    mesh_size=60,
#    smooth=30,
#    curve_mesh_size=5,
#    extent=[401377.0, 409314.0, 5859433.0, 5865390.0, -3678.0, -4453.0],
#    mapping_litho='none', # it can be 'manual', 'automatic_centers', 'automatic_corners' or 'none' (default: automatic_centers)

#)

#%%

# Plot the meshing results
#plot_mesh_3d(mesh_implicit, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_explicit_unstr, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_str, structural_model_result, show_plotter=True)

#%%

#%%



#%%


# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
#buf = export_mesh_results_to_exodus(mesh_explicit_unstr, type='unstr')
#with open("filename_Gross_Schoenebeck.exo", "wb") as f:
#    f.write(buf.getvalue())



# # Load engineering objects
wells = load_wells_from_csv(cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/engineering_objects/wells.csv")
csv_files=(cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/engineering_objects/seismic.csv",
           cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/engineering_objects/well.csv",
           cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/engineering_objects/multi.csv")
triangulations= load_triangulations_planes_from_csv(csv_files)

# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
refinement = Refinement()
refinement.wells = LinearWellRefinement(SizeMin=15.0, SizeMax=105.0, DistMin=30.0, DistMax=150.0)
refinement.triangulations = TriangulationRefinement(SizeMin=15.0, SizeMax=105.0, DistMin=30.0, DistMax=150.0)
#
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

# create big model wit traslate to center

mesh_unstructured_with_objects = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    wells=wells,
    triangulations=triangulations,
    mesh_size=75,
    curve_mesh_size=5,
    gmsh_flag=True,
    refinement=refinement,
    rotation_angle=0,

)
buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects , type='unstr')
with open("filename_Gross_Schoenebeck_BM_rotate_0.exo", "wb") as f:
    f.write(buf.getvalue())




# ceate small model with rotation
extent = [
    # Bottom face: z = -4400
    (405000.0353, 5863600.0626, -4320.23),
    (406521.6994, 5863105.5584, -4320.23),
    (406027.2389, 5861583.8632, -4320.23),
    (404505.5390, 5862078.2464, -4320.23),

    # Top face
    (405000.0353, 5863600.0626, -3750.26),
    (406521.6994, 5863105.5584, -3750.26),
    (406027.2389, 5861583.8632, -3750.26),
    (404505.5390, 5862078.2464, -3750.26),
]


mesh_unstructured_with_objects = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    extent=extent,
    wells=wells,
    triangulations=triangulations,
    mesh_size=65,
    curve_mesh_size=5,
    gmsh_flag=True,
    rotation_angle=18,
    refinement=refinement,
)
buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects , type='unstr')
with open("filename_Gross_Schoenebeck_SM_n.exo", "wb") as f:
    f.write(buf.getvalue())



#%%

# ------------------------------------------------------------------------------
# END SECTION 2: MESH GENERATION AND MESH UTILITIES
# ===============================================================================

# ==============================================================================
# SECTION 3: GROSS SCHÖNEBECK GOLEM SIMULATION
# ==============================================================================
# 3A. Steady-state simulation
# ------------------------------------------------------------------------------

# Solving steady state of the problem using custom golem input file.
reservoir_input_path = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/steady_state.i"
)

with open(reservoir_input_path, "r") as f:
    reservoir_input = f.read()

# Solving the problem with Golem
model_GS_Xindyu_transient_results = run_simulation_golem(
    problem=GolemCustomBuilder(
        input_file_contents=reservoir_input,
        mesh_results=mesh_unstructured_with_objects,
        geomodel_result=structural_model_result,
        print_linear_residuals=True,
    ),
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
    output_dir=(
        cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "GS_steady_state"
    ),
    nproc=4,
    stream_output=True,  # Show live GOLEM output (opt-in)
)

# ------------------------------------------------------------------------------
# 3B. Transient simulation
# ------------------------------------------------------------------------------
# Solving transient state of the problem using custom golem input file.

reservoir_input_trans_path = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/transient_state.i"
)

with open(reservoir_input_trans_path, "r") as f:
    reservoir_input_trans = f.read()

# Auxiliary CSV/TXT files referenced by transient_circ.i as files/<name>.
reservoir_auxiliary_files_dir = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/files"
)

# Solving the problem with Golem
model_GS_Xindyu_sready_results = run_simulation_golem(
    problem=GolemCustomBuilder(
        input_file_contents=reservoir_input_trans,
        input_file_path=reservoir_input_trans_path,
        auxiliary_files_dir=reservoir_auxiliary_files_dir,
        mesh_file=cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "GS_steady_state/wbgeo_out.e",
        geomodel_result=structural_model_result,
        print_linear_residuals=True,
    ),
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
    output_dir=(
        cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "GS_transient_state"
    ),
    nproc=3,
    stream_output=True,  # Show live GOLEM output (opt-in)
)




#%%

