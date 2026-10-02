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
from core.hierarchical_simulation.generate_mesh.hierarchical_mesh import hierarchical_mesh
from core.hierarchical_simulation.export_data.exodus.exo_format import export_meshes_exodus
from core.hierarchical_simulation.visualization.hierarchical_meshing_visualization import plot_hierarchical_meshes_3d
from core.hierarchical_simulation.visualization.hierarchical_paraview_visualization import open_hierarchical_exodus_in_paraview
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
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
# SECTION 3: Hierarchical mesh generation and export

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
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

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
csv_files=(cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/Engineering_objects/seismic.csv",
           cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/Engineering_objects/well.csv",
           cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/Engineering_objects/multi.csv")
triangulations= load_triangulations_planes_from_csv(csv_files)

# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
refinement = Refinement()
refinement.wells = LinearWellRefinement(SizeMin=15.0, SizeMax=65.0, DistMin=30.0, DistMax=150.0)
refinement.triangulations = TriangulationRefinement(SizeMin=15.0, SizeMax=65.0, DistMin=30.0, DistMax=150.0)
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

#mesh_unstructured_with_objects = create_unstructured_mesh_data(
#    geomodel_result=structural_model_result,
#    wells=wells,
#    triangulations=triangulations,
#    mesh_size=75,
#    curve_mesh_size=5,
#    gmsh_flag=True,
#    refinement=refinement,
#    rotation_angle=0,

#)
#buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects , type='unstr')
#with open("filename_Gross_Schoenebeck_BM_rotate_0.exo", "wb") as f:
#    f.write(buf.getvalue())





# ceate small model with rotation
extent = [
    # Bottom face: z = -4400
    (405000.0353, 5863600.0626, -4470.23),
    (406521.6994, 5863105.5584, -4470.23),
    (406027.2389, 5861583.8632, -4470.23),
    (404505.5390, 5862078.2464, -4470.23),

    # Top face
    (405000.0353, 5863600.0626, -3678.26),
    (406521.6994, 5863105.5584, -3678.26),
    (406027.2389, 5861583.8632, -3678.26),
    (404505.5390, 5862078.2464, -3678.26),
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
with open("filename_Gross_Schoenebeck_SM.exo", "wb") as f:
    f.write(buf.getvalue())



# create small model with rotation using csv files (instead of structual modeling results)
#mesh_unstructured_with_objects2 = create_unstructured_mesh_data(
#    surface_csv_dir= cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/surfaces",
#    extent=extent,
#    wells=wells,
#    triangulations=triangulations,
#    mesh_size=65,
#    curve_mesh_size=5,
#    gmsh_flag=True,
#    rotation_angle=18,
#    refinement=refinement,
#)
#buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects2 , type='unstr')
#with open("filename_Gross_Schoenebeck_CSV.exo", "wb") as f:
#    f.write(buf.getvalue())

#plot_mesh_3d(mesh_unstructured_with_objects2, structural_model_result, show_plotter=True)


#%%

# ------------------------------------------------------------------------------
# END SECTION 2: MESH GENERATION AND MESH UTILITIES
# ===============================================================================

# Solving steady state of the problem using custom golem input file.
reservoir_input_path = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/steady_full.i"
)

with open(reservoir_input_path, "r") as f:
    reservoir_input = f.read()

# Solving the problem with Golem
model_GS_Xindyu_transient_results = run_simulation_golem(
    problem=CustomGolemBuilder(
        input_file_contents=reservoir_input,
        mesh_results=mesh_unstructured_with_objects,
        geomodel_result=structural_model_result,
        print_linear_residuals=True,
    ),
    image_name="moose-dev:fixed",
    project_dir="/Users/marziehbaes/Documents/projects",
    docker_dir="/projects",
    output_dir=(
        cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "golem_GS_steady_output_Xinye"
    ),
    nproc=2,
)

# Solving transient state of the problem using custom golem input file.

#reservoir_input_trans_path = (
#    cwd
#    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/transient_circ.i"
#)

#with open(reservoir_input_trans_path, "r") as f:
#    reservoir_input_trans = f.read()

# Auxiliary CSV/TXT files referenced by transient_circ.i as files/<name>.
#reservoir_auxiliary_files_dir = (
#    cwd
#    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/files"
#)

# Solving the problem with Golem
#model_GS_Xindyu_sready_results = run_simulation_golem(
#    problem=CustomGolemBuilder(
#        input_file_contents=reservoir_input_trans,
#        input_file_path=reservoir_input_trans_path,
#        auxiliary_files_dir=reservoir_auxiliary_files_dir,
#        mesh_file=cwd
#        + "/examples/case_studies/Groß_Schoenebeck/"
#          "golem_GS_steady_output_Xinye/wbgeo_out.e",
#        geomodel_result=structural_model_result,
#        print_linear_residuals=True,
#    ),
#    image_name="moose-dev:fixed",
#    project_dir="/Users/marziehbaes/Documents/projects",
#    docker_dir="/projects",
#    output_dir=(
#        cwd
#        + "/examples/case_studies/Groß_Schoenebeck/"
#          "golem_GS_transient_output_Xinye"
#    ),
#    nproc=1,
#)




#%%

# ===============================================================================
# SECTION 3: HIERARCHICAL MESH GENERATION AND EXPORT
# ===============================================================================
# Build two nested Groß Schönebeck meshes. Two equivalent workflows are shown:
#   A) surfaces from the computed structural model;
#   B) surfaces read directly from CSV files.
#
# In both workflows the small model uses the same eight-point domain below and
# its output coordinates are rotated by 18 degrees. The large model covers the
# complete available geological domain.
# -------------------------------------------------------------------------------
# Small hierarchical domain: the same eight corner coordinates used above.
#hierarchical_small_extent = [
    # Bottom face
#    (405000.0353, 5863600.0626, -4470.23),
#    (406521.6994, 5863105.5584, -4470.23),
#    (406027.2389, 5861583.8632, -4470.23),
#    (404505.5390, 5862078.2464, -4470.23),

    # Top face
#    (405000.0353, 5863600.0626, -3678.26),
#    (406521.6994, 5863105.5584, -3678.26),
#    (406027.2389, 5861583.8632, -3678.26),
#    (404505.5390, 5862078.2464, -3678.26),
#]

# -------------------------------------------------------------------------------
# 3A. Hierarchical mesh from the structural model
# -------------------------------------------------------------------------------
# extent1="whole" uses the complete structural-model domain for the large mesh.
# The small mesh is clipped to the eight-point domain and rotated by 18 degrees.

#mesh1_structural, mesh2_structural, nodes_bcs_structural , = hierarchical_mesh(
#    geomodel_result=structural_model_result,
#    mesh_type="Un",
#    extent1="whole",
#    extent2=hierarchical_small_extent,
#    wells=wells,
#    triangulations=triangulations,
#    mesh_size1=85,
#    mesh_size2=35,
#    inside_nodes_csv_dir=cwd +"inside_nodes",

#    unstructured_params={
#        "curve_mesh_size": 5,
#        "refinement": refinement,
#    },

#    unstructured_params2={
#        "curve_mesh_size": 5,
#        "refinement": refinement,
#        "rotation_angle": 18,
#    },
#)

#export_meshes_exodus(
#    mesh1_structural,
#    mesh2_structural,
#    nodes_bcs_structural,
#    "GSB_hierarchical_structural_big.exo",
#    "GSB_hierarchical_structural_small.exo",
#    "GSB_hierarchical_structural_boundary_nodes",
#)

# Optional visualization.
# plot_hierarchical_meshes_3d(
#     mesh1_structural,
#     mesh2_structural,
#     structural_model_result,
#     show_plotter=True,
#     link_views=True,
#     volume_opacity=1.0,
#     show_block_controls=True,
# )

# -------------------------------------------------------------------------------
# 3B. Hierarchical mesh directly from CSV surfaces
# -------------------------------------------------------------------------------

#surface_csv_dir = (
#    cwd + "/examples/case_studies/Groß_Schoenebeck/input_data/surfaces"
#)

# CSV mode is selected with the top-level surface_csv_dir argument.
# extent1="whole" derives xmin/xmax/ymin/ymax/zmin/zmax from all CSV surfaces.
# extent_z_enlarged contains positive extension lengths [below_zmin, above_zmax].
# Here the inferred CSV domain is extended 10 m downward and 10 m upward in Z.
# If an explicit extent1 is supplied instead, that explicit extent takes
# precedence and extent_z_enlarged is not applied.
# No structural model is used to generate the geological surfaces in this
# hierarchy. Common unstructured parameters still apply to both mesh levels.
#mesh1_csv, mesh2_csv, nodes_bcs_csv = hierarchical_mesh(
#    geomodel_result=None,
#    mesh_type="Un",
#    surface_csv_dir=surface_csv_dir,
#    extent1="whole",
#    extent_z=[-4470.23, -3678.26],
#    extent2=hierarchical_small_extent,
#    wells=wells,
#    triangulations=triangulations,
#    mesh_size1=75,
#    mesh_size2=65,
#    unstructured_params={
#        "curve_mesh_size": 5,
#        "refinement": refinement,
#        "smooth": 0.5,

#    },
#    unstructured_params2={
#        "rotation_angle": 18,
#        "smooth": 0.5,

#    },
#)

#export_meshes_exodus(
#    mesh1_csv,
#    mesh2_csv,
#    nodes_bcs_csv,
#    "GSB_hierarchical_csv_big.exo",
#    "GSB_hierarchical_csv_small.exo",
#    "GSB_hierarchical_csv_boundary_nodes",
#)

# Optional visualization. structural_model_result is supplied here only as
# visualization context; the CSV meshes themselves are generated from CSV data.
# plot_hierarchical_meshes_3d(
#     mesh1_csv,
#     mesh2_csv,
#     structural_model_result,
#     show_plotter=True,
#     link_views=True,
#     volume_opacity=1.0,
#     show_block_controls=True,
# )

#%%

# -------------------------------------------------------------------------------
# END SECTION 3: HIERARCHICAL MESH GENERATION AND EXPORT
# ===============================================================================


# Solving Gross Schonebeck steady and transient i hierarchical manner
# Solve Steady state

# Solving the problem using custom golem input file. The GOLEM input file contains the Exodus material-block
#reservoir_input_path = (
#    cwd
#    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/steady_th.i"
#)

#with open(reservoir_input_path, "r") as f:
#    reservoir_input = f.read()

# Solving the problem with Golem
#model1_simple_reservoir_results = run_simulation_golem(
#    problem=CustomGolemBuilder(
#       input_file_contents=reservoir_input,

        # Original .i file location.
        # Used to save steady_hierarchy_sampling.i beside steady.i.
#        input_file_path=reservoir_input_path,

#        mesh1=mesh1_structural,
#        geomodel_result=structural_model_result,
#        print_linear_residuals=True,

        # Hierarchical GOLEM sampling
#        hierarchy_sample_nodes=nodes_bcs_structural,
#        sample_variables=[
#            "pore_pressure",
#            "temp",
#        ],
        #        sample_variables=[
        #    "pore_pressure",
        #    "temp",
        #    "stress_xx", "stress_yy", "stress_zz",
        #],
#    ),
#    image_name="moose-dev:fixed",
#    project_dir="/Users/marziehbaes/Documents/projects",
#    docker_dir="/projects",
#    output_dir=(
#        cwd
#        + "/examples/case_studies/Groß_Schoenebeck/"
#          "golem_GR_hie_steady_big_output"
#    ),
#    nproc=2,
#)


