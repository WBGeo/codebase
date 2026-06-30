# Importing necessary libraries
import pandas as pd
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

#%%

cwd = os.getcwd()

# WORKFLOW Groß Schoenebeck Case Study: 7-layer structural model

#%%

# Load input data
surface_points = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_surface_points_downsampled.csv")
orientations = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_orientations_downsampled.csv")

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
mesh_explicit_unstr = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=500,
    mesh_size=60,
    smooth=30,
    curve_mesh_size=5,
    extent=[401377.0, 409314.0, 5859433.0, 5865390.0, -3678.0, -4453.0],
    mapping_litho='none', # it can be 'manual', 'auto' or 'none' (default: auto)

)

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstr, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_str, structural_model_result, show_plotter=True)

#%%

# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
buf = export_mesh_results_to_exodus(mesh_explicit_unstr, type='unstr')
with open("filename_Gross_Schoenebeck.exo", "wb") as f:
    f.write(buf.getvalue())
