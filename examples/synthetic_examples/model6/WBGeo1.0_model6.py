# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D, plot_structural_model_3D, plot_input_data_3D)

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

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 6: no faults, 2 unconformities, 3 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),
    resolution=(50, 50, 50)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_6',
                                             mapping_object={
                                                 "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
                                                 "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
                                                 "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1'),
                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model6/input_data/geological_data/model6_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model6/input_data/geological_data/model6_orientations_df.csv")
                                             )

#%%

# Plot input data without model context
plot_input_data_3D(data_elements)

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

# Set another interpolation method per group
# frame["Shallow_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Medium_Strat"].set_interpolation_method("Universal Co-Kriging")
# frame["Deep_Strat"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Shallow_Strat"].configure_interpolation_params()
# frame["Medium_Strat"].configure_interpolation_params()
# frame["Deep_Strat"].configure_interpolation_params()

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

# Compute 3D meshes based on the structural model result using different meshing approaches

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=5,
    mesh_size=40,
    curve_mesh_size=5,
)

# Explicit structured mesh
# Not available here: the structured mesh doesn't support models whose layers don't span the whole model extent yet.

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)

#%%

# Optional: Export a mesh, here the unstructured mesh to Exodus.
# The other exporters work the same way (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# Only the Exodus exporter needs the mesh type ('imp', 'str' or 'unstr'), and some formats
# support only some mesh types; see the meshing manual.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured, type='unstr')
# with open("model6_unstructured_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Optional: Add engineering objects to the unstructured mesh (only works for unstructured meshes).
# This model's wells, sources, shafts, planes and ellipses are in input_data/engineering_objects/.
# Models 1 and 2 show the full workflow, including all refinement options around the objects.
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model6/input_data/engineering_objects/model_6_wells.csv")
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     wells=wells,
#     tolerance=5,
#     mesh_size=40,
#     curve_mesh_size=5,
# )
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)
