# Importing necessary libraries
import pandas as pd
import os

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

from core.simulation_components.simulation_packages.sfepy.simulation_run import run_sfepy
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results
from core.simulation_components.visualisation.results_visualisation import (
    plot_variable_at_a_time, plot_cross_section, plot_variable_along_line,
    print_variable_at_point, plot_variable_time_series)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 1: no faults, no unconformities, 2 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),
    resolution=(50, 50, 50)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_1',
                                             mapping_object={"Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model1/input_data/geological_data/model1_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model1/input_data/geological_data/model1_orientations_df.csv")
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
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()

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

# Compute 3D meshes based on the structural model result using different meshing approaches.

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
mesh_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    mesh_size=25,
    curve_mesh_size=5
)

# Explicit structured mesh
mesh_explicit_structured = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[10,10,10])

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_unstructured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_structured, structural_model_result, show_plotter=True)

#%%

# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# buf = export_mesh_results_to_exodus(mesh_unstructured)
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())

#%%

# Optional: Example of how to include objects (only works for unstructured mesh)

# # Load engineering objects
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_ellipses.csv")
# triangulations = load_triangulations_planes_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/seismic_plane_new_offset.csv")
#
# # Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     triangulations=triangulations,
#     extra_planes=planes,
#     ellipses=ellipses,
#     mesh_size=75,
#     curve_mesh_size=5
# )
#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%

# Process simulation with SfePy on the implicit structured mesh

mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)
Sim_out=run_sfepy(cwd +'/examples/synthetic_examples/model1/input_data/simulation_input_file/Hydro_thermal.py', mesh_implicit, 'results')

#%%

# Optional: Run simulation with unstructured or explicit structured mesh instead
# mesh_unst = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     mesh_size=50,
#     curve_mesh_size=5,
# )
# Sim_out=run_sfepy(cwd +'/examples/input_data/engineering_objects/Model1/Hydro_thermal.py', mesh_unst, 'results')
#
# mesh_str = create_structured_mesh_data(
#     geomodel_result=structural_model_result,
#     refinement_data=(40,40,40),
#     mesh_division=(40,40),
#     z_threshold=0.1,
#     tolerance=1
# )
# Sim_out=run_sfepy(cwd +'/examples/input_data/engineering_objects/Model1/Hydro_thermal.py', mesh_str, 'results')

#%%

# Visualize simulation results
data_by_time=load_vtk_results(Sim_out)
for t in data_by_time.nodes_by_time:
    print(f"\n⏱ Time {t}")
    print("  Node data keys:", list(data_by_time.node_data_by_time[t].keys()))
    print("  Cell data keys:", list(data_by_time.cell_data_by_time[t].keys()))
plot_variable_at_a_time(data_by_time,"p", 0, cmap="coolwarm", scale=(1,1,1))
plot_cross_section(data_by_time,"T", 0, origin=(540,20,100), normal=(1,0,0))
plot_variable_along_line(data_by_time,"p", 0, p0=(500,20,50), p1=(500,20,1000))
print_variable_at_point(data_by_time,"p", 0, point=(500.0,20.0,500.0))
plot_variable_time_series(data_by_time,"p", point=(500.0,20.0,500.0))

