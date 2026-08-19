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

from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, RockUnitProperties, FluidProperties)
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_run import run_simulation_sfepy
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement)
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
mesh_explicit_unstructured = create_unstructured_mesh_data(
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
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_ellipses.csv")
# csv_files=(cwd + "/examples/synthetic_examples/Model1/input_data/Engineering_objects/seismic_plane_new_offset_0.csv",
#           cwd + "/examples/synthetic_examples/Model1/input_data/Engineering_objects/seismic_plane_new_offset_1.csv")
# triangulations= load_triangulations_planes_from_csv(csv_files)

# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
# refinement = Refinement()
# refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
# refinement.triangulation = TriangulationRefinement(hmin=8.0, hmax=90.0, d1=50.0, d2=100.0, enabled=True)
#
# Available types per object:
#   wells / sources:                    LinearWellRefinement, FunctionWellRefinement
#                                       LinearSourceRefinement, FunctionSourceRefinement
#   faults / ellipses / triangulations: FaultRefinement, EllipseRefinement, TriangulationRefinement
#                                       (hmin/hmax = element sizes, d1/d2 = transition distances)


# # Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     extent=(0,1000,0,1000,0,950),
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     triangulations=triangulations,
#     extra_planes=planes,
#     ellipses=ellipses,
#     mesh_size=75,
#     curve_mesh_size=5,
#     gmsh_flag= True,   # to save original gmsh configuration (defaut is False)
#     mapping_litho='manual', # it can be 'manual', 'automatic_centers', 'automatic_corners' or 'none' (default: automatic_centers)
#     merge_file = cwd + "/examples/synthetic_examples/model1/input_data/block_groups.csv",  # if mapping_litho='manual'
#     refinement=refinement,
# )

# save the mesh
# buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type='unstr')
# with open("filename_example_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%

# Process simulation with SfePy on the implicit structured mesh.

rock_properties = {
    "basement": RockUnitProperties(name="basement", porosity=0.10, permeability=1e-13, k_solid=50, rho_c_solid=3000 * 1300),
    "rock1": RockUnitProperties(name="rock1", porosity=0.30, permeability=1e-11, k_solid=10, rho_c_solid=1000 * 800),
    "rock2": RockUnitProperties(name="rock2", porosity=0.20, permeability=1e-12, k_solid=20, rho_c_solid=2000 * 1700),
}
fluid = FluidProperties(mu=1.0e-3, k_fluid=15, rho_c_fluid=2000 * 1200)

builder = HydrothermalProblemBuilder(
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    include_flow=True,
    t1=5e11,
    num_steps=3,
)

#%%

# Pre-flight check: confirms rock_properties landed on the right cells.
plot_builder_materials(builder)

#%%

data_by_time = run_simulation_sfepy(builder)

#%%

# Visualize simulation results
final_time = max(data_by_time.nodes_by_time.keys())
plot_variable_at_a_time(data_by_time, "T", time=0, cmap="coolwarm", show_edges=True)
plot_variable_at_a_time(data_by_time, "T", time=final_time, cmap="coolwarm", show_edges=True)
plot_variable_at_a_time(data_by_time, "p", time=final_time, cmap="coolwarm", show_edges=True)
plot_cross_section_2D(data_by_time, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# CustomSfepyBuilder example: bring a complete, hand-written SfePy input
# file instead of letting HydrothermalProblemBuilder auto-generate one.
# CustomSfepyBuilder validates it against the mesh at construction time,
# then runs through the exact same run_simulation_sfepy() dispatcher.
#
# Two worked examples are provided for this mesh (no fault, implicit
# structured); pick one by (un)commenting custom_input_path below.
#
#   - custom_hydrothermal_fromscratch_implicit.py (active by default)
#     implements the same two-stage pressure -> Darcy velocity -> heat
#     problem as HydrothermalProblemBuilder(include_flow=True), but written
#     from first principles: regions, materials and boundary conditions are
#     derived using only public helper functions and coordinate-based SfePy
#     selectors, without relying on HydrothermalProblemBuilder's own
#     generated pressure-stage file. On this mesh (an axis-aligned
#     Cartesian grid, no fault) it reproduces HydrothermalProblemBuilder's
#     result exactly, not just closely -- see the file's own docstring for
#     why that's different from model2's fromscratch file, which only
#     agrees to within solver tolerance on its unstructured, faulted mesh.
#
#   - custom_hydrothermal_reproduction_implicit.py instead reproduces the
#     HydrothermalProblemBuilder call above bit-for-bit by embedding its
#     own build_pressure_input_file/build_heat_input_file output directly.
#
# See each file's own docstring for the full technical explanation.

custom_input_path = (cwd + "/examples/synthetic_examples/model1/input_data/simulation_files/"
                      "custom_hydrothermal_fromscratch_implicit.py")
# custom_input_path = (cwd + "/examples/synthetic_examples/model1/input_data/simulation_files/"
#                       "custom_hydrothermal_reproduction_implicit.py")
with open(custom_input_path) as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_implicit_structured,
    geomodel_result=structural_model_result,
)

#%%

custom_data_by_time = run_simulation_sfepy(custom_builder)

#%%

# Confirms the custom-builder result matches HydrothermalProblemBuilder's above.
custom_final_time = max(custom_data_by_time.nodes_by_time.keys())
print("HydrothermalProblemBuilder T mean:", data_by_time.node_data_by_time[final_time]["T"].mean())
print("CustomSfepyBuilder        T mean:", custom_data_by_time.node_data_by_time[custom_final_time]["T"].mean())
