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

from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
    load_csv_surface_coordinates, load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, \
    load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, LinearSourceRefinement, FunctionSourceRefinement,
    LinearShaftRefinement, FunctionShaftRefinement, FaultRefinement, EllipseRefinement, TriangulationRefinement,
    ExtraPlaneRefinement, GeneralRefinement)
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)

from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, RockUnitProperties, FluidProperties)
from core.simulation_components.simulation_packages.sfepy.sfepy_run import run_simulation_sfepy
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemHydrothermalProblemBuilder, GolemFaultProperties, CustomGolemBuilder, GolemSimpleReservoirProblemBuilder)
from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem
from core.simulation_components.simulation_packages.simulation_export import export_simulation_results
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 2: 1 fault, 1 unconformity, 2 stratigraphic groups
# This is one of the two detailed examples (with model 1): structural model with a fault, meshes,
# engineering objects with all refinement options, and simulations with SfePy and GOLEM.

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),
    resolution=(100, 40, 40)
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_2',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model2/input_data/geological_data/model2_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model2/input_data/geological_data/model2_orientations_df.csv"),
                                      fault_names=['fault'])

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
data_elements = InputData_StructuralElements(name='Model_2',
                                             mapping_object={
                                                 "Strat_Series2": ('rock4', 'rock3'),
                                                 "Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model2/input_data/geological_data/model2_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model2/input_data/geological_data/model2_orientations_df.csv")
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

# Set another interpolation method per group
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")
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

# Compute 3D meshes based on the structural model result using different meshing approaches

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh (contains the fault as its own 2D block, which GOLEM uses)
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=50,
    mesh_size=50,
    curve_mesh_size=5,
)

# Explicit structured mesh
# Not available here: the structured mesh doesn't support models with faults yet.

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)

#%%

# Optional: Mesh surfaces given as CSV files instead of a structural model (one file per surface).
# Without a structural model there are no lithology names, so mapping_litho is "none".
# mesh_from_csv_surfaces = create_unstructured_mesh_data(
#     geomodel_result=load_csv_surface_coordinates(cwd + "/examples/synthetic_examples/model2/input_data/surfaces"),
#     extent=(0, 2500, 0, 1000, 0, 1000),
#     tolerance=50,
#     mesh_size=50,
#     curve_mesh_size=5,
#     mapping_litho="none",
# )
# plot_mesh_3d(mesh_from_csv_surfaces, structural_model_result, show_plotter=True)

#%%

# Optional: Export a mesh, here the unstructured mesh to Exodus.
# The other exporters work the same way (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# Only the Exodus exporter needs the mesh type ('imp', 'str' or 'unstr'), and some formats
# support only some mesh types; see the meshing manual.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured, type='unstr')
# with open("model2_unstructured_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Optional: Add engineering objects to the unstructured mesh (only works for unstructured meshes).
# With all objects and the refinement below, the mesh has about 350k nodes instead of 20k and took
# about 14 minutes on a desktop PC (the plain mesh: a few seconds).

# Load the engineering objects
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_ellipses.csv")
# triangulations = load_triangulations_planes_from_csv((
#     cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/seismic_plane_new_offset_0.csv",
#     cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/seismic_plane_new_offset_1.csv"))

# Optional: Refine the mesh around the objects (and around the fault)
# refinement = Refinement()
# refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.shafts = LinearShaftRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
# refinement.triangulations = TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0)
# refinement.ellipses = EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
# refinement.faults = FaultRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
# refinement.extra_planes = ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0)
# refinement.general = [
#     GeneralRefinement(coordinates=[(500.0, 400.0, 0.0)], SizeMin=1.0, DistMax=70.0),
#     GeneralRefinement(coordinates=[(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0)], expression="2 + 85*(1 - exp(-DIST/100))"),
# ]

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
#
# The same configuration can also be written as one Refinement(...) call, e.g.
# Refinement(wells=LinearWellRefinement(...), faults=FaultRefinement(...), ...).

# Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     tolerance=50,
#     mesh_size=50,
#     curve_mesh_size=5,
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     triangulations=triangulations,
#     extra_planes=planes,
#     ellipses=ellipses,
#     refinement=refinement,
#     # gmsh_flag=True,  # also save the original Gmsh configuration (default: False)
# )
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%

# Simulate heat and fluid flow with SfePy on the unstructured mesh (generated hydrothermal problem).
# The fault gets its own material in a damage zone of cells along it (fault_zone_properties).
# The SfePy simulations use the PETSc solver and need petsc4py (e.g. in Docker). Without it, pass
# linear_solver="iterative" to the generated problem; the custom input files use PETSc themselves.
rock_properties = {
    "basement": RockUnitProperties(name="basement", porosity=0.05, permeability=1e-15, k_solid=3.0, rho_c_solid=2.2e6),
    "rock1": RockUnitProperties(name="rock1", porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6),
    "rock2": RockUnitProperties(name="rock2", porosity=0.15, permeability=1e-14, k_solid=2.5, rho_c_solid=2.2e6),
    "rock3": RockUnitProperties(name="rock3", porosity=0.15, permeability=1e-14, k_solid=2.0, rho_c_solid=2.1e6),
    "rock4": RockUnitProperties(name="rock4", porosity=0.15, permeability=1e-14, k_solid=2.0, rho_c_solid=2.1e6),
}
fluid = FluidProperties(
    viscosity=1.0e-3,
    thermal_conductivity=0.6,
    density=1000,
    fluid_modulus=2.2e9,
)
# Sealing fault (low porosity and permeability compared with the host rock); use higher values for a conduit
fault_zone_properties = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
)

builder = HydrothermalProblemBuilder(
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    include_flow=True,
    fault_zone_properties=fault_zone_properties,
    fault_zone_n_voxels=1,
    end_time=5.0e11,
    num_steps=3,
    print_linear_residuals=True,
)

#%%

# Check the material assignment before running the solver
plot_builder_materials(builder)
plot_builder_materials(builder, property_name="porosity")

#%%

# Run the simulation (stream_output=True prints SfePy's output while it runs)
data_by_time = run_simulation_sfepy(builder, stream_output=True)

#%%

# Plot the simulation results. normal=(0, 1, 0) cuts across the fault, which doesn't change along y.
final_time = max(data_by_time.nodes_by_time.keys())
plot_variable_at_a_time(data_by_time, "T", time=0, cmap="coolwarm", show_edges=False)
plot_variable_at_a_time(data_by_time, "T", time=final_time, cmap="coolwarm", show_edges=False)
plot_cross_section_2D(data_by_time, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# Optional: Export the results (VTK files with a ParaView collection, plus input files and solver output)
# buf = export_simulation_results(data_by_time, format="vtk")
# with open("model2_sfepy_results.zip", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Simulate with a hand-written SfePy input file instead of the generated one (custom problem).
# custom_hydrothermal.py solves the same two-stage problem (pressure, Darcy velocity, heat) with the
# fault damage zone; it agrees with the generated problem within solver tolerance (see its docstring).
with open(cwd + "/examples/synthetic_examples/model2/input_data/simulation_files/custom_hydrothermal.py") as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    fault_zone_n_voxels=1,
)

plot_builder_materials(custom_builder)
plot_builder_materials(custom_builder, property_name="porosity")

custom_data_by_time = run_simulation_sfepy(custom_builder, stream_output=True)

#%%

# Plot the custom simulation results
custom_final_time = max(custom_data_by_time.nodes_by_time.keys())
plot_variable_at_a_time(custom_data_by_time, "T", time=custom_final_time, cmap="coolwarm", show_edges=False)
plot_cross_section_2D(custom_data_by_time, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# Optional: Simulate with GOLEM (requires Docker with the GOLEM image), generated hydrothermal problem.
# GOLEM represents the fault as the mesh's 2D fault block. It uses the damage-zone rock values from
# above; the aperture scales the lower-dimensional fault elements.
# golem_fault_properties = GolemFaultProperties(
#     name="fault",
#     porosity=fault_zone_properties.porosity,
#     permeability=fault_zone_properties.permeability,
#     k_solid=fault_zone_properties.k_solid,
#     rho_c_solid=fault_zone_properties.rho_c_solid,
#     fluid_modulus=fault_zone_properties.fluid_modulus,
#     aperture=40,
# )
# golem_problem = GolemHydrothermalProblemBuilder(
#     mesh_results=mesh_explicit_unstructured,
#     geomodel_result=structural_model_result,
#     rock_properties=rock_properties,
#     fault_properties=golem_fault_properties,
#     fluid=fluid,
#     p_top=0.0,
#     p_bottom=1.0e6,
#     t_top=10.0,
#     t_bottom=60.0,
#     end_time=5.0e11,
#     num_steps=3,
# )
# plot_builder_materials(golem_problem, show_edges=True)
# plot_builder_materials(golem_problem, property_name="porosity")
# golem_results = run_simulation_golem(
#     problem=golem_problem,
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
#     stream_output=True,
# )
# golem_final_time = max(golem_results.nodes_by_time.keys())
# plot_variable_at_a_time(golem_results, "T", time=golem_final_time, cmap="coolwarm", show_edges=True)
# plot_cross_section_2D(golem_results, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm")
# buf = export_simulation_results(golem_results, format="exodus")
# with open("model2_golem_results.zip", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Optional: Simulate with GOLEM (requires Docker with the GOLEM image), custom input file
# with open(cwd + "/examples/synthetic_examples/model2/input_data/simulation_files/custom_hydrothermal.i",
#           encoding="utf-8") as f:
#     custom_golem_input = f.read()
# custom_golem_problem = CustomGolemBuilder(
#     input_file_contents=custom_golem_input,
#     mesh_results=mesh_explicit_unstructured,
#     geomodel_result=structural_model_result,
# )
# plot_builder_materials(custom_golem_problem, show_edges=True)
# plot_builder_materials(custom_golem_problem, property_name="porosity_initial")
# custom_golem_results = run_simulation_golem(
#     problem=custom_golem_problem,
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
# )
# custom_golem_final_time = max(custom_golem_results.nodes_by_time.keys())
# plot_cross_section_2D(custom_golem_results, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# Optional: Simulate a simple reservoir with GOLEM (requires Docker with the GOLEM image).
# The builder meshes the model itself: a doublet of two 100 m fracture planes 200 m apart and a well,
# centred in the model. The model's own fault is kept as a geological fault with the rock properties;
# all rock units share one material. Meshing takes about 1.5 minutes.
# reservoir_problem = GolemSimpleReservoirProblemBuilder(
#     geomodel_result=structural_model_result,
#     mesh_size=50.0,
#     curve_mesh_size=5.0,
#     unstructured_mesh_options={"tolerance": 50},
#     rock_porosity=0.10,
#     rock_permeability=1.0e-15,
#     rock_density=2.6e3,
#     rock_thermal_conductivity=3.0,
#     rock_heat_capacity=950.0,
#     geological_fault_aperture=1.0,
#     fracture_porosity=1.0,
#     fracture_aperture=1.0e-4,
#     well_porosity=1.0,
#     well_permeability=3.12e-4,
#     well_scaling_factor=0.05,
#     initial_pressure=1.0e7,
#     initial_temperature=150.0,
#     boundary_pressure=1.0e7,
#     boundary_temperature=150.0,
#     injection_temperature=70.0,
#     injection_rate=3.0,
#     extraction_rate=3.0,
# )
# plot_builder_materials(reservoir_problem, opacity=0.12)
# reservoir_results = run_simulation_golem(
#     problem=reservoir_problem,
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
# )
# plot_cross_section_2D(reservoir_results, "T", origin=reservoir_problem.injection_point, normal=(0, 1, 0), cmap="coolwarm")
# plot_cross_section_2D(reservoir_results, "p", origin=reservoir_problem.injection_point, normal=(0, 1, 0), cmap="coolwarm")
# The ParaView state model2_reservoir_visualisation.pvsm shows the results (see paraview_readme.md).

#%%

# Optional: Simulate the same simple reservoir with GOLEM from a custom input file
# (requires Docker with the GOLEM image). The mesh is built explicitly with the same geometry as the
# generated builder above (planes at x = 1150 m and 1350 m); custom_reservoir.i refers to its blocks.
# reservoir_mesh = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     extent=(0, 2500, 0, 1000, 0, 1000),
#     extra_planes=[
#         (1150.0, 450.0, 450.0, 1150.0, 550.0, 450.0, 1150.0, 550.0, 550.0, 1150.0, 450.0, 550.0),
#         (1350.0, 450.0, 450.0, 1350.0, 550.0, 450.0, 1350.0, 550.0, 550.0, 1350.0, 450.0, 550.0),
#     ],
#     wells=[(1325.0, 500.0, 490.0, 1337.5, 500.0, 495.0, 1350.0, 500.0, 500.0, 1362.5, 500.0, 505.0,
#             1375.0, 500.0, 510.0)],
#     tolerance=50,
#     mesh_size=50.0,
#     curve_mesh_size=5.0,
#     refinement=Refinement(
#         wells=LinearWellRefinement(SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0),
#         extra_planes=ExtraPlaneRefinement(SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0),
#     ),
# )
# plot_mesh_3d(reservoir_mesh, structural_model_result, show_plotter=True)
# with open(cwd + "/examples/synthetic_examples/model2/input_data/simulation_files/custom_reservoir.i",
#           encoding="utf-8") as f:
#     reservoir_input = f.read()
# custom_reservoir_results = run_simulation_golem(
#     problem=CustomGolemBuilder(
#         input_file_contents=reservoir_input,
#         mesh_results=reservoir_mesh,
#         geomodel_result=structural_model_result,
#     ),
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
#     stream_output=True,
# )
# plot_cross_section_2D(custom_reservoir_results, "T", origin=(1150.0, 500.0, 500.0), normal=(0, 1, 0), cmap="coolwarm")
