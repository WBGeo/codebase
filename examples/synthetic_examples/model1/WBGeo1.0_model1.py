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
    load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, \
    load_ellipses_from_csv, load_triangulations_planes_from_csv
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
    GolemHydrothermalProblemBuilder, CustomGolemBuilder, GolemSimpleReservoirProblemBuilder)
from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem
from core.simulation_components.simulation_packages.simulation_export import export_simulation_results
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials, plot_variable_time_series)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 1: no faults, no unconformities, 2 stratigraphic groups
# This is one of the two detailed examples (with model 2): structural model, meshes, engineering
# objects with all refinement options, and simulations with SfePy and GOLEM.

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

# Compute 3D meshes based on the structural model result using different meshing approaches

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    mesh_size=30,
    curve_mesh_size=5,
)

# Explicit structured mesh
mesh_explicit_structured = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[20, 20, 20],
)

#%%

# Plot the meshing results
plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_structured, structural_model_result, show_plotter=True)

#%%

# Optional: Export a mesh, here the structured mesh to Exodus.
# The other exporters work the same way (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# Only the Exodus exporter needs the mesh type ('imp', 'str' or 'unstr'), and some formats
# support only some mesh types; see the meshing manual.
# buf = export_mesh_results_to_exodus(mesh_explicit_structured, type='str')
# with open("model1_structured_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Optional: Add engineering objects to the unstructured mesh (only works for unstructured meshes)

# Load the engineering objects
# wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_wells.csv")
# shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_shafts.csv")
# sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_sources.csv")
# planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_planes.csv")
# ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_ellipses.csv")
# triangulations = load_triangulations_planes_from_csv((
#     cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/seismic_plane_new_offset_0.csv",
#     cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/seismic_plane_new_offset_1.csv"))

# Optional: Refine the mesh around the objects
# refinement = Refinement()
# refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.shafts = LinearShaftRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
# refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
# refinement.triangulations = TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0)
# refinement.ellipses = EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
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
# Refinement(wells=LinearWellRefinement(...), sources=FunctionSourceRefinement(...), ...).

# Explicit unstructured mesh with objects
# mesh_unstructured_with_objects = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     extent=(0, 1000, 0, 1000, 0, 950),
#     wells=wells,
#     sources=sources,
#     shafts=shafts,
#     triangulations=triangulations,
#     extra_planes=planes,
#     ellipses=ellipses,
#     mesh_size=75,
#     curve_mesh_size=5,
#     refinement=refinement,
#     # gmsh_flag=True,  # also save the original Gmsh configuration (default: False)
#     # mapping_litho='manual',  # 'manual', 'automatic_centers' (default), 'automatic_corners' or 'none'
#     # merge_file=cwd + "/examples/synthetic_examples/model1/input_data/block_groups.csv",  # for mapping_litho='manual'
# )
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%

# Simulate heat and fluid flow with SfePy on the unstructured mesh (generated hydrothermal problem)
# The SfePy simulations use the PETSc solver and need petsc4py (e.g. in Docker). Without it, pass
# linear_solver="iterative" to the generated problem; the custom input files use PETSc themselves.
rock_properties = {
    "basement": RockUnitProperties(
        name="basement", porosity=0.10, permeability=1e-13,
        thermal_conductivity=50, density=3000, heat_capacity=1300,
    ),
    "rock1": RockUnitProperties(
        name="rock1", porosity=0.30, permeability=1e-11,
        thermal_conductivity=10, density=1000, heat_capacity=800,
    ),
    "rock2": RockUnitProperties(
        name="rock2", porosity=0.20, permeability=1e-12,
        thermal_conductivity=20, density=2000, heat_capacity=1700,
    ),
}
fluid = FluidProperties(
    viscosity=1.0e-3,
    thermal_conductivity=15,
    density=2000,
    heat_capacity=1200,
)

builder = HydrothermalProblemBuilder(
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    rock_properties=rock_properties,
    fluid=fluid,
    include_flow=True,
    end_time=5e11,
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

# Plot the simulation results
final_time = max(data_by_time.nodes_by_time.keys())
plot_variable_at_a_time(data_by_time, "T", time=0, cmap="coolwarm", show_edges=True)
plot_variable_at_a_time(data_by_time, "T", time=final_time, cmap="coolwarm", show_edges=True)
plot_variable_at_a_time(data_by_time, "p", time=final_time, cmap="coolwarm", show_edges=True)
plot_cross_section_2D(data_by_time, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")
plot_variable_time_series(data_by_time, "T", point=(500, 500, 500))

#%%

# Optional: Export the results (VTK files with a ParaView collection, plus input files and solver output)
# buf = export_simulation_results(data_by_time, format="vtk")
# with open("model1_sfepy_results.zip", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Simulate with a hand-written SfePy input file instead of the generated one (custom problem).
# custom_hydrothermal.py solves the same two-stage problem (pressure, Darcy velocity, heat) as the
# generated builder above; on this mesh it reproduces that result exactly (see the file's docstring).
with open(cwd + "/examples/synthetic_examples/model1/input_data/simulation_files/custom_hydrothermal.py") as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
)

plot_builder_materials(custom_builder)
plot_builder_materials(custom_builder, property_name="porosity")

custom_data_by_time = run_simulation_sfepy(custom_builder, stream_output=True)

#%%

# Compare with the generated problem and plot the results
custom_final_time = max(custom_data_by_time.nodes_by_time.keys())
print("HydrothermalProblemBuilder T mean:", data_by_time.node_data_by_time[final_time]["T"].mean())
print("CustomSfepyBuilder        T mean:", custom_data_by_time.node_data_by_time[custom_final_time]["T"].mean())
plot_variable_at_a_time(custom_data_by_time, "T", time=custom_final_time, cmap="coolwarm", show_edges=True)
plot_cross_section_2D(custom_data_by_time, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# Optional: Simulate with GOLEM (requires Docker with the GOLEM image), generated hydrothermal problem.
# Same rock and fluid properties as the SfePy problem above. Plots accept "T"/"p" as well as GOLEM's
# own names "temperature"/"pore_pressure".
# golem_problem = GolemHydrothermalProblemBuilder(
#     mesh_results=mesh_explicit_unstructured,
#     geomodel_result=structural_model_result,
#     rock_properties=rock_properties,
#     fluid=fluid,
#     p_top=0.0,
#     p_bottom=1.0e6,
#     t_top=10.0,
#     t_bottom=60.0,
#     end_time=5e11,
#     num_steps=3,
# )
# plot_builder_materials(golem_problem)
# golem_results = run_simulation_golem(
#     problem=golem_problem,
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
# )
# golem_final_time = max(golem_results.nodes_by_time.keys())
# plot_variable_at_a_time(golem_results, "T", time=golem_final_time, cmap="coolwarm", show_edges=True)
# plot_cross_section_2D(golem_results, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")
# buf = export_simulation_results(golem_results, format="exodus")
# with open("model1_golem_results.zip", "wb") as f:
#     f.write(buf.getvalue())

#%%

# Optional: Simulate with GOLEM (requires Docker with the GOLEM image), custom input file
# with open(cwd + "/examples/synthetic_examples/model1/input_data/simulation_files/custom_hydrothermal.i",
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
#     stream_output=True,
# )
# custom_golem_final_time = max(custom_golem_results.nodes_by_time.keys())
# plot_cross_section_2D(custom_golem_results, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")

#%%

# Optional: Simulate a simple reservoir with GOLEM (requires Docker with the GOLEM image).
# The builder meshes the model itself: a doublet of two 100 m fracture planes 200 m apart and a well,
# centred in the model, with injection and extraction at the plane centres. All rock units get the
# same rock properties. The time sequence below reproduces the model 1 benchmark (20 output states).
# reservoir_problem = GolemSimpleReservoirProblemBuilder(
#     geomodel_result=structural_model_result,
#     mesh_size=50.0,
#     curve_mesh_size=5.0,
#     fracture_size=100.0,
#     fracture_spacing=200.0,
#     rock_porosity=0.1,
#     rock_permeability=1.0e-15,
#     rock_density=2.6e3,
#     rock_thermal_conductivity=3.0,
#     rock_heat_capacity=950.0,
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
#     time_sequence=[1.0, 86400.0, 604800.0, 2592000.0, 7776000.0, 15552000.0, 31536000.0, 63072000.0,
#                    126144000.0, 315360000.0, 630720000.0, 946080000.0, 1261440000.0, 1576800000.0,
#                    1892160000.0, 2207520000.0, 2522880000.0, 2838240000.0, 3153600014.0],
# )
# plot_builder_materials(reservoir_problem, opacity=0.12)
# plot_builder_materials(reservoir_problem, property_name="rock_porosity", cmap="plasma", opacity=0.25)
# reservoir_results = run_simulation_golem(
#     problem=reservoir_problem,
#     image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#     workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#     nproc=4,
# )
# plot_cross_section_2D(reservoir_results, "T", origin=reservoir_problem.injection_point, normal=(0, 1, 0), cmap="coolwarm")
# plot_cross_section_2D(reservoir_results, "p", origin=reservoir_problem.injection_point, normal=(0, 1, 0), cmap="coolwarm")
# The ParaView state model1_reservoir_visualisation.pvsm shows the results (see paraview_readme.md).

#%%

# Optional: Simulate the same simple reservoir with GOLEM from a custom input file
# (requires Docker with the GOLEM image). The mesh is built explicitly with the same geometry as the
# generated builder above; model1_simple_reservoir.i refers to its material blocks.
# reservoir_mesh = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     extent=(0, 1000, 0, 1000, 0, 1000),
#     extra_planes=[
#         (400.0, 450.0, 450.0, 400.0, 550.0, 450.0, 400.0, 550.0, 550.0, 400.0, 450.0, 550.0),
#         (600.0, 450.0, 450.0, 600.0, 550.0, 450.0, 600.0, 550.0, 550.0, 600.0, 450.0, 550.0),
#     ],
#     wells=[(575.0, 500.0, 490.0, 587.5, 500.0, 495.0, 600.0, 500.0, 500.0, 612.5, 500.0, 505.0,
#             625.0, 500.0, 510.0)],
#     mesh_size=50.0,
#     curve_mesh_size=5.0,
#     refinement=Refinement(
#         wells=LinearWellRefinement(SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0),
#         extra_planes=ExtraPlaneRefinement(SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0),
#     ),
# )
# with open(cwd + "/examples/synthetic_examples/model1/input_data/simulation_files/model1_simple_reservoir.i",
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
# )
# plot_cross_section_2D(custom_reservoir_results, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")
