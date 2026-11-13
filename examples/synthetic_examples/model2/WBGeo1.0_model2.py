from pathlib import Path
# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Importing necessary libraries
import pandas as pd
import os
import re
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D, plot_fault_model_2D,
plot_fault_input_data_3D, plot_input_data_3D)
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data

from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,ExtraPlaneRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement, LinearShaftRefinement, FunctionShaftRefinement, GeneralRefinement)

from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    HydrothermalProblemBuilder, CustomSfepyBuilder, RockUnitProperties, FluidProperties)
from core.simulation_components.simulation_packages.sfepy.sfepy_run import (run_simulation_sfepy,export_simulation_results,)

from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials)

from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemHydrothermalProblemBuilder,
    GolemFaultProperties,
    CustomGolemBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemSimpleReservoirProblemBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_run import (
    run_simulation_golem,
)

from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, load_csv_surface_coordinates, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv

#%%

cwd = os.getcwd()

# ==============================================================================
# MODEL 2 - SCRIPT MAP
# ==============================================================================
# SECTION 1: Structural geological model
# SECTION 2: Mesh generation and mesh utilities
# SECTION 3: SfePy generated hydrothermal simulation
# SECTION 4: SfePy custom hydrothermal simulation
# SECTION 5: GOLEM generated hydrothermal simulation
# SECTION 6: GOLEM custom hydrothermal simulation
# SECTION 7: GOLEM generated simple-reservoir simulation
# SECTION 8: GOLEM custom simple-reservoir simulation
#
# Each section starts and ends with a large banner so the different sections are
# visually separated while keeping all existing code in one file.
# ==============================================================================


#%%

# ==============================================================================
# SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
# Build the fault model, stratigraphy, structural frame, and structural result.
# ------------------------------------------------------------------------------

# WORKFLOW Synthetic Model 2: 1 fault, 1 unconformity, 2 stratigraphic groups

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 40, 40)  # Example resolution
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

# Plot the fault input input_data (2D and 3D possible)
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

# Set interpolation methods for each stratigraphic series

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


# ------------------------------------------------------------------------------
# END SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
#%%

# ==============================================================================
# SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================
# Build the structured, unstructured and implicit meshes.
# ------------------------------------------------------------------------------

# Compute 3D meshes based on the structural model result using different meshing approaches.

# Implicit structured mesh
mesh_implicit_structured= create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
#mesh_explicit_unstructured = create_unstructured_mesh_data(
#    geomodel_result=structural_model_result,
#    tolerance=50,
#    mesh_size=50,
#    curve_mesh_size=5,

#)
#plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)

# Explicit structured mesh
# NOTE: Currently structured mesh does not support models with faults

#%%

# Plot the meshing results
#plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)

mesh_explicit_unst_from_file = create_unstructured_mesh_data(
    geomodel_result= load_csv_surface_coordinates(cwd+ "/examples/synthetic_examples/model2/input_data/surfaces"),

    extent=(0, 2500, 0, 1000, 0, 1000),

    mesh_size=50,
    curve_mesh_size=5,
    tolerance=50,

    mapping_litho="none",
)
plot_mesh_3d(mesh_explicit_unst_from_file, structural_model_result, show_plotter=True)

#%%

# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstructured)
# with open("filename_example_mesh.exo", "wb") as f:
#    f.write(buf.getvalue())


#%%

# Optional: Example of how to include objects (only works for unstructured mesh)

# # Load engineering objects
wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_wells.csv")
shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_shafts.csv")
sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_sources.csv")
planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_planes.csv")
ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_ellipses.csv")
csv_files=(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/seismic_plane_new_offset_0.csv",
           cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/seismic_plane_new_offset_1.csv")
triangulations= load_triangulations_planes_from_csv(csv_files)

# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
refinement = Refinement()
refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
refinement.shafts = LinearShaftRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
refinement.triangulations = TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0)
refinement.ellipses = EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
refinement.faults = FaultRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)

refinement.extra_planes = ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0)
#refinement.general = [GeneralRefinement(coordinates=[(500.0, 400.0, 0.0),],SizeMin=1.0,DistMax=70.0,),
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
#    triangulations=TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0,),
#    ellipses=EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0,),
#    extra_planes=ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0,),
#    general=[
#        GeneralRefinement(coordinates=[(500.0, 400.0, 0.0),], SizeMin=4.0, DistMax=300.0,),
#        GeneralRefinement(coordinates=[(0.0, 0.0, 0.0),(1000.0, 0.0, 0.0),], expression="2 + 85*(1 - exp(-DIST/100))",),],)


# # Explicit unstructured mesh with objects
# Note: with all objects and the refinement above, this mesh is much finer than the
# plain one (about 350k nodes vs. 20k) and took about 14 minutes on a desktop PC
# (the plain mesh: a few seconds).
mesh_unstructured_with_objects = create_unstructured_mesh_data(
     geomodel_result=structural_model_result,
     tolerance=50,
     mesh_size=50,
     curve_mesh_size=5,
     wells=wells,
     sources=sources,
     shafts=shafts,
     triangulations=triangulations,
     extra_planes=planes,
     ellipses=ellipses,
#     gmsh_flag= True,
#         # to save original gmsh configuration (defaut is False)
     refinement=refinement,
)
#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)
buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type='unstr')
with open("filename_model2_with_objects_mesh.exo", "wb") as f:
     f.write(buf.getvalue())
#%%


# ------------------------------------------------------------------------------
# END SECTION 2: BASE MESH GENERATION AND MESH UTILITIES
# ==============================================================================
#%%

# ==============================================================================
# SECTION 3: SFEPY - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
# Run HydrothermalProblemBuilder on the standard unstructured mesh.
# ------------------------------------------------------------------------------

# Process simulation with SfePy on the unstructured mesh.

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
    fluid_modulus = 2.2e9 ,
)
# Sealing fault (low k/porosity/permeability vs. host rock); flip higher for a conduit instead.
fault_zone_properties = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
)

#builder = HydrothermalProblemBuilder(
#    mesh_results=mesh_explicit_unstructured,
#    geomodel_result=structural_model_result,
#    rock_properties=rock_properties,
#    fluid=fluid,
#    print_linear_residuals=True,

#    include_flow=True,
#    fault_zone_properties=fault_zone_properties,
#    fault_zone_n_voxels=1,

#    start_time=0.0,
#    end_time=5.0e11,
#    num_steps=3,
#)

#%%

# Pre-flight check: confirms rock_properties landed on the right cells.
#plot_builder_materials(builder)
#plot_builder_materials(
#    builder=builder,
#    property_name="porosity",
#    show_plotter=True,
#)

#%%

#data_by_time = run_simulation_sfepy(builder, stream_output=True)

#%%

# Visualize simulation results
#final_time = max(data_by_time.nodes_by_time.keys())
#plot_variable_at_a_time(data_by_time, "T", time=0, cmap="coolwarm", show_edges=False)
#plot_variable_at_a_time(data_by_time, "T", time=final_time, cmap="coolwarm", show_edges=False)

# normal=(0, 1, 0): the fault plane is Y-invariant, so this actually cuts across it.
#plot_cross_section_2D(
#    data_by_time, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
#)

#%%
# save the outputs as zip file

#buf = export_simulation_results(data_by_time , format="vtk")
#with open("simulation_results.zip", "wb") as f:
#    f.write(buf.getvalue())

# ------------------------------------------------------------------------------
# END SECTION 3: SFEPY - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
#%%

# ==============================================================================
# SECTION 4: SFEPY - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
# Run a hand-written SfePy input file to solve similar problem as the HydrothermalProblemBuilder.
# ------------------------------------------------------------------------------

# CustomSfepyBuilder example: bring a complete, hand-written SfePy input
# file instead of letting HydrothermalProblemBuilder auto-generate one.
# CustomSfepyBuilder validates it against the mesh at construction time,
# then runs through the exact same run_simulation_sfepy() dispatcher.
#
# One worked example is provided for this mesh (real fault, active damage
# zone, unstructured):
#
#   - custom_hydrothermal.py implements the same two-stage pressure -> Darcy velocity ->
#     heat problem as HydrothermalProblemBuilder(include_flow=True), but
#     written from first principles: regions, materials and boundary
#     conditions are derived using only public helper functions and
#     coordinate-based SfePy selectors, without relying on
#     HydrothermalProblemBuilder's own generated pressure-stage file. It
#     reproduces HydrothermalProblemBuilder's result closely (agreeing to
#     within numerical solver tolerance, not bit-for-bit, since the two are
#     independently authored) -- demonstrating that the same two-stage
#     physics can be set up in a custom file without prior knowledge of how
#     HydrothermalProblemBuilder constructs its pressure stage internally.
#
# See the file's own docstring for the full technical explanation.

#custom_input_path = (cwd + "/examples/synthetic_examples/model2/input_data/simulation_files/"
#                      "custom_hydrothermal.py")

#with open(custom_input_path) as f:
#    custom_input_file_contents = f.read()

#custom_builder = CustomSfepyBuilder(
#    input_file_contents=custom_input_file_contents,
#    mesh_results=mesh_explicit_unstructured,
#    geomodel_result=structural_model_result,
#    fault_zone_n_voxels=1,
#)

#%%
# Pre-flight check: confirms rock_properties landed on the right cells.
#plot_builder_materials(custom_builder)
#plot_builder_materials(
#    builder=custom_builder,
#    property_name="porosity",
#    show_plotter=True,
#)
#
#custom_data_by_time = run_simulation_sfepy(custom_builder, stream_output=True)

#%%

# Visualize simulation results
#custom_final_time = max(custom_data_by_time.nodes_by_time.keys())
#plot_variable_at_a_time(custom_data_by_time, "T", time=0, cmap="coolwarm", show_edges=False)
#plot_variable_at_a_time(custom_data_by_time, "T", time=custom_final_time, cmap="coolwarm", show_edges=False)

# normal=(0, 1, 0): the fault plane is Y-invariant, so this actually cuts across it.
#plot_cross_section_2D(
#    custom_data_by_time, "T", origin=(1250, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
#)




# ------------------------------------------------------------------------------
# END SECTION 4: SFEPY - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
#%%

# ==============================================================================
# SECTION 5: GOLEM - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
# Run GolemHydrothermalProblemBuilder on the standard unstructured mesh.
# ------------------------------------------------------------------------------

# Run the hydrothermal model using GOLEM with the same material objects.
# GOLEM represents the fault as the actual 2D element block. The intrinsic
# properties below match the SfePy damage-zone values, while aperture supplies
# the lower-dimensional scaling required by GOLEM.
#golem_fault_properties = GolemFaultProperties(
#    name="fault",
#    porosity=fault_zone_properties.porosity,
#    permeability=fault_zone_properties.permeability,
#    thermal_conductivity=fault_zone_properties.thermal_conductivity,
#    density=fault_zone_properties.density,
#    heat_capacity=fault_zone_properties.heat_capacity,
#    fluid_modulus=fault_zone_properties.fluid_modulus,
#    aperture=40,
#    permeability_model="constant",
#)

#problem_golem = GolemHydrothermalProblemBuilder(
    # Exact GOLEM faults require the explicit unstructured mesh containing
    # the exported 2D fault element block(s).
#    mesh_results=mesh_explicit_unstructured,
#    geomodel_result=structural_model_result,
#    fluid=fluid,

#    fault_properties=golem_fault_properties,

#    print_linear_residuals=True,
#    use_supg=True,
#    rock_properties=rock_properties,
#    p_top=0.0,
#    p_bottom=1.0e6,
#    t_top=10.0,
#    t_bottom=60.0,
#    initial_pressure=0.0,
#    # Omitted/None: calculate the 60 -> 10 C linear profile from z.
#    initial_temperature=None,
#    start_time=0.0,
#    end_time=5.0e11,
#    num_steps=3,
#)
# Plot material assignments BEFORE running GOLEM
#plot_builder_materials(problem_golem, show_edges=True)
#plot_builder_materials(
#    builder=problem_golem,
#    property_name="porosity",
#    show_plotter=True,
#)

# copy the input file to a diectory
#problem_golem.build_input_file("/tmp/wbgeo_golem_model2.i")

# Run the constructed problem in GOLEM's Docker image. The runner creates the
# temporary input file automatically, so no separate build_input_file() call
# is required here.
#problem_golem_results = run_simulation_golem(
#    problem=problem_golem,
#    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#    workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#    output_dir=(
#        cwd
#        + "/examples/synthetic_examples/model2/"
#          "golem_auto_output"
#    ),

#    nproc=4,
#    stream_output=True,  # Show live GOLEM output (opt-in)
#)

# Results

#print("GOLEM simulation completed")
#final_time = max(problem_golem_results.nodes_by_time.keys())
#initial_time = min(problem_golem_results.nodes_by_time.keys())

#print("Loaded times:", sorted(problem_golem_results.nodes_by_time.keys()))
#print("GOLEM point variables:", list(problem_golem_results.node_data_by_time[final_time].keys()))
#print("GOLEM cell variables:", list(problem_golem_results.cell_data_by_time[final_time].keys()))

# Visualize GOLEM's native result names.
#plot_variable_at_a_time(problem_golem_results, "temperature", time=initial_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(problem_golem_results, "temperature", time=final_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(problem_golem_results, "pore_pressure", time=final_time, cmap="coolwarm", show_edges=True)
#plot_cross_section_2D(
#    problem_golem_results, "temperature", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
#)

# save the outputs as zip file
#buf = export_simulation_results(problem_golem_results, format="exodus")

#with open("simulation_golem_results.zip", "wb") as f:
#    f.write(buf.getvalue())


# ------------------------------------------------------------------------------
# END SECTION 5: GOLEM - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
#%%

# ==============================================================================
# SECTION 6: GOLEM - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
# Run the custom_hydrothermal.i to re-produce the same results as the GolemHydrothermalProblemBuilder.
# ------------------------------------------------------------------------------

# Running golem from a costum file
#custom_input_path = (
#    cwd
#    + "/examples/synthetic_examples/model2/input_data/simulation_files/"
#      "custom_hydrothermal.i"
#)

#with open(custom_input_path, encoding="utf-8") as f:
#    custom_input_file_contents = f.read()

# Create the custom GOLEM problem
#custom_golem_problem = CustomGolemBuilder(
#    input_file_contents=custom_input_file_contents,
#    mesh_results=mesh_explicit_unstructured,
#    geomodel_result=structural_model_result,
#    print_linear_residuals=True,
#)
# Plot material assignments BEFORE running GOLEM
#plot_builder_materials(custom_golem_problem, show_edges=True)

#plot_builder_materials(
#    builder=custom_golem_problem,
#    property_name="porosity_initial",
#    show_plotter=True,
#)
# Run the GOLEM simulation
#custom_golem_results = run_simulation_golem(
#    problem=custom_golem_problem,

#    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",

#    workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
#    output_dir=(
#        cwd
#        + "/examples/synthetic_examples/model2/"
#          "golem_custom_output"
#    ),

#    nproc=4,
#)

#custom_initial_time = min(custom_golem_results.nodes_by_time.keys())
#custom_final_time = max(custom_golem_results.nodes_by_time.keys())

# Visualize GOLEM's native result names.
#plot_variable_at_a_time(custom_golem_results, "temperature", time=custom_initial_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(custom_golem_results, "temperature", time=custom_final_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(custom_golem_results, "pore_pressure", time=custom_final_time, cmap="coolwarm", show_edges=True)
#plot_cross_section_2D(
#    custom_golem_results, "temperature", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
#)


# ------------------------------------------------------------------------------
# END SECTION 6: GOLEM - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
#%%

# ==============================================================================
# SECTION 7: GOLEM - GENERATED SIMPLE-RESERVOIR MODEL
# ==============================================================================
# Automatically create the reservoir extent, two synthetic planes, well, mesh, and GOLEM input.
# ------------------------------------------------------------------------------


# This builder creates two synthetic fracture planes + well for a
# faulted or non-faulted structural model. Geological faults already present in
# structural_model_result are detected automatically and use the same intrinsic
# material values as the 3D reservoir.



#reservoir_problem_model2 = GolemSimpleReservoirProblemBuilder(
#    geomodel_result=structural_model_result,

#    # Model-2 unstructured meshing inputs
#    mesh_size=50.0,
#    curve_mesh_size=5.0,
#    unstructured_mesh_options={
#        "tolerance": 50,
#    },

#    # Same refinement concept used for Model 1.
#    well_refine_size_min=5.0,
#    well_refine_size_max=30.0,
#    well_refine_dist_min=10.0,
#    well_refine_dist_max=60.0,
#    fracture_refine_size_min=5.0,
#    fracture_refine_size_max=30.0,
#    fracture_refine_dist_min=10.0,
#    fracture_refine_dist_max=60.0,

#    # ONE common intrinsic rock material for all 3D blocks AND geological faults.
#    rock_porosity=0.10,
#    rock_permeability=1.0e-15,
#    rock_density=2.6e3,
#    rock_thermal_conductivity=3.0,
#    rock_heat_capacity=950.0,

#    # Geological fault: same rock properties above; this is only the 2D
#    # aperture/scaling needed by GOLEM for a lower-dimensional block.
#    geological_fault_aperture=1.0,

#    # The two synthetic reservoir fractures retain their special properties.
#    fracture_porosity=1.0,
#    fracture_aperture=1.0e-4,

#    well_porosity=1.0,
#    well_permeability=3.12e-4,
#    well_scaling_factor=0.05,

#    fluid_density=1.0e3,
#    fluid_viscosity=1.0e-3,
#    fluid_heat_capacity=4.2e3,
#    fluid_thermal_conductivity=0.65,
#    fluid_modulus=2.5e9,

#    initial_pressure=1.0e7,
#    initial_temperature=150.0,
#    boundary_pressure=1.0e7,
#    boundary_temperature=150.0,
#    injection_temperature=70.0,

#    injection_rate=3.0,
#    extraction_rate=3.0,
#)

# Mesh is created inside the builder.
#mesh_model2_reservoir = reservoir_problem_model2.mesh_results

#print("Model extent:", reservoir_problem_model2.model_extent)
#print("Reservoir extent:", reservoir_problem_model2.reservoir_extent)
#print("3D reservoir blocks:", reservoir_problem_model2.volume_blocks)
#print("2D geological fault blocks:", reservoir_problem_model2.geological_fault_blocks)
#print("2D synthetic fracture blocks:", reservoir_problem_model2.fracture_blocks)
#print("1D well blocks:", reservoir_problem_model2.well_blocks)




# The mesh and mesh genrating parameters and input file for golem are available from the builder if you want to save them:
#mesh_model1_simple_reservoir = reservoir_problem_model2.mesh_results
#reservoir_problem_model2.build_input_file(cwd + "/examples/synthetic_examples/model2/model2_generated_reservoir.i")
#reservoir_problem_model2.build_mesh_generation_file(cwd + "/examples/synthetic_examples/model2/model2_generated_reservoir_mesh.py")

# run the model in Golem docker image.
#reservoir_results_model2 = run_simulation_golem(
#    problem=reservoir_problem_model2,
#    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
#    output_dir=(
#        cwd + "/examples/synthetic_examples/model2/golem_universal_reservoir_output"
#    ),
#    nproc=4,
#)

# Visualisation of results
#reservoir_initial_time = min(reservoir_results_model2.nodes_by_time.keys())
#reservoir_final_time = max(reservoir_results_model2.nodes_by_time.keys())

#plot_cross_section_2D(
#    reservoir_results_model2,
#    "temperature",
#    origin=reservoir_problem_model2.injection_point,
#    normal=(0, 1, 0),
#    cmap="coolwarm",
#)
#plot_cross_section_2D(
#    reservoir_results_model2,
#    "pore_pressure",
#    origin=reservoir_problem_model2.injection_point,
#    normal=(0, 1, 0),
#    cmap="coolwarm",
#)

# NOTE: You can use resourvor_visualisation.pvsm in Paraview to visualize the results.


# ------------------------------------------------------------------------------
# END SECTION 7: GOLEM - GENERATED SIMPLE-RESERVOIR MODEL
# ==============================================================================
#%%

# ==============================================================================
# SECTION 8: GOLEM - CUSTOM SIMPLE-RESERVOIR INPUT
# ==============================================================================
# Explicitly create a separate reservoir mesh, then solve custom_reservoir.i.
# ------------------------------------------------------------------------------


# Simple reservoir solved with a custom GOLEM input file.
# The mesh is created explicitly here (not through GolemSimpleReservoirProblemBuilder).

# Create unstructured mesh.

model2_reservoir_extent = (0, 2500, 0, 1000, 0, 1000)

# Match GolemSimpleReservoirProblemBuilder exactly:
# - full Model-2 domain: x = 0..2500
# - preserve original absolute plane spacing: 200 m
# - center the plane pair in x, giving equal lateral margins
# - preserve the original 100 x 100 m plane size in y/z
#
# Domain center x = 1250 m -> planes at 1150 m and 1350 m.
model2_reservoir_fracture_planes = [
    (
        1150.0, 450.0, 450.0,
        1150.0, 550.0, 450.0,
        1150.0, 550.0, 550.0,
        1150.0, 450.0, 550.0,
    ),
    (
        1350.0, 450.0, 450.0,
        1350.0, 550.0, 450.0,
        1350.0, 550.0, 550.0,
        1350.0, 450.0, 550.0,
    ),
]

# Preserve the builder well geometry relative to the second fracture plane:
# x offsets = -25, -12.5, 0, +12.5, +25 m
# z offsets = -10, -5, 0, +5, +10 m
model2_reservoir_wells = [
    (
        1325.0, 500.0, 490.0,
        1337.5, 500.0, 495.0,
        1350.0, 500.0, 500.0,
        1362.5, 500.0, 505.0,
        1375.0, 500.0, 510.0,
    )
]

# Local mesh refinement around the well and the two synthetic planes.
model2_reservoir_refinement = Refinement(
    wells=LinearWellRefinement(
        SizeMin=5.0,
        SizeMax=30.0,
        DistMin=10.0,
        DistMax=60.0,
    ),
    extra_planes=ExtraPlaneRefinement(
        SizeMin=5.0,
        SizeMax=30.0,
        DistMin=10.0,
        DistMax=60.0,
    ),
)

# Build a NEW mesh specifically for the custom reservoir simulation.

mesh_model2_custom_reservoir = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    extent=model2_reservoir_extent,
    wells=model2_reservoir_wells,
    extra_planes=model2_reservoir_fracture_planes,

    tolerance=50,
    mesh_size=50.0,
    curve_mesh_size=5.0,


    mapping_litho="automatic_centers",
    refinement=model2_reservoir_refinement,
)

# Optional inspection of the custom reservoir mesh.
plot_mesh_3d(
    mesh_model2_custom_reservoir,
    structural_model_result,
    show_plotter=True,
)

# Solve the problem using the custom GOLEM input file.
#
# custom_reservoir.i was originally written for the old Exodus block layout.
# The code below remaps its "block = ..." assignments to the actual full-domain
# Model-2 mesh layout before CustomGolemBuilder validates the input.
model2_reservoir_input_path = (
    cwd
    + "/examples/synthetic_examples/model2/input_data/simulation_files/"
      "custom_reservoir.i"
)

with open(model2_reservoir_input_path, encoding="utf-8") as stream:
    model2_reservoir_input = stream.read()



# Solve the custom reservoir problem with GOLEM.
model2_custom_reservoir_results = run_simulation_golem(
    problem=CustomGolemBuilder(
        input_file_contents=model2_reservoir_input,
        mesh_results=mesh_model2_custom_reservoir,
        geomodel_result=structural_model_result,
        print_linear_residuals=True,
    ),
    image_name="registry.git.rwth-aachen.de/wbgeo/container-registry/golem:latest",
    workspace_dir=os.environ.get("GOLEM_PROJECT_DIR", "./projects"),
    output_dir=(
        cwd
        + "/examples/synthetic_examples/model2/"
          "golem_custom_reservoir_output"
    ),
    nproc=4,
    stream_output=True,  # Show live GOLEM output (opt-in)
)

# Visualisation of results.
model2_reservoir_initial_time = min(
    model2_custom_reservoir_results.nodes_by_time.keys()
)
model2_reservoir_final_time = max(
    model2_custom_reservoir_results.nodes_by_time.keys()
)

plot_cross_section_2D(
    model2_custom_reservoir_results,
    "temperature",
    origin=(1150.0, 500.0, 500.0),
    normal=(0, 1, 0),
    cmap="coolwarm",
)

plot_cross_section_2D(
    model2_custom_reservoir_results,
    "pore_pressure",
    origin=(1150.0, 500.0, 500.0),
    normal=(0, 1, 0),
    cmap="coolwarm",
)

# NOTE: You can use resourvor_visualisation.pvsm in Paraview to visualize the results.

# ------------------------------------------------------------------------------
# END SECTION 8: GOLEM - CUSTOM SIMPLE-RESERVOIR INPUT
# ==============================================================================

# =============================== END OF MODEL 2 ===============================
