from pathlib import Path
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
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemHydrothermalProblemBuilder,
    GolemFaultProperties,
    CustomGolemBuilder,
    GolemSimpleReservoirProblemBuilder,
)

from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials, plot_variable_time_series)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,ExtraPlaneRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement, LinearShaftRefinement, FunctionShaftRefinement, GeneralRefinement)
#%%

from core.hierarchical_simulation.generate_mesh.hierarchical_mesh import hierarchical_mesh
from core.hierarchical_simulation.export_data.exodus.exo_format import export_meshes_exodus
from core.hierarchical_simulation.visualization.hierarchical_meshing_visualization import plot_hierarchical_meshes_3d
from core.hierarchical_simulation.visualization.hierarchical_paraview_visualization import open_hierarchical_exodus_in_paraview
cwd = os.getcwd()
os.environ["GOLEM_EXECUTION_MODE"] = "docker"


# ==============================================================================
# MODEL 1 - SCRIPT MAP
# ==============================================================================
# SECTION 1: Structural geological model
# SECTION 2: Mesh generation and mesh utilities
# SECTION 3: SfePy generated hydrothermal simulation
# SECTION 4: SfePy custom hydrothermal simulation
# SECTION 5: GOLEM generated hydrothermal simulation
# SECTION 6: GOLEM custom hydrothermal simulation
# SECTION 7: GOLEM generated simple-reservoir simulation
# SECTION 8: GOLEM custom simple-reservoir simulation
# SECTION 9: Hierarchical mesh generation and export
#
# Each section starts and ends with a large banner so the different sections are
# visually separated while keeping the existing Model 1 code in one file.
# ==============================================================================

#%%

# ==============================================================================
# SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================
# Build the stratigraphy, structural frame, and structural result.
# ------------------------------------------------------------------------------

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

#%%

# ------------------------------------------------------------------------------
# END SECTION 1: STRUCTURAL GEOLOGICAL MODEL
# ==============================================================================

#%%

# ==============================================================================
# SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================
# Build the implicit structured, explicit unstructured, and explicit structured meshes.
# ------------------------------------------------------------------------------

# Compute 3D meshes based on the structural model result using different meshing approaches.

# Implicit structured mesh
mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Explicit unstructured mesh
mesh_explicit_unstructured = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    mesh_size=30,
    curve_mesh_size=5
)

# Explicit structured mesh
mesh_explicit_structured = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[20,20,20],
    #mesh_division=(25,25)
    )
#%%

# Plot the meshing results
#plot_mesh_3d(mesh_implicit_structured, structural_model_result, show_plotter=True)
# plot_mesh_3d(mesh_explicit_unstructured, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_explicit_structured, structural_model_result, show_plotter=True)

#%%


# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
buf = export_mesh_results_to_exodus(mesh_explicit_structured, type='str')
with open("filename_st1.exo", "wb") as f:
    f.write(buf.getvalue())

buf = export_mesh_results_to_exodus(mesh_explicit_unstructured, type='unstr')
with open("filename_unst1.exo", "wb") as f:
    f.write(buf.getvalue())

buf = export_mesh_results_to_exodus(mesh_implicit_structured, type='imp')
with open("filename_imp1.exo", "wb") as f:
    f.write(buf.getvalue())
#%%

# Optional: Example of how to include objects (only works for unstructured mesh)

# # Load engineering objects
wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_wells.csv")
shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_shafts.csv")
sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_sources.csv")
planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_planes.csv")
ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/model_1_ellipses.csv")
csv_files=(cwd + "/examples/synthetic_examples/Model1/input_data/Engineering_objects/seismic_plane_new_offset_0.csv",
           cwd + "/examples/synthetic_examples/Model1/input_data/Engineering_objects/seismic_plane_new_offset_1.csv")
triangulations= load_triangulations_planes_from_csv(csv_files)

# --- Optional: adaptive mesh refinement around objects ---
# Uncomment, adjust the values, and pass refinement=refinement to create_unstructured_mesh_data.
#
#refinement = Refinement()
#refinement.wells = LinearWellRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
#refinement.shafts = LinearShaftRefinement(SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0)
#refinement.sources = FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))")
#refinement.triangulations = TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0)
#refinement.ellipses = EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0)
#refinement.extra_planes = ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0)
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
refinement = Refinement(
    wells=LinearWellRefinement( SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0,),
    shafts=LinearShaftRefinement( SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0,),
    sources=FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))",),
    triangulations=TriangulationRefinement(SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0,),
    ellipses=EllipseRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0,),
    extra_planes=ExtraPlaneRefinement(SizeMin=10.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0,),
    general=[
        GeneralRefinement(coordinates=[(500.0, 400.0, 0.0),], SizeMin=4.0, DistMax=300.0,),
        GeneralRefinement(coordinates=[(0.0, 0.0, 0.0),(1000.0, 0.0, 0.0),], expression="2 + 85*(1 - exp(-DIST/100))",),],)


# # Explicit unstructured mesh with objects
#mesh_unstructured_with_objects = create_unstructured_mesh_data(
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
#     gmsh_flag= True,
#         # to save original gmsh configuration (defaut is False)
#     mapping_litho='manual', # it can be 'manual', 'automatic_centers', 'automatic_corners' or 'none' (default: automatic_centers)
#     merge_file = cwd + "/examples/synthetic_examples/model1/input_data/block_groups.csv",  # if mapping_litho='manual'
#     refinement=refinement,
#)

# save the mesh
#buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type='unstr')
#with open("filename_model1___mesh.exo", "wb") as f:
#     f.write(buf.getvalue())

#buf = export_mesh_results_to_exodus(mesh_implicit_structured, type='imp')
#with open("filename_model1__im_mesh.exo", "wb") as f:
#     f.write(buf.getvalue())
#
# # Plot the resulting mesh
# plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, show_plotter=True)

#%%

#%%

# ------------------------------------------------------------------------------
# END SECTION 2: MESH GENERATION AND MESH UTILITIES
# ==============================================================================

#%%

# ==============================================================================
# SECTION 3: SFEPY - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
# Run HydrothermalProblemBuilder on the Model 1 mesh.
# ------------------------------------------------------------------------------

# Process simulation with SfePy on the implicit structured mesh.

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
    print_linear_residuals=True,
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
plot_variable_time_series(data_by_time, "T", point=(500,500,500))
#%%

#%%

# ------------------------------------------------------------------------------
# END SECTION 3: SFEPY - GENERATED HYDROTHERMAL MODEL
# ==============================================================================

#%%

# ==============================================================================
# SECTION 4: SFEPY - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
# Run a hand-written SfePy input file for the same Model 1 hydrothermal workflow.
# ------------------------------------------------------------------------------

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
                      "custom_hydrothermal.py")

with open(custom_input_path) as f:
    custom_input_file_contents = f.read()

custom_builder = CustomSfepyBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    print_linear_residuals=True,

)

#%%

custom_data_by_time = run_simulation_sfepy(custom_builder)

#%%

# Confirms the custom-builder result matches HydrothermalProblemBuilder's above.
custom_final_time = max(custom_data_by_time.nodes_by_time.keys())
print("HydrothermalProblemBuilder T mean:", data_by_time.node_data_by_time[final_time]["T"].mean())
print("CustomSfepyBuilder        T mean:", custom_data_by_time.node_data_by_time[custom_final_time]["T"].mean())
# Visualize results.
plot_variable_at_a_time(custom_data_by_time, "T", time=0, cmap="coolwarm", show_edges=True)
plot_variable_at_a_time(custom_data_by_time, "T", time=custom_final_time, cmap="coolwarm", show_edges=True)
plot_cross_section_2D(custom_data_by_time, "T", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm")




#%%

# ------------------------------------------------------------------------------
# END SECTION 4: SFEPY - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================

#%%

# ==============================================================================
# SECTION 5: GOLEM - GENERATED HYDROTHERMAL MODEL
# ==============================================================================
# Run GolemHydrothermalProblemBuilder on the Model 1 mesh.
# ------------------------------------------------------------------------------

# Run the hydrothermal model using GOLEM with the same material objects.
##problem_golem = GolemHydrothermalProblemBuilder(
##    mesh_results=mesh_explicit_unstructured,
##    geomodel_result=structural_model_result,
##    fluid=fluid,
##    print_linear_residuals=True,
##    rock_properties=rock_properties,
##    p_top=0.0,
##    p_bottom=1.0e6,
##    t_top=10.0,
##    t_bottom=60.0,
##    initial_pressure=0.0,
##    # Omitted/None: calculate the 60 -> 10 C linear profile from z.
##    initial_temperature=None,
##    start_time=0.0,
##    end_time=5.0e11,
##    dt=5.0e11 / 3,
##    num_steps=3,
##)

# copy the input file to a diectory
##problem_golem.build_input_file("/tmp/wbgeo_golem_model1.i")

# Run the constructed problem in GOLEM's Docker image. The runner creates the
# temporary input file automatically, so no separate build_input_file() call
# is required here.
##problem_golem_results = run_simulation_golem(
##    problem=problem_golem,
##    image_name="moose-dev:fixed",
##    project_dir=os.environ["GOLEM_PROJECT_DIR"],
##    docker_dir="/projects",

##    output_dir=(
##        cwd
##        + "/examples/synthetic_examples/model1/"
##          "golem_auto_output"
##    ),

##    nproc=4,
##)

# Results

#print("GOLEM simulation completed")
##final_time = max(problem_golem_results.nodes_by_time.keys())
##initial_time = min(problem_golem_results.nodes_by_time.keys())

##print("Loaded times:", sorted(problem_golem_results.nodes_by_time.keys()))
##print("GOLEM point variables:", list(problem_golem_results.node_data_by_time[final_time].keys()))
##print("GOLEM cell variables:", list(problem_golem_results.cell_data_by_time[final_time].keys()))

# Visualize GOLEM's native result names.
##plot_variable_at_a_time(problem_golem_results, "temperature", time=initial_time, cmap="coolwarm", show_edges=True)
##plot_variable_at_a_time(problem_golem_results, "temperature", time=final_time, cmap="coolwarm", show_edges=True)
##plot_variable_at_a_time(problem_golem_results, "pore_pressure", time=final_time, cmap="coolwarm", show_edges=True)
##plot_cross_section_2D(
##    problem_golem_results, "temperature", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
##)




#%%

# ------------------------------------------------------------------------------
# END SECTION 5: GOLEM - GENERATED HYDROTHERMAL MODEL
# ==============================================================================

#%%

# ==============================================================================
# SECTION 6: GOLEM - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================
# Run the custom GOLEM hydrothermal input for Model 1.
# ------------------------------------------------------------------------------

from core.simulation_components.simulation_visualization.golem_material_preview import (
    plot_golem_materials,
)



# Running hydrothermal problem using golem from a costum file
custom_input_path = (
    cwd
    + "/examples/synthetic_examples/model1/input_data/simulation_files/"
      "custom_hydrothermal.i"
)

with open(custom_input_path, encoding="utf-8") as f:
   custom_input_file_contents = f.read()

# Build the problem without running the solver
custom_problem = CustomGolemBuilder(
    input_file_contents=custom_input_file_contents,
    mesh_results=mesh_explicit_unstructured,
    geomodel_result=structural_model_result,
    print_linear_residuals=True,
)

# Plot material assignments BEFORE running GOLEM
plot_golem_materials(custom_problem, show_edges=True)

custom_golem_results = run_simulation_golem(
    problem=custom_problem,
    image_name="moose-dev:fixed",
    project_dir=os.environ["GOLEM_PROJECT_DIR"],
    docker_dir="/projects",
    output_dir=cwd + "/examples/synthetic_examples/model1/golem_custom_output",
    nproc=4,
    stream_output=True,
)

#custom_initial_time = min(custom_golem_results.nodes_by_time.keys())
#custom_final_time = max(custom_golem_results.nodes_by_time.keys())

# Visualize GOLEM's native result names.
#plot_variable_at_a_time(custom_golem_results, "temperature", time=custom_initial_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(custom_golem_results, "temperature", time=custom_final_time, cmap="coolwarm", show_edges=True)
#plot_variable_at_a_time(custom_golem_results, "pore_pressure", time=custom_final_time, cmap="coolwarm", show_edges=True)
#plot_cross_section_2D(
#    custom_golem_results, "temperature", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm"
#)




#%%

# ------------------------------------------------------------------------------
# END SECTION 6: GOLEM - CUSTOM HYDROTHERMAL INPUT
# ==============================================================================

#%%

# ==============================================================================
# SECTION 7: GOLEM - GENERATED SIMPLE-RESERVOIR MODEL
# ==============================================================================
# Automatically create the reservoir geometry, mesh, GOLEM input, and simulation.
# ------------------------------------------------------------------------------

# Simple reservoir problem solving with builder using Golem

##reservoir_problem = GolemSimpleReservoirProblemBuilder(
##    geomodel_result=structural_model_result,

    # --------------------------------------------------------------
    # Unstructured mesh inputs
    # --------------------------------------------------------------
##    mesh_size=50.0,
##    curve_mesh_size=5.0,
##    mapping_litho="automatic_centers",
##    gmsh_flag=False,
    # z_threshold=None,
    # merge_file=None,
    #
    # Any other create_unstructured_mesh_data() input can be passed as:
    # unstructured_mesh_options={"your_argument": your_value},

    # --------------------------------------------------------------
    # Refinement inputs
    # Defaults reproduce the original simple-reservoir refinement.
    # --------------------------------------------------------------
##    well_refine_size_min=5.0,
##    well_refine_size_max=30.0,
##    well_refine_dist_min=10.0,
##    well_refine_dist_max=60.0,

##    fracture_refine_size_min=5.0,
##    fracture_refine_size_max=30.0,
##    fracture_refine_dist_min=10.0,
##    fracture_refine_dist_max=60.0,

    # Scale refinement lengths when the structural model extent changes.
##    scale_refinement_with_model=True,

    # --------------------------------------------------------------
    # One common material for ALL 3D reservoir blocks
    # --------------------------------------------------------------
##    rock_porosity=0.1,
##    rock_permeability=1.0e-15,
##    rock_density=2.6e3,
##    rock_thermal_conductivity=3.0,
##    rock_heat_capacity=950.0,

    # Fracture
##    fracture_porosity=1.0,
##    fracture_aperture=1.0e-4,

    # Well
##    well_porosity=1.0,
##    well_permeability=3.12e-4,
##    well_scaling_factor=0.05,

    # Fluid
##    fluid_density=1.0e3,
##    fluid_viscosity=1.0e-3,
##    fluid_heat_capacity=4.2e3,
##    fluid_thermal_conductivity=0.65,
##    fluid_modulus=2.5e9,

    # --------------------------------------------------------------
    # Initial / boundary / source inputs from simple_reservoir.i
    # --------------------------------------------------------------
##    initial_pressure=1.0e7,
##    initial_temperature=150.0,
##    boundary_pressure=1.0e7,
##    boundary_temperature=150.0,
##    injection_temperature=70.0,

    # None means: use the automatically scaled reference positions.
##    injection_point=None,
##    extraction_point=None,
##    injection_rate=3.0,
##    extraction_rate=3.0,
##    source_start_time=0.0,
##    source_end_time=None,

    # --------------------------------------------------------------
    # Transient solve
    # 19 solve times + INITIAL -> 20 Exodus states
    # --------------------------------------------------------------
##    start_time=0.0,
##    time_sequence=[
##        1.0,
##        86400.0,
##        604800.0,
##        2592000.0,
##        7776000.0,
##        15552000.0,
##        31536000.0,
##        63072000.0,
##        126144000.0,
##        315360000.0,
##        630720000.0,
##        946080000.0,
##        1261440000.0,
##        1576800000.0,
##        1892160000.0,
##        2207520000.0,
##        2522880000.0,
##        2838240000.0,
##        3153600014.0,
##    ],
##    end_time=None,
##    automatic_scaling=True,
##    compute_scaling_once=False,

##    print_linear_residuals=True,
##    print_nonlinear_converged_reason=True,
##    perf_graph=True,
##)

# The mesh and mesh genrating parameters and input file for golem are available from the builder if you want to save them:
##mesh_model1_simple_reservoir = reservoir_problem.mesh_results
##reservoir_problem.build_input_file("model1_generated_reservoir.i")
##reservoir_problem.build_mesh_generation_file("model1_generated_reservoir_mesh.py")

# Run the simulation using Golem
##model1_simple_reservoir_results = run_simulation_golem(
##    problem=reservoir_problem,
##    image_name="moose-dev:fixed",
##    project_dir=os.environ["GOLEM_PROJECT_DIR"],
##    docker_dir="/projects",
##    output_dir=(
##        cwd
##        + "/examples/synthetic_examples/model1/"
##          "golem_simple_reservoir_output"
##    ),
##    nproc=4,
##)

# Visualisation of results
##reservoir_initial_time = min(model1_simple_reservoir_results.nodes_by_time.keys())
##reservoir_final_time = max(model1_simple_reservoir_results.nodes_by_time.keys())

##plot_cross_section_2D(
##    model1_simple_reservoir_results,
##    "temperature",
##    origin=reservoir_problem.injection_point,
##    normal=(0, 1, 0),
##    cmap="coolwarm",
##)
##plot_cross_section_2D(
##    model1_simple_reservoir_results,
##    "pore_pressure",
##    origin=reservoir_problem.injection_point,
##    normal=(0, 1, 0),
##    cmap="coolwarm",
##)


# NOTE: You can use resourvor_visualisation.pvsm in Paraview to visualize the results.



#%%

# ------------------------------------------------------------------------------
# END SECTION 7: GOLEM - GENERATED SIMPLE-RESERVOIR MODEL
# ==============================================================================

#%%

# ==============================================================================
# SECTION 8: GOLEM - CUSTOM SIMPLE-RESERVOIR INPUT
# ==============================================================================
# Build the Model 1 reservoir mesh explicitly and run the custom GOLEM input.
# ------------------------------------------------------------------------------

# Simple reservoir solving with custom GOLEM input file.
# Use exactly the same geometry as GolemSimpleReservoirProblemBuilder:
# the full structural-model domain, two 100 x 100 vertical planes at x=400
# and x=600 (200 m apart and equally 400 m from the lateral boundaries),
# and the original well translated with/around the second plane.
#reservoir_extent = (0, 1000, 0, 1000, 0, 1000)

# Two vertical 100 x 100 m planes, matching the generated builder geometry.
# Plane separation = 200 m; lateral margins = 400 m on both sides.
#reservoir_fracture_planes = [(
#            400.0, 450.0, 450.0,
#            400.0, 550.0, 450.0,
#            400.0, 550.0, 550.0,
#            400.0, 450.0, 550.0,
#        ),
#        (
#            600.0, 450.0, 450.0,
#            600.0, 550.0, 450.0,
#            600.0, 550.0, 550.0,
#            600.0, 450.0, 550.0,
#        ),
#    ]
# The reference well contains four connected line elements around the
# extraction point. WBGeo accepts each well as one flat x,y,z sequence.
#reservoir_wells = [(
#        575.0, 500.0, 490.0,
#        587.5, 500.0, 495.0,
#        600.0, 500.0, 500.0,
#        612.5, 500.0, 505.0,
#        625.0, 500.0, 510.0,
#    )]

#reservoir_refinement = Refinement(
#        wells=LinearWellRefinement(
#            SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0,
#        ),
#        extra_planes=ExtraPlaneRefinement(
#            SizeMin=5.0, SizeMax=30.0, DistMin=10.0, DistMax=60.0,
#        ),
#    )

#mesh_model1_simple_reservoir = create_unstructured_mesh_data(
#        geomodel_result=structural_model_result,
#        extent=reservoir_extent,
#        wells=reservoir_wells,
#        extra_planes=reservoir_fracture_planes,
#        mesh_size=50.0,
#        curve_mesh_size=5.0,
#        mapping_litho="automatic_centers",
#        refinement=reservoir_refinement,
#    )

# Solving the problem using custom golem input file. The GOLEM input file contains the Exodus material-block
#reservoir_input_path = (
#        cwd
#        + "/examples/synthetic_examples/model1/input_data/simulation_files/model1_simple_reservoir.i"
#    )
#with open(reservoir_input_path, encoding="utf-8") as stream:
#        reservoir_input = stream.read()

# Solving the problem with Golem
#model1_simple_reservoir_results = run_simulation_golem(
#        problem=CustomGolemBuilder(
#            input_file_contents=reservoir_input,
#            mesh_results=mesh_model1_simple_reservoir,
#            geomodel_result=structural_model_result,
#            print_linear_residuals=True,
#        ),
#        image_name="moose-dev:fixed",
#        project_dir=os.environ["GOLEM_PROJECT_DIR"],
#        docker_dir="/projects",
#        output_dir=(
#            cwd
#            + "/examples/synthetic_examples/model1/"
#              "golem_custom_simple_reservoir_output"
#        ),
#        nproc=4,
#    )

# Visualisation of results
#reservoir_initial_time = min(model1_simple_reservoir_results.nodes_by_time.keys())
#reservoir_final_time = max(model1_simple_reservoir_results.nodes_by_time.keys())

#plot_cross_section_2D(model1_simple_reservoir_results, "temperature", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm",)
#plot_cross_section_2D(model1_simple_reservoir_results, "pore_pressure", origin=(500, 500, 500), normal=(0, 1, 0), cmap="coolwarm",)

# NOTE: You can use resourvor_visualisation.pvsm in Paraview to visualize the results.


#%%

# ------------------------------------------------------------------------------
# END SECTION 8: GOLEM - CUSTOM SIMPLE-RESERVOIR INPUT
# ==============================================================================



# ==============================================================================
# SECTION 9: HIERARCHICAL MESH GENERATION AND EXPORT
# ==============================================================================
# Build two nested unstructured meshes, export them, and visualize the hierarchy.
# ------------------------------------------------------------------------------

# Unstructured hierarchical mesh

engineering_objects_dir = (
    cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/"
)

# Load all engineering objects in hierarchical mode.
hierarchical_wells = load_wells_from_csv(
    engineering_objects_dir + "model_1_wells_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_sources = load_sources_from_csv(
    engineering_objects_dir + "model_1_sources_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_mining_shafts = load_shafts_from_csv(
    engineering_objects_dir + "model_1_shafts_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_ellipses = load_ellipses_from_csv(
    engineering_objects_dir + "model_1_ellipses_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_extra_planes = load_planes_from_csv(
    engineering_objects_dir + "model_1_planes_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_triangulation_files = (
    engineering_objects_dir + "seismic_plane_new_offset_0_hierarchical.csv",
    engineering_objects_dir + "seismic_plane_new_offset_1_hierarchical.csv",
)
hierarchical_triangulations = load_triangulations_planes_from_csv(
    hierarchical_triangulation_files,
    key_hierarchical=True,
)

# Refinement
refinement = Refinement()
refinement.wells = LinearWellRefinement(
    SizeMin=20.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.sources = LinearSourceRefinement(
    SizeMin=20.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.shafts = LinearShaftRefinement(
    SizeMin=20.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.ellipses = EllipseRefinement(
    SizeMin=20.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0
)
refinement.extra_planes = ExtraPlaneRefinement(
    SizeMin=20.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0
)
refinement.triangulations = TriangulationRefinement(
    SizeMin=20.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0
)

mesh1, mesh2, nodes_bcs = hierarchical_mesh(
    geomodel_result=structural_model_result,
    mesh_type="Un",

    extent1="whole",
    extent2=[300, 900, 200, 900, 100, 800],


    wells=hierarchical_wells,
    sources=hierarchical_sources,
    shafts=hierarchical_mining_shafts,
    ellipses=hierarchical_ellipses,
    extra_planes=hierarchical_extra_planes,
    triangulations=hierarchical_triangulations,

    # if the engoneering obkjects in both meshhes are the same, one can define only engineering bjects for the bigger model and the
    # program cuts the objects to be inside the smaller one.
    #wells=wells,
    #sources=sources,
    #shafts=shafts,
    #ellipses=ellipses,
    #extra_planes=planes,
    #triangulations=triangulations,
    # unstructured
    mesh_size1=40,
    mesh_size2=30,
    unstructured_params={
        "curve_mesh_size": 5,
        "refinement": refinement,
    },
    unstructured_params2={
        "rotation_angle":30,
    },
)


# Export the hierarchical meshes to Exodus first. ParaView opens these files.
mesh1_exodus = "mesh1r.exo"
mesh2_exodus = "mesh2r.exo"
boundary_nodes_file = "boundary_nodesr"

export_meshes_exodus(
    mesh1,
    mesh2,
    nodes_bcs,
    mesh1_exodus,
    mesh2_exodus,
    boundary_nodes_file,
)

# Open both hierarchical Exodus meshes in one ParaView GUI session.
# ParaView is launched as a separate process, so the Python script continues
# immediately to the WBGeo/PyVista hierarchical visualization below.
open_hierarchical_exodus_in_paraview(
    mesh1_exodus=mesh1_exodus,
    mesh2_exodus=mesh2_exodus,
    paraview_executable=(
        "/Applications/ParaView-5.12.0.app/"
        "Contents/MacOS/paraview"
    ),
)


# Interactive WBGeo/PyVista hierarchical mesh visualization.
plot_hierarchical_meshes_3d(
    mesh1,
    mesh2,
    structural_model_result,
    show_plotter=True,
    link_views=True,
    volume_opacity=1.0,
    show_block_controls=True,
)

#%%

# ------------------------------------------------------------------------------
# END SECTION 9: HIERARCHICAL MESH GENERATION AND EXPORT
# ==============================================================================
