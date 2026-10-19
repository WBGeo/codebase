# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Model 2 on an adaptive OctreeGrid instead of a RegularGrid: fault and structural modeling
# and their plots. Meshing and simulation are not available on an octree yet (the meshing
# components refuse octree results), so this workflow stops after the structural model.

# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    OctreeGrid, GapMitigation, CellBalance)
from core.structural_modeling_components.structural_objects.structural_objects import GeoMeshType, FaultMeshType

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D, plot_fault_model_2D,
plot_fault_input_data_3D, plot_input_data_3D)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 2: 1 fault, 1 unconformity, 2 stratigraphic groups

#%%

# Create an octree grid for the model.
# The extent is split into base_resolution base blocks (level 0); cells are halved down to
# max_level where the model changes. Finest cells: base_resolution * 2**max_level, here
# 160 x 64 x 64 (about 15.6 m), the same as grid.equivalent_regular_grid().
grid = OctreeGrid(
    extent=(0, 2500, 0, 1000, 0, 1000),
    base_resolution=(5, 2, 2),  # base blocks of 500 m
    max_level=5,
    gap_mitigation=GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY,  # or GapMitigation.NONE (surfaces can get gaps)
    cell_balance=CellBalance.NONE,  # or CellBalance.FACE_2TO1 (face neighbours differ by at most one level)
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
plot_fault_model_2D(fault_frame, show_grid=True)
plot_fault_model_3D(fault_frame)

#%%

# Compute fault model result (on the octree: refined to max level along the fault)
fault_model_result = general_faults.compute_fault_domains(fault_frame)

# Octree cells of the fault model per level
fault_model_result.fault_frame.detailed_report()

#%%

# Plot the fault model results (2D and 3D possible)
plot_fault_model_2D(fault_model_result.fault_frame, show_grid=True)
plot_fault_model_3D(fault_model_result.fault_frame)

# Plotting options to play with:
# 2D: section axis ('x', 'y', 'z') and index (along the finest resolution, here 0..159 / 0..63 / 0..63),
#     show_grid=True draws the outlines of the octree cells the section cuts
# plot_fault_model_2D(fault_model_result.fault_frame, axis='y', index=32, show_grid=True)
# plot_fault_model_2D(fault_model_result.fault_frame, axis='z', index=40, show_grid=True)
# 3D: fault mesh type (UNMASKED, EXTENDED), domain blocks from the fault cells
# plot_fault_model_3D(fault_model_result.fault_frame, mesh_type=FaultMeshType.EXTENDED)
# plot_fault_model_3D(fault_model_result.fault_frame, show_domain_map=False)


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

# Create a StructuralFrame and include the fault frame (computed on the same octree grid)
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       fault_model_results=fault_model_result
                                       )

frame.detailed_report()

#%%

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame, show_grid=True)
plot_structural_model_3D(frame, show_lith_block=True)

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
# On an octree: RBF (default), Finite Differences, GeoINR, Ordinary Kriging, Universal Kriging,
# Universal Co-Kriging. Piecewise Linear is not available on an octree yet.
# frame["Strat_Series1"].set_interpolation_method("Ordinary Kriging")
# frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()
# frame["Strat_Series2"].configure_interpolation_params()

# frame.detailed_report()


#%%

# Compute structural model result (surfaces by dual contouring on the octree)
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

# Octree cells of the structural model per level
structural_model_result.structural_frame.detailed_report()

#%%

# Plot the results (2D and 3D possible)
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True, show_lith_block=True)

#%%

# Plotting options to play with:

# 2D sections: axis and index (along the finest resolution), octree cell outlines
# plot_structural_model_2D(structural_model_result.structural_frame, axis='y', index=32, show_grid=True)
# plot_structural_model_2D(structural_model_result.structural_frame, axis='x', index=80, show_grid=True)
# plot_structural_model_2D(structural_model_result.structural_frame, axis='z', index=32, show_grid=True,
#                          show_input_data=False)

# 3D: surface mesh type (MASKED default, UNMASKED, EXTENDED, COMBINED) and fault mesh type
# (MASKED, UNMASKED, EXTENDED, EXTENDED_MASKED)
# plot_structural_model_3D(structural_model_result.structural_frame, mesh_type=GeoMeshType.UNMASKED)
# plot_structural_model_3D(structural_model_result.structural_frame, mesh_type=GeoMeshType.EXTENDED,
#                          fault_mesh_type=FaultMeshType.EXTENDED)
# plot_structural_model_3D(structural_model_result.structural_frame, mesh_type=GeoMeshType.COMBINED)

# 3D: lithology on the cell faces of the six outer walls, with the octree cell edges
# plot_structural_model_3D(structural_model_result.structural_frame, show_lith_block=True,
#                          show_surface_meshes=False, show_points=False, show_orientations=False)

#%%

# Optional: Plotting age masks and scalar fields (index along the finest resolution on an octree)
structural_model_result.structural_frame.plot_scalar_field_section(group_nr=0, axis='y', index=32)
structural_model_result.structural_frame.plot_age_mask_section(group_nr=0, axis='y', index=32)

#%%

# Optional: the same model on the equivalent regular grid, for comparison
# grid_regular = grid.equivalent_regular_grid()
