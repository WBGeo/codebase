# Uncomment the two lines below to enable detailed log output from all WBGeo components.
# import logging
# logging.basicConfig(level=logging.DEBUG)

# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_FaultElements
from core.structural_modeling_components import general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_fault_model_2D, plot_fault_model_3D, plot_fault_input_data_3D)

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 4: 2 faults, no unconformities, 1 stratigraphic group

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),
    resolution=(100, 50, 50)
)

#%%

# Create input data for the fault elements
data_faults = InputData_FaultElements(name='Faults_Model_4',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model4/input_data/geological_data/model4_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/synthetic_examples/model4/input_data/geological_data/model4_orientations_df.csv"),
                                      fault_names=['fault1', 'fault2']
                                      )

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

# Compute fault model result.
#
# fault1 and fault2 genuinely cross-cut each other in this model (their
# isovalue bands overlap -- confirmed by compute_fault_domains' own crosscut
# check). Cross-cutting faults create grid cells whose domain membership
# would otherwise be contradictory (claimed by more than one fault side
# simultaneously), which the domain-splitting algorithm cannot resolve --
# so domaining (domain_map, domain_masks, and each fault's domain_pairs/
# separated_domains) is skipped entirely and left at its None defaults.
# Fault surfaces are still extracted either way, so the result remains fully
# inspectable; this no longer raises (it used to, before cross_cutting was
# added to FaultModelResults).
fault_model_result = general_faults.compute_fault_domains(fault_frame)

print(f"cross_cutting: {fault_model_result.cross_cutting}")
assert fault_model_result.cross_cutting is True
assert fault_model_result.fault_frame.domain_map is None

#%%

# Inspect the cross-cutting result directly in 3D. plot_fault_model_3D
# highlights the exact line where fault1 and fault2's surfaces actually
# cross, in red, by default (highlight_crosscuts=True) -- this is purely
# geometric (the real surface-surface intersection, not a voxel
# approximation), so it only ever draws something when two surfaces
# genuinely intersect. No translucent domain blocks are shown, since there's
# no domain_map to color them from.
plot_fault_model_3D(fault_model_result.fault_frame)

#%%

# Inspect the same result in 2D. Domain-map coloring is skipped (and warns,
# since domain_map is None), but the fault contours -- including right where
# fault1 and fault2 cross the slice plane -- still draw normally, since they
# only need each fault's own scalar_field/scalar_value, not the domain map.
plot_fault_model_2D(fault_model_result.fault_frame)

#%%

# A cross-cutting fault model cannot be wired into Compute Structural Model --
# this is enforced both as a Workbench input_checks pre-check (so the
# Workbench can refuse the connection before the node is ever run) and here,
# directly in code, as a clear, specific ValueError.
from core.structural_modeling_components.general_faults import check_fault_model_not_crosscutting

try:
    check_fault_model_not_crosscutting(fault_model_result)
except ValueError as exc:
    print(f"Rejected as expected: {exc}")
