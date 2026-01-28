# Importing necessary libraries
import pandas as pd
import os

import core.structuralmodeling_components.general_faults
from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)

from core.structuralmodeling_components import general, general_faults

#%%

cwd = os.getcwd()

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(50, 50, 50)  # Example resolution
)

#%%

# Input data for elements
data_elements = InputData_StructuralElements(name='Model_7',
                                             mapping_object={
                                                 "Top": ('UnitD', 'UnitC'),
                                                 "Bot": ('UnitB', 'UnitA')
                                             },
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/input_data/model7_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/input_data/model7_orientations_df.csv")
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(data_elements.mapping_object,
                                       grid,
                                       data_elements.surface_points,
                                       data_elements.orientations)
frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame, axis='y', show_result=False)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Input data for faults
data_faults = InputData_FaultElements(name='Faults_Model_7',
                                      fault_surface_points=pd.read_csv(
                                          cwd + "/examples/input_data/model7_surface_points_df.csv"),
                                      fault_orientations=pd.read_csv(
                                          cwd + "/examples/input_data/model7_orientations_df.csv"))

# Create FaultFrame
fault_frame = general_faults.build_fault_frame(
    fault_surface_points_df=data_faults.fault_surface_points,
    fault_orientations_df=data_faults.fault_orientations,
    fault_names=["FaultA", "faultB", "FaultC"],
    colors=["#A9A9A9", "#696969", "#808080"],
    grid=grid
)

fault_frame.detailed_report()

#%%

# Compute fault domains
general_faults.compute_fault_domains(fault_frame)

#%%

# Plot fault domains (2D and 3D possible)
fault_frame.plot_fault_domain_section(axis='y', index=12)
plot_fault_frame_3D(fault_frame)

#%%

# Set interpolation methods for each stratigraphic series

# UCK
frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# OK
# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
# frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
# frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

# RBF
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")
# frame["Top"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)
# frame["Bot"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)

# GeoINR
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")

# Loop Structural
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")


frame.detailed_report()

#%%

# Component 2 --> Component 3: Interpolation to geomodel result
general.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                         fault_frame=None,
                         axis='y',
                         show_input_data=True,
                         index=0)

#%%

plot_structural_model_3D(frame=frame,
                         fault_frame=None,
                         mesh_type="masked",
                         show_orientations=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)























import numpy as np
import pandas as pd

import core.structuralmodeling_components.general_faults
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)
from core.structuralmodeling_components import general

#%%

# Data generation

def generate_vertical_fault_data(x_pos: float, name: str, y_range=(100, 900), z_range=(100, 900), n_points=10):
    """
    Generate synthetic surface points and orientations for a vertical fault plane at x = x_pos.
    """
    y_vals = np.linspace(*y_range, n_points)
    z_vals = np.linspace(*z_range, n_points)

    surface_points = pd.DataFrame({
        "X": np.full(n_points, x_pos),
        "Y": y_vals,
        "Z": z_vals,
        "formation": name
    })

    orientations = pd.DataFrame({
        "X": np.full(n_points, x_pos),
        "Y": y_vals,
        "Z": z_vals,
        "G_x": np.ones(n_points),  # Fault normal points in +X
        "G_y": np.zeros(n_points),
        "G_z": np.zeros(n_points),
        "formation": name
    })

    return surface_points, orientations


def generate_horizontal_fault_data(z_pos: float, name: str, x_range=(100, 900), y_range=(100, 900), n_points=10):
    """
    Generate synthetic surface points and orientations for a vertical fault plane at x = x_pos.
    """
    y_vals = np.linspace(*y_range, n_points)
    x_vals = np.linspace(*x_range, n_points)

    surface_points = pd.DataFrame({
        "X": x_vals,
        "Y": y_vals,
        "Z": np.full(n_points, z_pos),
        "formation": name
    })

    orientations = pd.DataFrame({
        "X": x_vals,
        "Y": y_vals,
        "Z": np.full(n_points, z_pos),
        "G_x": np.zeros(n_points),
        "G_y": np.zeros(n_points),
        "G_z": np.ones(n_points),
        "formation": name
    })

    return surface_points, orientations


#%%

# Generate fault input_data
sp_a, ori_a = generate_vertical_fault_data(x_pos=250, name="FaultA")
sp_b, ori_b = generate_vertical_fault_data(x_pos=750, name="FaultB")
sp_c, ori_c = generate_vertical_fault_data(x_pos=500, name="FaultC")
# sp_c, ori_c = generate_horizontal_fault_data(z_pos=500, name="FaultC")

# Combine into full DataFrames
fault_surface_points_df = pd.concat([sp_a, sp_b, sp_c], ignore_index=True)
fault_orientations_df = pd.concat([ori_a, ori_b, ori_c], ignore_index=True)

#%%

grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(25, 50, 50)  # Example resolution
)

fault_names = ["FaultC", "FaultA", "FaultB"]  # FaultB is younger than FaultA
fault_colors = ["#A9A9A9", "#A9A9A9", "#A9A9A9"]

fault_frame = core.structuralmodeling_components.general_faults.build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=fault_names,
    colors=fault_colors,
    grid=grid
)



#%%
fault_frame.detailed_report()
fault_frame.compute_fault_domains()

#%%

# Plot the fault meshes using pyvista
plot_fault_frame_3D(fault_frame)

#%%
fault_frame.plot_fault_domain_section(axis='y', index=25)

#%%

# Setup
elements = ["UnitA", "UnitB", "UnitC", "UnitD"]
fault_blocks = [0, 1, 2, 3]
block_offsets = {0: 0, 1: 100, 2: 200, 3: 300}
base_z = {"UnitA": 100, "UnitB": 200, "UnitC": 300, "UnitD": 400}
dip_gradient = 0.3  # Inclination for UnitC and UnitD

surface_data = []
orientation_data = []

for elem in elements:
    for block in fault_blocks:
        z_offset = block_offsets[block]
        for i in range(2):  # 2 surface points per block
            x = 100 + 50 * i + np.random.uniform(-2, 2) + block * 250
            y = 100 + np.random.uniform(-5, 5)
            z = base_z[elem] + z_offset
            if elem in ["UnitC", "UnitD"]:
                z += dip_gradient * x  # add inclination
            surface_data.append([x, y, z, elem])

        # One orientation per block
        x_ori = 105 + block * 250
        y_ori = 105
        z_ori = base_z[elem] + z_offset
        if elem in ["UnitC", "UnitD"]:
            z_ori += dip_gradient * x_ori
            normal = [0, 0, 1]
            normal = [-dip_gradient, 0, 1]  # simple slope in x
        else:
            normal = [0, 0, 1]

        orientation_data.append([x_ori, y_ori, z_ori, *normal, elem])

# Create DataFrames
structural_surface_points_df = pd.DataFrame(surface_data, columns=["X", "Y", "Z", "formation"])
structural_orientations_df = pd.DataFrame(orientation_data, columns=["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"])

#%%

# Create a StructuralFrame
frame = general.build_structural_frame({"Top": ('UnitD', 'UnitC'), "Bot": ('UnitB', 'UnitA')},
                                       grid,
                                       structural_surface_points_df,
                                       structural_orientations_df)
frame.detailed_report()

#%%

# combine fault dfs and structural dfs if needed and export to csv
combined_surface_points_df = pd.concat([structural_surface_points_df, fault_surface_points_df], ignore_index=True)
combined_orientations_df = pd.concat([structural_orientations_df, fault_orientations_df], ignore_index=True)

#%%

combined_surface_points_df

#%%

combined_surface_points_df.to_csv("model7_surface_points_df.csv", index=False)
combined_orientations_df.to_csv("model7_orientations_df.csv", index=False)

#%%

# TODO: Works after transposing in function
frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# TODO: Kind of works, needs high range
# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
# frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
# frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

# TODO: Works after transposing in function
# frame["Top"].set_interpolation_method("Radial Basis Function")
# frame["Bot"].set_interpolation_method("Radial Basis Function")
# frame["Top"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)
# frame["Bot"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)

# TODO: Works like a charm
# frame["Top"].set_interpolation_method("Loop Structural")
# frame["Bot"].set_interpolation_method("Loop Structural")

# TODO: Works after transposing in function
# frame["Top"].set_interpolation_method("GeoINR")
# frame["Bot"].set_interpolation_method("GeoINR")


#%%

frame.detailed_report()


#%%

general.compute_structural_model(
    frame,
    fault_frame=fault_frame,                     # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_2D(frame=frame,
                        fault_frame=fault_frame,
                        axis='y',
                        show_input_data=True,
                        index=0)

#%%

plot_structural_model_3D(frame=frame,
                        fault_frame=fault_frame,
                        mesh_type="masked",
                        show_orientations=True)

#%%

frame.plot_scalar_field_section(group_nr=1, axis='y', index=12)

#%%

frame.plot_age_mask_section(group_nr=0, axis='y', index=12)





