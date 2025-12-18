import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.grids.grid_classes import RegularGrid
from core.visualization_components_new import visualize_fault_frame

from core.interpolator_components.interpolators_per_group import general_updated

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

# Generate fault data
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
    resolution=(50, 50, 50)  # Example resolution
)

fault_names = ["FaultC", "FaultA", "FaultB"]  # FaultB is younger than FaultA
fault_colors = ["#A9A9A9", "#A9A9A9", "#A9A9A9"]

fault_frame = general_updated.build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=fault_names,
    colors=fault_colors,
    grid=grid
)

#%%
fault_frame.detailed_report()

#%%

# Compute result for fault frame
fault_frame.generate_fault_domains()

#%%

# Plot the fault meshes using pyvista
visualize_fault_frame(fault_frame)


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
frame = general_updated.build_structural_frame({"Top": ('UnitD', 'UnitC'), "Bot": ('UnitB', 'UnitA')},
                                       np.array([0, 1000, 0, 1000, 0, 1000]),
                                       np.array([50, 50, 50]),
                                       structural_surface_points_df,
                                       structural_orientations_df)
frame.detailed_report()


#%%

# TODO: Works after transposing in function
#frame["Top"].set_interpolation_method("Universal Co-Kriging")
#frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# TODO: Kind of works, needs high range
frame["Top"].set_interpolation_method("Ordinary Kriging")
frame["Bot"].set_interpolation_method("Ordinary Kriging")
frame["Top"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)
frame["Bot"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.1)

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

# TODO: List of still missing things
# TODO: Things that still make this example break
# - DONE! Need to change the interpolator functions so they return the scalar field and the scalar values so that I can set them afterwards per domain

# TODO: Important next steps
# - DONE! What happens if no faults exist (e.g no fault frame given) - seems to somehow just work, single domain, little tricky with parameters
# - Add a thing to the fault frame, that defines which groups a fault affects.
# - make a proper masking that accounts for this so we can have faults that are older than groups. Best case these groups would then be calculated in one go
# - check if there is enough data for each element in each domain and set a reasonable default what happens if not
# - (so I have the group masks and the domain masks but naming is weird and maybe I need a combined one per group within domain?)
# - Unify grids for both frames
# - Storing Update - I think only one mesh per element is sufficient, if we can store separate fault block meshes as a single mesh,
#   maybe then also only one combined scalar field is necessary, this would really improve the whole storage madness

# TODO: Additional nice to haves
# - Think about a nice structure for the user to understand this full mess
# - update all the plotting functions
# - go through the whole code for cleaning and consistent structure
# - also to make all setter and getter functions as consistent as possible
# - mesh types are not actually that different at the moment - no combined scalar filed so no combined mesh possible
# - for combined scalar field I would need consistent scalar values across domains, which atm is only true for interpolators where I actively set this
# - DONE! Make computation efficient - only compute over subgrids per domain? - at least low level this works now, can be turned off if it sucks
# - all the tests

# TODO: Cross functionality
# - see what I need to change so that GFZ can still use it as before


general_updated.combined_interpolator_with_domains(
    frame,
    fault_frame=fault_frame,                     # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

frame.structural_groups[0].scalar_field.dtype

#%%

frame.structural_groups[1].structural_elements[1].scalar_values_by_domain()

#%%

# plot slice of final_scalar for debugging
import matplotlib.pyplot as plt
extent = frame.grid.extent[:4]
plt.imshow(frame.structural_groups[1].scalar_field[:, 25, :], cmap='viridis', extent=extent, origin="lower")
plt.colorbar()
plt.contour(frame.structural_groups[1].scalar_field[:, 25, :], colors='black',
                   extent=extent, origin="lower", levels=[1,2])
plt.show()

#%%

frame.structural_groups[1].structural_elements[1].id

#%%

def plot_scalar_field_section(scalar_field, grid, axis='y', index=0):
    """
    Plot a section of a scalar field along a specified axis at a given index.
    """
    if axis == 'y':
        data_slice = scalar_field[:, index, :]
        extent = grid.extent[:4]
    elif axis == 'x':
        data_slice = scalar_field[index, :, :]
        extent = grid.extent[[0, 2, 4, 1]]
    elif axis == 'z':
        data_slice = scalar_field[:, :, index]
        extent = grid.extent[[0, 2, 1, 3]]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    plt.imshow(data_slice, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label='Scalar Value')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f"Scalar Field Section along {axis.upper()} at Index {index}")
    plt.show()

#%%

# Plot a slice of the lithology model
plot_scalar_field_section(frame.get_LithBlock(), frame.grid, index=0)


#%%

# Masks from groups for age relationships
plot_scalar_field_section(frame.structural_groups[1].masks_by_domain()[0], frame.grid, index=25)

#%%

# Same as before but with getter function
plot_scalar_field_section(frame.structural_groups[0].get_mask_for_domain(0), frame.grid, index=25)

#%%

# Masks for fault domains
plot_scalar_field_section(fault_frame.domain_masks[1], frame.grid, index=25)

# %%

plot_scalar_field_section(frame.structural_groups[1].scalar_fields_by_domain()[1], frame.grid, index=25)


#%%

from core.visualization_components_new_new import plot_structural_slice_with_faults

#%%

plot_structural_slice_with_faults(frame=frame, fault_frame=fault_frame, lith_block=lith, axis='y', index=0)

#%%

from core.visualization_components_new_new import visualize_structural_frame_with_faults

#%%

visualize_structural_frame_with_faults(frame=frame, fault_frame=fault_frame)


#%%

fault_frame.detailed_report()

#%%

frame.detailed_report()