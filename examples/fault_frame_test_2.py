from typing import List, Optional
from pydantic import BaseModel, PrivateAttr
import numpy as np
import pandas as pd
from core.grids.grid_classes import RegularGrid
import gempy as gp

from core.utility.surface_mesh_extraction import marching_cubes_new, marching_cubes_per_element

from core.visualization_components_new import visualize_fault_frame

from core.structural_objects.objects_updated import StructuralFrame
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

def assign_domain_ids_to_points(grid: RegularGrid, domain_map: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a domain ID to each point based on the domain_map.

    Args:
        grid (RegularGrid): The model grid.
        domain_map (np.ndarray): 3D array with domain IDs (shape: [Z, Y, X]).
        df (pd.DataFrame): DataFrame with 'X', 'Y', 'Z' columns.

    Returns:
        pd.DataFrame: The same DataFrame with a new 'domain_id' column.
    """
    if df.empty:
        df["domain_id"] = pd.Series(dtype=int)
        return df

    coords = df[["X", "Y", "Z"]].values
    indices = grid.xyz_to_indices(coords)  # Shape: [N, 3], order: [X, Y, Z]

    # Clamp to bounds
    for dim in range(3):
        indices[:, dim] = np.clip(indices[:, dim], 0, domain_map.shape[2 - dim] - 1)

    # Reverse the indexing to [Z, Y, X]
    domain_ids = domain_map[indices[:, 2], indices[:, 1], indices[:, 0]]

    df = df.copy()
    df["domain_id"] = domain_ids
    return df


#%%

import numpy as np
import pandas as pd

# Setup
elements = ["UnitA", "UnitB", "UnitC", "UnitD"]
fault_blocks = [0, 1, 2, 3]
block_offsets = {0: 0, 1: 100, 2: 200, 3: 300}
base_z = {"UnitA": 100, "UnitB": 200, "UnitC": 300, "UnitD": 400}
dip_gradient = 0.25  # Inclination for UnitC and UnitD

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

from core.interpolator_components.interpolators_per_group import general_updated
from core.visualization_components_new import visualize_structural_frame, plot_structural_slice

#%%

# Create a StructuralFrame
frame = general_updated.build_structural_frame({"Top": ('UnitD', 'UnitC'), "Bot": ('UnitB', 'UnitA')},
                                       np.array([0, 1000, 0, 1000, 0, 1000]),
                                       np.array([50, 50, 50]),
                                       structural_surface_points_df,
                                       structural_orientations_df)
frame.detailed_report()


#%%

# Plot a slice of the structural model
plot_structural_slice(frame, axis='y', index=10, show_scalar_contours=True)

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")


#%%

frame.detailed_report()

#%%

import numpy as np
import pandas as pd
from copy import deepcopy

from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import \
    interpolate_group_universal_cokriging
from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import \
    interpolate_group_ordinary_kriging

from core.interpolator_components.interpolators_per_group.general import set_scalar_masks, compute_lithology_block

#%%



#%%

from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import (
    interpolate_group_ordinary_kriging,
)
from core.interpolator_components.interpolators_per_group.radial_basis_function_per_group import (
    interpolate_group_radial_basis_function,
)
from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import (
    interpolate_group_universal_cokriging,
)
from core.interpolator_components.interpolators_per_group.geoinr_per_group import (
    interpolate_group_geo_inr,
)
from core.interpolator_components.interpolators_per_group.loop_structural_per_group import (
    interpolate_group_loop_structural,
)
from core.interpolator_components.interpolators_per_group.inverse_distance_per_group import (
    interpolate_group_idw,
)

from core.structural_objects.objects_updated import InterpolationMethod

#%%

# TODO: List of still missing things
# TODO: Things that still make this example break
# - Need to change the interpolator functions so they return the scalar field and the scalar values so that I can set them afterwards per domain

# TODO: Important next steps
# - What happens if no faults exist (e.g no fault frame given)
# - Add a thing to the fault frame, that defines which groups a fault affects.
# - make a proper masking that accounts for this so we can have faults that are older than groups. Best case these groups would then be calculated in one go
# - check if there is enough data for each element in each domain and set a reasonable default what ahppens if not

# TODO: Additional nice to haves
# - Thin about a nice strucuture for the user to understand this full mess
# - update all the plotting functions
# - go through the whole code for cleaning and consistent structure
# - also to make all setter and getter functions as consistent as possible

# TODO: Cross functionality
# - see what I need to change so that GFZ can still use it as before

interpolate_dispatch = {
    InterpolationMethod.ORDINARY_KRIGING: interpolate_group_ordinary_kriging,
    InterpolationMethod.RADIAL_BASIS_FUNCTION: interpolate_group_radial_basis_function,
    InterpolationMethod.UNIVERSAL_COKRIGING: interpolate_group_universal_cokriging,
    InterpolationMethod.GEOINR: interpolate_group_geo_inr,
    InterpolationMethod.LOOP_STRUCTURAL: interpolate_group_loop_structural,
    InterpolationMethod.IDW: interpolate_group_idw,
}

lith = general_updated.combined_interpolator_with_domains(
    frame,
    fault_frame=fault_frame,                     # or None for single-domain
    interpolate_dispatch=interpolate_dispatch,
    assign_domain_ids_to_points=assign_domain_ids_to_points,
    marching_cubes_per_element=marching_cubes_per_element,
    extract_meshes=True,
    verbose=True,
)



#%%
#
# # def combine_group_scalar_fields(groups):
# #     # Assume each group has `scalar_field`, and elements have `scalar_value`
# #     combined = np.full(groups[0].scalar_field.shape, fill_value=-1.0)
# #     for group in groups:
# #         group_field = group.scalar_field
# #         for elem in group.structural_elements:
# #             value = elem.scalar_value
# #             mask = group_field >= value  # Or your logic
# #             combined[mask] = value
# #     return combined


#%%

import matplotlib.pyplot as plt

#%%

def set_scalar_masks_per_domain(
    structural_frame: StructuralFrame,
    scalar_fields_per_domain: dict,
    scalar_values_per_domain: dict,
) -> dict:
    """
    Compute lithology masks per domain and group based on scalar field and scalar values.

    Returns:
        masks_per_domain: dict of {domain_id: {group_name: mask}}
    """
    masks_per_domain = {}
    groups = structural_frame.structural_groups

    for domain_id, scalar_fields_for_groups in scalar_fields_per_domain.items():
        masks_per_domain[domain_id] = {}

        for group in groups:
            scalar_field = scalar_fields_for_groups.get(group.name)
            if scalar_field is None:
                raise ValueError(f"Domain {domain_id}, group '{group.name}': No scalar field found.")

            if group.name not in scalar_values_per_domain[domain_id]:
                raise ValueError(f"Domain {domain_id}: No scalar values found for group '{group.name}'.")

            # Get scalar values of elements in the group (in defined order)
            element_names = [e.name for e in group.structural_elements]
            element_values = [scalar_values_per_domain[domain_id][group.name][name] for name in element_names]

            if not element_values:
                raise ValueError(f"Group '{group.name}' has no scalar values.")

            if group == groups[-1]:
                # Oldest group: full True mask
                mask = np.ones_like(scalar_field, dtype=bool)
            else:
                oldest_scalar = element_values[-1]
                mask = scalar_field >= oldest_scalar

            masks_per_domain[domain_id][group.name] = mask

    return masks_per_domain


#%%

# Assume we have these:
# fault_frame: FaultFrame
# structural_frame: StructuralFrame

grid = fault_frame.grid
domain_map = fault_frame.domain_map
surface_points = frame._surface_points
orientations = frame._orientations
groups = frame.structural_groups

# 1️⃣ Get unique domain IDs
domain_ids = np.unique(domain_map)

# 2️⃣ Loop through domains
final_lith_blocks = []


# # Preparation: Assign domain IDs to surface points and orientations
# sp_in_domain = assign_domain_ids_to_points(grid, domain_map, surface_points)
# ori_in_domain = assign_domain_ids_to_points(grid, domain_map, orientations)

# # TODO: This is where I need to set the storage options
# scalar_fields_per_domain = {}
# scalar_values_per_domain = {}
# masks_per_domain = dict()  # {domain_id: {group_name: lith_mask}}

from copy import deepcopy

# Prep: compute domain ids and tag input data
domain_ids = np.unique(fault_frame.domain_map)
sp_in_domain = assign_domain_ids_to_points(frame.grid, fault_frame.domain_map, frame.surface_points)
ori_in_domain = assign_domain_ids_to_points(frame.grid, fault_frame.domain_map, frame.orientations)

# 1) Interpolate per domain and store results into per-domain slots
for domain_id in domain_ids:
    print(f"🔎 Processing domain {domain_id}")

    # Filter input data for this domain (drop helper column when passing to interpolators)
    sp_filtered = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
    ori_filtered = ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")

    # Basic sanity: at least a couple of points per element present in this domain
    for group in frame.structural_groups:
        for elem in group.structural_elements:
            n_pts = (sp_filtered["formation"] == elem.name).sum()
            if n_pts < 2:
                raise ValueError(f"❌ Not enough surface points for '{elem.name}' in domain {domain_id}")

    # Work on a temporary copy to avoid mutating the base group during interpolation
    domain_groups = deepcopy(frame.structural_groups)

    # Run interpolators per group on the copy
    for g_copy in domain_groups:
        interpolate_group_universal_cokriging(
            group=g_copy,
            grid=frame.grid,
            group_surface_points_df=sp_filtered,
            group_orientations_points_df=ori_filtered,
        )

    # Copy results back into the original groups, but **per domain**
    for g_copy in domain_groups:
        # find the matching original group
        g_orig = next(g for g in frame.structural_groups if g.name == g_copy.name)

        # store scalar field for this domain
        g_orig.set_scalar_field_for_domain(domain_id, g_copy.scalar_field)

        # store scalar values for this domain, per element
        for e_copy in g_copy.structural_elements:
            e_orig = next(e for e in g_orig.structural_elements if e.name == e_copy.name)
            e_orig.set_scalar_value_for_domain(domain_id, float(e_copy.scalar_value))

# 2) Build and store masks per domain using your domain-aware helper
masks_per_domain = set_scalar_masks_per_domain(
    structural_frame=frame,
    # pass per-domain fields/values in the shape this helper expects
    scalar_fields_per_domain={
        d: {g.name: g.get_scalar_field_for_domain(d) for g in frame.structural_groups}
        for d in domain_ids
    },
    scalar_values_per_domain={
        d: {
            g.name: {e.name: e.get_scalar_value_for_domain(d) for e in g.structural_elements}
            for g in frame.structural_groups
        }
        for d in domain_ids
    },
)

# Persist masks into the new per-domain storage on each group
for domain_id, group_map in masks_per_domain.items():
    for group_name, mask in group_map.items():
        g = frame[group_name]
        g.set_mask_for_domain(domain_id, mask)


#%%

# Access specific scalar field for a domain and group
scalar_fields_per_domain[0]["Top"]

#%%

# Access specific scalar field for a domain and group
scalar_values_per_domain[0]["Top"]["UnitD"]

#%%
masks_per_domain

#%%

# plot a section of a specific scalar field
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

plot_scalar_field_section(scalar_fields_per_domain[1]["Bot"], frame.grid, index=25)

#%%

plot_scalar_field_section(masks_per_domain[2]["Top"].T, frame.grid, index=25)

#%%

def compute_lithology_block_from_domains(
    structural_frame: 'StructuralFrame',
    scalar_fields_per_domain: dict,
    scalar_values_per_domain: dict,
    masks_per_domain: dict,
    domain_map: np.ndarray,
) -> np.ndarray:
    """
    Combines scalar fields and masks from multiple fault domains into a single lithology block.

    Args:
        structural_frame: StructuralFrame containing groups and elements.
        scalar_fields_per_domain: dict[domain_id][group_name] = scalar_field (3D array)
        scalar_values_per_domain: dict[domain_id][group_name][element_name] = scalar_value
        masks_per_domain: dict[domain_id][group_name] = mask (3D bool array)
        domain_map: 3D array with domain IDs.

    Returns:
        lith_block: 3D NumPy array with lithology IDs (0 = undefined)
    """
    shape = domain_map.shape
    lith_block = np.zeros(shape, dtype=int)

    # 1️⃣ Assign unique IDs globally (same element -> same ID across domains)
    current_id = 1
    element_id_map = {}

    for group in reversed(structural_frame.structural_groups):  # oldest to youngest
        for element in reversed(group.structural_elements):
            if element.name not in element_id_map:
                element.set_id(current_id)
                element_id_map[element.name] = current_id
                current_id += 1

    # 2️⃣ Loop over domains and compute lithology per domain
    domain_ids = np.unique(domain_map)

    for domain_id in domain_ids:
        domain_lith_block = np.zeros(shape, dtype=int)

        for group in reversed(structural_frame.structural_groups):  # oldest to youngest
            group_name = group.name

            scalar_field = scalar_fields_per_domain[domain_id][group_name]
            mask = masks_per_domain[domain_id][group_name]
            group_block = np.zeros(shape, dtype=int)

            for element in group.structural_elements:
                scalar_value = scalar_values_per_domain[domain_id][group_name][element.name]
                element_id = element_id_map[element.name]

                # Build element mask
                element_mask = (scalar_field >= scalar_value) & (group_block == 0)
                group_block[element_mask] = element_id

            # Apply group-level mask
            group_block = np.where(mask, group_block, 0)

            # Combine into domain_lith_block (youngest group takes precedence)
            domain_lith_block = np.where(group_block > 0, group_block, domain_lith_block)

        # 3️⃣ Mask domain-specific values into the global lith_block
        domain_mask = domain_map == domain_id
        lith_block[domain_mask] = domain_lith_block[domain_mask]

    return lith_block

#%%

# Compute the lithology block from all domains
lith_block = compute_lithology_block_from_domains(
    structural_frame=frame,
    scalar_fields_per_domain=scalar_fields_per_domain,
    scalar_values_per_domain=scalar_values_per_domain,
    masks_per_domain=masks_per_domain,
    domain_map=domain_map
)

#%%

# Plot the lithology block slice, not as fucntion
lith_block_reshaped = lith_block.reshape(grid.resolution)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(lith_block_reshaped[:, 1, :], extent=grid.extent[:4], origin='lower', cmap='viridis')
ax.scatter(sp_filtered['X'], sp_filtered['Z'], c=sp_filtered['formation'].astype('category').cat.codes,
           cmap='viridis', s=10, alpha=1)
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.title("Lithology Block Slice")
plt.show()

#%%

def extract_masked_meshes_per_domain(
    structural_frame: StructuralFrame,
    grid_spacing: np.ndarray,
    extent: np.ndarray,
    scalar_fields_per_domain: dict,
    scalar_values_per_domain: dict,
    masks_per_domain: dict,
    domain_map: np.ndarray,
) -> dict:
    """
    Extract masked surface meshes per element per fault domain.

    Args:
        structural_frame: The StructuralFrame with groups/elements.
        grid_spacing: (dx, dy, dz) tuple for marching cubes.
        extent: (xmin, xmax, ymin, ymax, zmin, zmax)
        scalar_fields_per_domain: dict[domain][group] = 3D scalar field
        scalar_values_per_domain: dict[domain][group][element] = scalar value
        masks_per_domain: dict[domain][group] = 3D bool mask
        domain_map: 3D domain block array
        marching_cubes_per_element: Callable(scalar_field.T, isovalue, grid_spacing, extent, mask.T)

    Returns:
        dict[domain][element_name] = (vertices, edges)
    """
    meshes_per_domain = {}

    domain_ids = np.unique(domain_map)
    groups = structural_frame.structural_groups

    for domain_id in domain_ids:
        domain_mask = domain_map == domain_id
        domain_meshes = {}

        for i, group in enumerate(groups):
            group_name = group.name
            scalar_field = scalar_fields_per_domain[domain_id][group_name]
            group_mask = masks_per_domain[domain_id][group_name]

            # Domain + lithology masking
            combined_mask = group_mask & domain_mask

            for element in group.structural_elements:
                element_name = element.name
                scalar_value = scalar_values_per_domain[domain_id][group_name][element_name]

                # Extract masked surface mesh
                vertices, edges = marching_cubes_per_element(
                    scalar_field.T,
                    scalar_value,
                    grid_spacing,
                    extent,
                    mask=combined_mask.T
                )

                domain_meshes[element_name] = (vertices, edges)

        meshes_per_domain[domain_id] = domain_meshes

    return meshes_per_domain

#%%

meshes_per_domain = extract_masked_meshes_per_domain(frame,
                                             grid_spacing=frame.grid.spacing,
                                             extent=frame.grid.extent,
                                             scalar_fields_per_domain=scalar_fields_per_domain,
                                             scalar_values_per_domain=scalar_values_per_domain,
                                             masks_per_domain=masks_per_domain,
                                             domain_map=domain_map)

#%%

meshes_per_domain[0]["UnitD"]

#%%

import pyvista as pv
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_3d_geology_model(
    meshes_per_domain: dict,
    fault_frame,
    structural_frame,
    point_size=6,
    show_faults=True,
):
    import pyvista as pv
    import numpy as np

    p = pv.Plotter()

    # Create a lookup for structural elements by name
    element_color_map = {
        element.name: getattr(element, "color", "blue")
        for group in structural_frame.structural_groups
        for element in group.structural_elements
    }

    # 2️⃣ Plot structural element meshes
    for domain_id, domain_meshes in meshes_per_domain.items():
        for element_name, (vertices, faces) in domain_meshes.items():
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Get color
            color = element_color_map.get(element_name, "blue")

            # Prepare mesh
            faces_flat = np.hstack([[3, *tri] for tri in faces])
            surf = pv.PolyData(vertices, faces_flat)

            # Add to plot
            p.add_mesh(surf, color=color, opacity=1.0, label=f"{element_name} (D{domain_id})")

    # 3️⃣ Plot fault meshes
    if show_faults:
        for fault in fault_frame.fault_elements:
            if len(fault.vertices) == 0 or len(fault.edges) == 0:
                continue

            # Default to black if color not set
            fault_color = getattr(fault, "color", "black")

            fault_faces_flat = np.hstack([[3, *tri] for tri in fault.edges])
            mesh = pv.PolyData(fault.vertices, fault_faces_flat)
            p.add_mesh(mesh, color=fault_color, opacity=1.0, label=f"Fault: {fault.name}")

    # 4️⃣ Add legend and show
    p.add_legend()
    p.show()


#%%

fault_frame.fault_elements[0].edges


#%%

plot_3d_geology_model(meshes_per_domain, fault_frame, frame, show_faults=True)


#%%

# TODO: The big questions
# Overarching storage and setup structure (combine structural frame and fault frame?)
# Think what happens if there is no fault
# How to store the scalar fields, values, masks per domain?
# How to store the meshes per domain?
# How to store the lithology block per domain?

# Warnings ad stops
# if faults are cross-cutting each other
# Check for data in each fault domain/group (what happens if no data for group in domain)

# Logical additions
# age relations between faults and groups (fault eroded etc), how to model that?
# mainly this means rewriting the masking process at the end I think
# might also require changes to the actual masks or new masks

# Efficiency
# Implement Computation only within domain to reduce overhead



#%%

# TODO: Are these masks actually correct in how I want them?
# TODO> remove rault relations at current state, are not used/should not be used?
fault_frame.fault_elements[0].mask

# plot slice of that mask
plt.imshow(fault_frame.fault_elements[2].mask[:, 1, :], extent=grid.extent[:4], origin='lower', cmap='gray')
plt.show()
