import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import itertools
import colorsys

from core.object_components import InputData, GeomodelResults
from core.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes_new

from typing import Callable, Dict, Tuple, Optional

from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import \
    interpolate_group_ordinary_kriging
from core.interpolator_components.interpolators_per_group.radial_basis_function_per_group import \
    interpolate_group_radial_basis_function
from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import \
    interpolate_group_universal_cokriging
from core.interpolator_components.interpolators_per_group.loop_structural_per_group import \
    interpolate_group_loop_structural
from core.interpolator_components.interpolators_per_group.geoinr_per_group import interpolate_group_geo_inr

from core.structural_objects.objects import StructuralFrame, StructuralGroup, StructuralElement, InterpolationMethod


#%%

def set_scalar_masks(structural_frame: 'StructuralFrame'):
    """
    Set masks for all groups in the structural frame based on scalar field values.

    - The last group (oldest) gets a full True mask.
    - Other groups are masked where their scalar field is less than or equal to
      the scalar value of their oldest (last) structural element.
    """
    if not structural_frame.structural_groups:
        raise ValueError("The structural frame contains no groups.")

    groups = structural_frame.structural_groups

    for i, group in enumerate(groups):
        scalar_field = group.scalar_field
        if scalar_field is None:
            raise ValueError(f"Group '{group.name}' has no scalar field set.")

        if i == len(groups) - 1:
            # Oldest group: full True mask
            mask = np.ones_like(scalar_field, dtype=bool)
        else:
            # Get the oldest element in the group (last in list)
            if not group.structural_elements:
                raise ValueError(f"Group '{group.name}' has no structural elements.")

            oldest_element = group.structural_elements[-1]
            if oldest_element.scalar_value is None:
                raise ValueError(
                    f"Oldest element '{oldest_element.name}' in group '{group.name}' has no scalar value set.")

            mask = scalar_field >= oldest_element.scalar_value

        group.set_mask(mask)


def compute_lithology_block(structural_frame: 'StructuralFrame') -> np.ndarray:
    """
    Computes the lithology block by combining all structural groups,
    honoring scalar fields and masks.

    Groups are assumed to be ordered from youngest to oldest.
    Returns:
        A NumPy array with lithology IDs (0 = undefined).
    """
    shape = structural_frame.structural_groups[0].scalar_field.shape
    lith_block = np.zeros(shape, dtype=int)

    # Assign IDs from oldest to youngest element for consistent mapping
    current_id = 1
    element_id_map = {}

    # Reverse groups to go from oldest to youngest
    for group in reversed(structural_frame.structural_groups):
        for element in reversed(group.structural_elements):
            element.set_id(current_id)
            element_id_map[element.name] = current_id
            current_id += 1

    # Process from oldest to youngest (reverse group order)
    for group in reversed(structural_frame.structural_groups):
        group_block = np.zeros(shape, dtype=int)
        scalar_field = group.scalar_field
        mask = group.mask

        if scalar_field is None or mask is None:
            raise ValueError(f"Group '{group.name}' missing scalar_field or mask.")

        # Process elements from youngest to oldest (reverse to give youngest priority)
        # for element in reversed(group.structural_elements):
        for element in group.structural_elements:

            if element.scalar_value is None:
                raise ValueError(f"Element '{element.name}' missing scalar_value.")

            element_id = element.id  # pre-assigned, increasing with age
            element_mask = (scalar_field >= element.scalar_value) & (group_block == 0)
            group_block[element_mask] = element_id

        # Apply group-level mask
        group_block = np.where(mask, group_block, 0)

        # Overwrite into lith_block (youngest group takes precedence)
        lith_block = np.where(group_block > 0, group_block, lith_block)

    return lith_block


def extract_all_meshes(
        structural_frame: StructuralFrame,
        grid_spacing: np.ndarray,
        extent: np.ndarray,
        combined_lithology_block: np.ndarray,
        marching_cubes_per_element: Callable,
        marching_cubes: Callable
):
    """
    Extracts 'masked', 'unmasked', and 'combined' surface meshes for all structural elements in a frame.

    Args:
        structural_frame: The StructuralFrame containing groups and elements.
        grid_spacing: Tuple of (dx, dy, dz) spacing for the marching cubes.
        extent: Spatial extent (xmin, xmax, ymin, ymax, zmin, zmax).
        combined_lithology_block: Final lithology volume for 'combined' mesh.
        marching_cubes_per_element: Callable for computing meshes per scalar value (with optional mask).
        marching_cubes: Callable for computing combined mesh from lithology block.

    Notes:
        Updates each StructuralElement with its respective surface meshes under
        keys: 'masked', 'unmasked', and 'combined'.
    """
    for i, group in enumerate(structural_frame.structural_groups):
        scalar_field = group.scalar_field

        if i == 0:
            # First group (youngest) has no mask, so we create a full True mask
            mask = np.ones_like(scalar_field, dtype=bool)
        else:
            # Take mask from the previous group
            mask = ~structural_frame.structural_groups[i - 1].mask

        if scalar_field is None:
            raise ValueError(f"Group '{group.name}' is missing a scalar field.")
        if mask is None:
            raise ValueError(f"Group '{group.name}' is missing a mask.")

        for element in group.structural_elements:
            if element.scalar_value is None:
                raise ValueError(f"Element '{element.name}' is missing a scalar value.")

            # Masked version (within unconformities)
            vertices, edges = marching_cubes_per_element(
                scalar_field.T,
                element.scalar_value,
                grid_spacing,
                extent,
                mask=mask.T
            )
            element.set_mesh("masked", vertices, edges)

            # Unmasked version (through unconformities)
            vertices, edges = marching_cubes_per_element(
                scalar_field.T,
                element.scalar_value,
                grid_spacing,
                extent,
                mask=None
            )
            element.set_mesh("unmasked", vertices, edges)

    # Combined mesh extraction
    unique_ids = [
        element.id - 0.1  # Adjust IDs to get contour levels slightly below the lowest ID
        for group in structural_frame.structural_groups
        for element in group.structural_elements
        if element.id is not None
    ]

    combined_vertices, combined_edges = marching_cubes(
        combined_lithology_block.T,
        unique_ids,
        grid_spacing,
        extent
    )

    # Assign combined mesh to each element
    idx = 0
    for group in structural_frame.structural_groups:
        for element in group.structural_elements:
            element.set_mesh("combined", combined_vertices[idx], combined_edges[idx])
            idx += 1


def combined_interpolator(frame):
    """
    Compute a model based on input data using a combination of Ordinary Kriging and RBF interpolation.
    """

    # 4. Perform interpolation for each structural group based on its method
    # TODO: This needs to return a scalar field per group and scalar values per element in this group
    for group in frame.structural_groups:
        # get the corresponding surface points
        group_surface_points = frame.get_surface_points_for_group(group.name)
        if group.interpolation_method == InterpolationMethod.ORDINARY_KRIGING:
            # Perform Ordinary Kriging interpolation
            print("Hello, I am Ordinary Kriging")
            interpolate_group_ordinary_kriging(
                group=group,
                group_surface_points_df=group_surface_points,
                grid=frame.grid
            )
        elif group.interpolation_method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            # Perform Radial Basis Function interpolation
            print("Hello, I am Radial Basis Function")
            interpolate_group_radial_basis_function(
                group=group,
                group_surface_points_df=group_surface_points,
                grid=frame.grid
            )
        elif group.interpolation_method == InterpolationMethod.UNIVERSAL_COKRIGING:
            # Perform Universal CoKriging interpolation
            print("Hello, I am Universal CoKriging")
            interpolate_group_universal_cokriging(
                group=group,
                grid=frame.grid,
                group_surface_points_df=group_surface_points,
                group_orientations_points_df=frame.get_orientations_for_group(group.name),
            )
        elif group.interpolation_method == InterpolationMethod.GEOINR:
            # Perform GeoINR interpolation
            print("Hello, I am GeoINR")
            interpolate_group_geo_inr(
                group=group,
                grid=frame.gridd,
                group_surface_points_df=group_surface_points,
                group_orientations_points_df=frame.get_orientations_for_group(group.name)
            )
            pass
        elif group.interpolation_method == InterpolationMethod.LOOP_STRUCTURAL:
            # Perform Loop Structural interpolation
            print("Hello, I am Loop Structural")
            interpolate_group_loop_structural(
                group=group,
                grid=frame.grid,
                group_surface_points_df=group_surface_points,
                group_orientations_points_df=frame.get_orientations_for_group(group.name),
            )
            pass
        else:
            raise ValueError(f"Unsupported interpolation method: {group.interpolation_method}")

    print("Interpolation done")

    # 5. Create masks based on order of structural groups, scalar fields and scalar values
    set_scalar_masks(frame)

    print("Masking done")

    # 6. Create a combined result (lith_block) based on masks, scalar fields and scalar values
    lith_block = compute_lithology_block(frame)

    print("Combining lithology block done")

    # 7. Extract surface meshes based on the combined result, scalar fields and scalar values
    extract_all_meshes(
        structural_frame=frame,
        grid_spacing=frame.grid.spacing,
        extent=frame.grid.extent,
        combined_lithology_block=lith_block,
        marching_cubes_per_element=marching_cubes_per_element,
        marching_cubes=marching_cubes_new
    )

    print("Mesh extraction done")

    return frame, lith_block


#%%

def generate_grouped_colors_per_element(groups, base_colormap="Accent"):
    """
    Generate distinct color shades for each element in each group.

    Args:
        groups: List of StructuralGroup instances.
        base_colormap: Matplotlib colormap for base group colors.

    Returns:
        dict: {element_name: hex_color}
    """
    group_cmap = plt.get_cmap(base_colormap)
    color_map = {}

    n_groups = len(groups)
    for group_idx, group in enumerate(groups):
        base_rgb = group_cmap(group_idx / n_groups)[:3]  # Get RGB triple (ignore alpha)
        base_hls = colorsys.rgb_to_hls(*base_rgb)

        n_elements = len(group.structural_elements)
        for i, element in enumerate(group.structural_elements):
            # Create variation in lightness (from dark to light)
            lightness = 0.35 + 0.5 * (i / max(n_elements - 1, 1))  # Range ~[0.35, 0.85]
            varied_rgb = colorsys.hls_to_rgb(base_hls[0], lightness, base_hls[2])
            hex_color = to_hex(varied_rgb)
            color_map[element.name] = hex_color

    return color_map


def build_structural_frame(
        mapping_object: Dict[str, Tuple[str, ...]],
        extent: np.ndarray,
        resolution: np.ndarray,
        surface_points: pd.DataFrame,
        orientations: Optional[pd.DataFrame] = None,
        default_interpolation: InterpolationMethod = InterpolationMethod.ORDINARY_KRIGING
) -> StructuralFrame:
    # Normalize column names for consistency
    surface_points = surface_points.rename(columns=str.strip)
    if orientations is not None:
        orientations = orientations.rename(columns=str.strip)

    # Validate required columns
    required_surface_cols = {"X", "Y", "Z", "formation"}
    if not required_surface_cols.issubset(surface_points.columns):
        missing = required_surface_cols - set(surface_points.columns)
        raise ValueError(f"Surface points missing required columns: {missing}")

    if orientations is not None:
        required_orientation_cols = {"X", "Y", "Z", "G_x", "G_y", "G_z", "formation"}
        if not required_orientation_cols.issubset(orientations.columns):
            missing = required_orientation_cols - set(orientations.columns)
            raise ValueError(f"Orientations missing required columns: {missing}")

    all_element_names = list(itertools.chain.from_iterable(mapping_object.values()))
    unique_elements = list(dict.fromkeys(all_element_names))  # preserve order

    # Validate that each element has at least 1 surface point
    for elem in unique_elements:
        count = surface_points[surface_points["formation"] == elem].shape[0]
        if count == 0:
            raise ValueError(f"No surface points found for structural element '{elem}'")

    # If orientations are provided, ensure each group has at least one relevant entry
    if orientations is not None:
        for group_name, element_names in mapping_object.items():
            group_orient = orientations[orientations["formation"].isin(element_names)]
            if group_orient.empty:
                raise ValueError(f"No orientations found for structural group '{group_name}'")

    # Build groups first
    group_objects = []
    for group_name, element_names in mapping_object.items():
        elements = [StructuralElement(name=name) for name in element_names]
        group = StructuralGroup(
            name=group_name,
            interpolation_method=None,  # placeholder
            scalar_field=np.array([]),
            structural_elements=elements
        )
        group.set_interpolation_method(default_interpolation)
        group_objects.append(group)

    # Generate colors AFTER groups exist
    color_map = generate_grouped_colors_per_element(group_objects)

    # Now assign colors to each element
    element_objects = {}
    for group in group_objects:
        for elem in group.structural_elements:
            elem.set_color(color_map[elem.name])
            element_objects[elem.name] = elem

    # Create regular grid
    grid = RegularGrid(extent=extent, resolution=resolution)

    frame = StructuralFrame(structural_groups=group_objects)
    frame._grid = grid
    frame._surface_points = surface_points
    frame._orientations = orientations

    return frame
