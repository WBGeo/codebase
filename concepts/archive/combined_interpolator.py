import numpy as np
import pandas as pd

from core.object_components import InputData_StructuralElements
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.utility.surface_mesh_extraction import marching_cubes_per_element, marching_cubes_new
import os

from typing import Dict, Tuple, Optional, List,Callable
from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import to_hex
import itertools

from pykrige.ok3d import OrdinaryKriging3D

cwd = os.getcwd()

#%%


class InterpolationMethod(str, Enum):
    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"


class StructuralElement(BaseModel):
    """
    A structural element within a structural group.

    Attributes:
        name: Name of the element.
        scalar_value: Scalar field value (float), set during computation.
        id: Unique identifier for the element (int), set later.
        color: Display color (hex string), set during frame generation.
        vertices: Dictionary of surface mesh vertices arrays keyed by mesh type ('masked', 'unmasked', 'combined').
        edges: Dictionary of surface mesh edges arrays keyed by mesh type ('masked', 'unmasked', 'combined').
    """
    name: str
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _id: Optional[int] = PrivateAttr(default=None)
    _color: Optional[str] = PrivateAttr(default=None)
    _vertices: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _edges: Dict[str, np.ndarray] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    # Read-only properties
    @property
    def scalar_value(self) -> Optional[float]:
        return self._scalar_value

    @property
    def id(self) -> Optional[int]:
        return self._id

    @property
    def color(self) -> Optional[str]:
        return self._color

    @property
    def vertices(self) -> Optional[np.ndarray]:
        return self._vertices

    @property
    def edges(self) -> Optional[np.ndarray]:
        return self._edges

    # Controlled setters
    def set_scalar_value(self, value: float):
        self._scalar_value = value

    def set_id(self, element_id: int):
        self._id = element_id

    def set_color(self, hex_color: str):
        self._color = hex_color

    def set_mesh(self, mesh_type: str, vertices: np.ndarray, edges: np.ndarray):
        """
        Set the vertices and edges for a specific mesh type (e.g., 'masked', 'unmasked', 'combined').

        Raises:
            ValueError if mesh_type is not one of the allowed types or already exists.
        """
        if mesh_type not in {"masked", "unmasked", "combined"}:
            raise ValueError(f"Invalid mesh type '{mesh_type}'. Allowed types are: masked, unmasked, combined.")
        if mesh_type in self._vertices or mesh_type in self._edges:
            raise ValueError(f"Mesh type '{mesh_type}' already set for element '{self.name}'.")

        self._vertices[mesh_type] = vertices
        self._edges[mesh_type] = edges

    def get_mesh(self, mesh_type: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the vertices and edges for the given mesh type.

        Raises:
            KeyError if the mesh type does not exist.
        """
        try:
            return self._vertices[mesh_type], self._edges[mesh_type]
        except KeyError:
            raise KeyError(f"Mesh '{mesh_type}' not found in element '{self.name}'.")

class StructuralGroup(BaseModel):
    """
    A structural group that contains multiple structural elements and associated input_data.

    Attributes:
        name: Name of the structural group.
        structural_elements: Ordered list of StructuralElement objects.
        interpolation_method: Interpolation method used.
        scalar_field: Computed scalar field (1D or multi-D array), set after interpolation.
        mask: Optional mask for the scalar field (1D or multi-D array), set after interpolation.
    """
    name: str
    structural_elements: List['StructuralElement'] = Field(default_factory=list)
    interpolation_method: Optional['InterpolationMethod'] = None
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, element_name: str) -> 'StructuralElement':
        for elem in self.structural_elements:
            if elem.name == element_name:
                return elem
        raise KeyError(f"Structural element '{element_name}' not found in group '{self.name}'.")

    @property
    def scalar_field(self) -> Optional[np.ndarray]:
        return self._scalar_field

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_scalar_field(self, field: np.ndarray):
        self._scalar_field = field

    def set_mask(self, mask_array: np.ndarray):
        self._mask = mask_array



class StructuralFrame(BaseModel):
    structural_groups: List[StructuralGroup] = Field(default_factory=list)
    surface_points: pd.DataFrame
    orientations: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, group_name: str) -> StructuralGroup:
        for group in self.structural_groups:
            if group.name == group_name:
                return group
        raise KeyError(f"Structural group '{group_name}' not found.")

    def get_surface_points_for_element(self, element_name: str) -> pd.DataFrame:
        return self.surface_points[self.surface_points["formation"] == element_name]

    def get_orientations_for_element(self, element_name: str) -> Optional[pd.DataFrame]:
        if self.orientations is None:
            return None
        return self.orientations[self.orientations["formation"] == element_name]

    def get_surface_points_for_group(self, group_name: str) -> pd.DataFrame:
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self.surface_points[self.surface_points["formation"].isin(element_names)]

    def get_orientations_for_group(self, group_name: str) -> Optional[pd.DataFrame]:
        if self.orientations is None:
            return None
        group = self[group_name]
        element_names = [e.name for e in group.structural_elements]
        return self.orientations[self.orientations["formation"].isin(element_names)]

    def pretty_print(self):
        print("📦 Structural Frame Overview")
        print("────────────────────────────")
        print(f"• Number of structural groups: {len(self.structural_groups)}")
        print(f"• Total surface points: {len(self.surface_points)} entries")
        if self.orientations is not None:
            print(f"• Total orientations: {len(self.orientations)} entries")
        else:
            print("• Orientation input_data: None")

        print("\n🧱 Structural Groups:\n")

        for group in self.structural_groups:
            print(f"  ▶ Group: {group.name}")
            print(f"    ├─ Interpolation method: {group.interpolation_method}")

            # Count surface points for group
            group_element_names = [e.name for e in group.structural_elements]
            group_surface_points = self.surface_points[self.surface_points["formation"].isin(group_element_names)]
            print(f"    ├─ Surface points in group: {len(group_surface_points)}")

            # Count orientations for group
            if self.orientations is not None:
                group_orientations = self.orientations[self.orientations["formation"].isin(group_element_names)]
                print(f"    ├─ Orientations in group: {len(group_orientations)}")

            print(f"    └─ Structural Elements:")
            for elem in group.structural_elements:
                print(f"        • {elem.name}")
                if elem.color:
                    print(f"           - Color: {elem.color}")

                elem_surface_points = self.surface_points[self.surface_points["formation"] == elem.name]
                print(f"           - Surface points: {len(elem_surface_points)}")

                if self.orientations is not None:
                    elem_orientations = self.orientations[self.orientations["formation"] == elem.name]
                    print(f"           - Orientations: {len(elem_orientations)}")
            print()  # Blank line between groups


#%%

def generate_distinct_colors(n):
    """Generates `n` visually distinct hex colors using a colormap."""
    cmap = cm.get_cmap("tab20", n)
    return [to_hex(cmap(i)) for i in range(n)]


def build_structural_frame(
    mapping_object: Dict[str, Tuple[str, ...]],
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

    # Generate colors for each element
    def generate_colors(n):
        cmap = colormaps.get_cmap("tab20").resampled(n)
        return [to_hex(cmap(i)) for i in range(n)]

    color_map = dict(zip(unique_elements, generate_colors(len(unique_elements))))

    # Build elements
    element_objects = {
        name: StructuralElement(
            name=name,
            color=color_map[name]
        )
        for name in unique_elements
    }

    # Build groups
    group_objects = []
    for group_name, element_names in mapping_object.items():
        elements = [element_objects[name] for name in element_names]
        group = StructuralGroup(
            name=group_name,
            interpolation_method=default_interpolation,
            scalar_field=np.array([]),  # placeholder
            structural_elements=elements
        )
        group_objects.append(group)

    return StructuralFrame(
        structural_groups=group_objects,
        surface_points=surface_points,
        orientations=orientations
    )

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
                raise ValueError(f"Oldest element '{oldest_element.name}' in group '{group.name}' has no scalar value set.")

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

    current_id = 1
    element_id_map = {}

    # Process from oldest to youngest (reverse group order)
    for group in reversed(structural_frame.structural_groups):
        group_block = np.zeros(shape, dtype=int)
        scalar_field = group.scalar_field
        mask = group.mask

        if scalar_field is None or mask is None:
            raise ValueError(f"Group '{group.name}' missing scalar_field or mask.")

        # Process elements from youngest to oldest (reverse to give youngest priority)
        for element in reversed(group.structural_elements):
            if element.scalar_value is None:
                raise ValueError(f"Element '{element.name}' missing scalar_value.")

            element.set_id(current_id)
            element_id_map[element.name] = current_id

            element_mask = (scalar_field >= element.scalar_value) & (group_block == 0)
            group_block[element_mask] = current_id

            current_id += 1

        # Apply group-level mask
        group_block = np.where(mask, group_block, 0)

        # Overwrite into lith_block (youngest group takes precedence)
        lith_block = np.where(group_block > 0, group_block, lith_block)

    return lith_block





#%%


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
    print("hello")
    for group in structural_frame.structural_groups:
        print(group)
        scalar_field = group.scalar_field
        mask = group.mask

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


    # TODO: Combined version from lithology block (structured mesh across IDs)
    # unique_ids = [
    #     element.id
    #     for group in structural_frame.structural_groups
    #     for element in group.structural_elements
    #     if element.id is not None
    # ]
    #
    # combined_vertices, combined_edges = marching_cubes(
    #     combined_lithology_block,
    #     unique_ids,
    #     grid_spacing,
    #     extent
    # )

    # Assign to the corresponding element (assumes element ID maps directly to index in unique_ids)
    # id_to_element = {
    #     element.id: element
    #     for group in structural_frame.groups
    #     for element in group.structural_elements
    #     if element.id is not None
    # }

    # TODO: Combined
    # for idx, lith_id in enumerate(unique_ids):
    #     if lith_id in id_to_element:
    #         id_to_element[lith_id].set_mesh("combined", combined_vertices[idx], combined_edges[idx])


#%%

def interpolate_group_ordinary_kriging(
        group: StructuralGroup,
        group_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        grid,
        variogram_model: str,
        variogram_parameters: list,
        anisotropy_scaling_z: float,
        neighbors: Optional[int] = None,
) -> None:
    """
    Perform Ordinary Kriging interpolation for a single structural group.

    Args:
        group: StructuralGroup instance to interpolate.
        group_surface_points_df: DataFrame with columns ['X', 'Y', 'Z', 'formation'] filtered for this group.
        grid: Grid object containing gridx, gridy, gridz arrays for interpolation.
        variogram_model: Variogram model name, e.g. 'spherical'.
        variogram_parameters: List of variogram parameters [sill, range, nugget].
        anisotropy_scaling_z: Scaling factor for z-direction anisotropy.
        neighbors: Number of closest points to use in kriging.
    """
    # 1. Assign strictly increasing scalar values: oldest = 1, youngest = n
    for i, elem in enumerate(reversed(group.structural_elements), start=1):
        elem.set_scalar_value(float(i))

    if group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group {group.name}")

    # 2. Map formation (element name) to scalar_value
    formation_to_scalar = {elem.name: elem.scalar_value for elem in group.structural_elements}
    scalar_values = group_surface_points_df['formation'].map(formation_to_scalar).values.astype(float)

    # 3. Extract coordinates
    x = group_surface_points_df['X'].values
    y = group_surface_points_df['Y'].values
    z = group_surface_points_df['Z'].values

    # 4. Perform Ordinary Kriging
    ok3d = OrdinaryKriging3D(
        x, y, z,
        scalar_values,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )

    k3d1, ss3d = ok3d.execute(
        "grid",
        grid.gridx,
        grid.gridy,
        grid.gridz,
        n_closest_points=neighbors
    )

    # 5. Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    group.set_scalar_field(k3d1)

    # Save results, need to explicitly limit to maximum value as defined by replacement mapping
    # max_value = max(replacements[element] for element in value)
    # k3d1[k3d1 > max_value] = max_value
    # results.append(k3d1.astype(int).T.copy())





#%%

mapping_object = {
    "Strat_Series1": ("rock4", "rock3"),
    "Strat_Series2": ("rock2", "rock1"),
}

surface_points = pd.read_csv("examples/input_data/model12_surface_points_df.csv")

# orientations = pd.read_csv("examples/input_data/model12_orientations_df.csv")

frame = build_structural_frame(mapping_object, surface_points) #, orientations)
frame.pretty_print()

#%%

frame.get_orientations_for_group("Strat_Series2")

#%%

frame["Strat_Series1"].interpolation_method = InterpolationMethod.RADIAL_BASIS_FUNCTION

#%%

frame.pretty_print()

#%%


def combined_interpolator(input_data):
    """
    Compute a model based on input input_data using a combination of Ordinary Kriging and RBF interpolation.

    Args:
        input_data (InputData_StructuralElements): The input input_data for the geological model.

    Returns:
        results (GeomodelResults): The results of the geological model.
    """
    # This is a function to allow the computation of a model using a different interpolation method per structural group.

    # TODO: Steps
    # 1. Define grid
    # 2. Mapping object needs to contain interpolation method per structural group and corresponding parameters
    # 3. Data needs to be separated based on structural groups, and transformed to correct input for method
    # 4. Perform interpolation per structural group, returning one scalar field and the corresponding scalar values
    # 5. Create masks based on order of structural groups, scalar fields and scalar values
    # 6. Create a combined result (lith_block) based on masks, scalar fields and scalar values
    # 7. Extract surface meshes based on the combined result, scalar fields and scalar values

    # 1. Create a Grid instance
    grid = RegularGrid(input_data.extent, input_data.resolution)

    print("grid done")

    # 2. Build the structural frame from input input_data
    frame = build_structural_frame(input_data.mapping_object, input_data.surface_points, input_data.orientations)

    print("frame done")

    # 3.

    # 4. Perform interpolation for each structural group based on its method
    # TODO: This needs to return a scalar field per group and scalar values per element in this group
    for group in frame.structural_groups:
        # get the corresponding surface points
        group_surface_points = frame.get_surface_points_for_group(group.name)
        if group.interpolation_method == InterpolationMethod.ORDINARY_KRIGING:
            # Perform Ordinary Kriging interpolation
            interpolate_group_ordinary_kriging(
                group=group,
                group_surface_points_df=group_surface_points,
                grid=grid,
                variogram_model='gaussian',
                variogram_parameters=[1, 500, 0],  # Example parameters, adjust as needed
                anisotropy_scaling_z=0.3,
                neighbors=None
            )
        elif group.interpolation_method == InterpolationMethod.RADIAL_BASIS_FUNCTION:
            # Perform Radial Basis Function interpolation
            pass
        elif group.interpolation_method == InterpolationMethod.UNIVERSAL_COKRIGING:
            # Perform Universal Co-Kriging interpolation
            pass
        elif group.interpolation_method == InterpolationMethod.GEOINR:
            # Perform GeoINR interpolation
            pass
        elif group.interpolation_method == InterpolationMethod.LOOP_STRUCTURAL:
            # Perform Loop Structural interpolation
            pass
        else:
            raise ValueError(f"Unsupported interpolation method: {group.interpolation_method}")

    print("interpolation done, scalar fields confirmed correct")

    # 5. Create masks based on order of structural groups, scalar fields and scalar values
    set_scalar_masks(frame)

    print("masks done, confirmed correct")

    # 6. Create a combined result (lith_block) based on masks, scalar fields and scalar values
    lith_block = compute_lithology_block(frame)

    print("lithology block done")

    # 7. Extract surface meshes based on the combined result, scalar fields and scalar values
    extract_all_meshes(
        structural_frame=frame,
        grid_spacing=grid.spacing,
        extent=input_data.extent,
        combined_lithology_block=lith_block,
        marching_cubes_per_element=marching_cubes_per_element,
        marching_cubes=marching_cubes_new
    )

    print("surface meshes done")

    return frame, lith_block


#%%


data_test = InputData_StructuralElements(name='Model_12',
                                         extent=np.array([0, 2000, 0, 1000, 0, 1000]),
                                         resolution=np.array([100, 50, 50]),
                                         surface_points=pd.read_csv(
                          cwd + "/examples/input_data/model12_surface_points_df.csv"),
                                         orientations=pd.read_csv(
                          cwd + "/examples/input_data/model12_orientations_df.csv"),
                                         mapping_object={
                          "Strat_Series1": ('rock4', 'rock3'),
                          "Strat_Series2": ('rock2', 'rock1')},
                                         )

#%%

frame, block = combined_interpolator(input_data=data_test)

#%%


#%%

# plot section of scalar field for group "Strat_Series1"
import matplotlib.pyplot as plt
group = frame["Strat_Series1"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series1"]["rock3"].scalar_value, frame["Strat_Series1"]["rock4"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

group = frame["Strat_Series2"]
plt.imshow(group.scalar_field[:, 0, :], cmap='viridis', origin='lower')
# add contour lines for scalar values
contour_levels = [frame["Strat_Series2"]["rock1"].scalar_value, frame["Strat_Series2"]["rock2"].scalar_value]
contour = plt.contour(group.scalar_field[:, 0, :], levels=contour_levels, colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

plt.colorbar()
plt.title("Scalar Field Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

#%%

# plot slices of masks
plt.imshow(frame["Strat_Series1"].mask[:, 0, :], cmap='gray', origin='lower')
plt.colorbar()
plt.title("Mask Section for Strat_Series1")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

plt.imshow(frame["Strat_Series2"].mask[:, 0, :], cmap='gray', origin='lower')
plt.colorbar()
plt.title("Mask Section for Strat_Series2")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()



#%%

# plot slice of block

import matplotlib.pyplot as plt
plt.imshow(block[:, 0, :], cmap='viridis', origin='lower')
plt.colorbar()
plt.title("Lithology Block Section")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()

#%%

frame.pretty_print()

#%%

# plot surface meshes
import pyvista as pv

pv.global_theme.allow_empty_mesh = True

# Create a PyVista plotter
plotter = pv.Plotter(notebook=False)

# loop over all elements from all groups in frame
for group in frame.structural_groups:
    for element in group.structural_elements:
        print(element.color)
        plotter.add_mesh(pv.PolyData(element.vertices['masked'],
                            np.insert(element.edges["masked"], 0, 3, axis=1).ravel()),
                            color=element.color, label=element.name)

plotter.add_legend(size=(0.13, 0.13), loc='lower right', face='circle')

# Set the bounds and grid of the plotter
plotter.show_bounds(bounds=data_test.extent,
                        location="furthest",
                        grid=True)

# Set the camera position
plotter.camera.view_angle = 30.0
plotter.camera.azimuth = 25.0
plotter.camera.elevation = -15.0

plotter.show()
