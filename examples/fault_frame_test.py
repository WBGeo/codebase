from typing import List, Optional
from pydantic import BaseModel, PrivateAttr
import numpy as np
import pandas as pd
from core.grids.grid_classes import RegularGrid
import gempy as gp


#%%
class FaultElement(BaseModel):
    """
    Represents a geological fault surface to be interpolated.

    Attributes:
        name (str): Unique identifier for the fault.
        scalar_value (Optional[float]): Value used in scalar field interpolation.
        scalar_field (Optional[np.ndarray]): Interpolated scalar field values on the fault surface.
        affects_groups (Optional[List[str]]): Structural groups offset by this fault.
        color (str): Display color for the fault in hex format (default: "#AAAAAA").
        vertices (np.ndarray): Coordinates of the fault surface vertices.
        edges (np.ndarray): Connectivity of the fault surface edges.
        mask (Optional[np.ndarray]): Boolean mask separating two fault blocks.
    """
    _name: str = PrivateAttr()
    _scalar_value: Optional[float] = PrivateAttr(default=None)
    _scalar_field: Optional[np.ndarray] = PrivateAttr(default=None)
    _affects_groups: Optional[List[str]] = PrivateAttr(default=None)
    _color: str = PrivateAttr(default="#AAAAAA")  # Default color in hex format
    _vertices: np.ndarray = PrivateAttr(default_factory=None)
    _edges: np.ndarray = PrivateAttr(default_factory=None)
    _mask: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, name: str, scalar_value: Optional[float] = None,
                 affects_groups: Optional[List[str]] = None):
        super().__init__()
        self._name = name
        self._scalar_value = scalar_value
        self._affects_groups = affects_groups

    def __repr__(self):
        return f"FaultElement(name='{self.name}')"

    @property
    def name(self) -> str:
        return self._name

    @property
    def scalar_value(self) -> Optional[float]:
        return self._scalar_value

    @property
    def color(self) -> str:
        return self._color

    @property
    def vertices(self) -> Optional[np.ndarray]:
        return self._vertices

    @property
    def edges(self) -> Optional[np.ndarray]:
        return self._edges

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def scalar_field(self) -> Optional[np.ndarray]:
        """Get the interpolated scalar field values on the fault surface."""
        return self._scalar_field

    def set_scalar_field(self, scalar_field: np.ndarray):
        """Assign the interpolated scalar field values on the fault surface."""
        if not isinstance(scalar_field, np.ndarray):
            raise ValueError("Scalar field must be a numpy array.")
        self._scalar_field = scalar_field

    def set_scalar_value(self, value: float):
        """Assign scalar value used for interpolation."""
        self._scalar_value = value

    def set_color(self, color: str):
        """Set the display color for this fault in hex format."""
        if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
            raise ValueError("Color must be a valid hex string (e.g., '#RRGGBB').")
        self._color = color

    def set_vertices(self, vertices: np.ndarray):
        """Assign the coordinates of the fault surface vertices."""
        if not isinstance(vertices, np.ndarray):
            raise ValueError("Vertices must be a numpy array.")
        self._vertices = vertices

    def set_edges(self, edges: np.ndarray):
        """Assign the connectivity of the fault surface edges."""
        if not isinstance(edges, np.ndarray):
            raise ValueError("Edges must be a numpy array.")
        self._edges = edges

    def set_domain_mask(self, mask: np.ndarray):
        """
            Store the boolean mask (True/False) separating two fault blocks.
            """
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Mask must be a boolean NumPy array.")
        self._mask = mask

    def get_domain_mask(self) -> np.ndarray:
        """
            Retrieve the fault mask (True = one block, False = other).
            """
        if self._mask is None:
            raise ValueError(f"No mask set for fault '{self.name}'.")
        return self._mask

    def get_inverse_domain_mask(self) -> np.ndarray:
        """
            Get the inverse of the fault mask (opposite block).
            """
        return ~self.get_domain_mask()

    @property
    def affects_groups(self) -> Optional[List[str]]:
        return self._affects_groups

    def set_affects_groups(self, groups: List[str]):
        """Specify which structural groups are offset by this fault."""
        self._affects_groups = groups


class FaultFrame(BaseModel):
    """
    Container for managing fault elements and their relationships.

    Attributes:
        fault_elements (List[FaultElement]): Ordered list of faults (oldest to youngest).
        fault_relations (np.ndarray): Boolean matrix [younger_idx, older_idx] = True if younger offsets older.
        grid (Optional[RegularGrid]): Regular grid for spatial context.
        fault_surface_points_df (Optional[pd.DataFrame]): DataFrame with fault surface points.
        fault_orientations_df (Optional[pd.DataFrame]): DataFrame with fault surface orientations.
        domain_map (Optional[np.ndarray]): Map of fault domains for scalar field interpolation.
    """
    _fault_elements: List[FaultElement] = PrivateAttr()
    _fault_relations: np.ndarray = PrivateAttr()
    _grid: Optional[RegularGrid] = PrivateAttr(default=None)
    _fault_surface_points_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _fault_orientations_df: Optional[pd.DataFrame] = PrivateAttr(default=None)
    _domain_map: Optional[np.ndarray] = PrivateAttr(default=None)

    def __init__(self, fault_elements: List[FaultElement], fault_relations: Optional[np.ndarray] = None):
        super().__init__()
        self._fault_elements = fault_elements
        self._fault_relations = (
            fault_relations if fault_relations is not None else self._generate_default_relations()
        )

    @property
    def fault_elements(self) -> List[FaultElement]:
        return self._fault_elements

    @property
    def fault_relations(self) -> np.ndarray:
        return self._fault_relations

    @property
    def grid(self) -> RegularGrid:
        return self._grid

    @property
    def fault_surface_points_df(self) -> Optional[pd.DataFrame]:
        return self._fault_surface_points_df

    @property
    def fault_orientations_df(self) -> Optional[pd.DataFrame]:
        return self._fault_orientations_df

    @property
    def domain_map(self) -> Optional[np.ndarray]:
        return self._domain_map

    def _generate_default_relations(self) -> np.ndarray:
        """By default, younger faults affect all older ones."""
        n = len(self._fault_elements)
        relations = np.zeros((n, n), dtype=bool)
        for younger in range(n):
            for older in range(younger):
                relations[younger, older] = True
        return relations

    def get_element_by_name(self, name: str) -> Optional[FaultElement]:
        """Retrieve a fault element by its name."""
        return next((f for f in self._fault_elements if f.name == name), None)

    def add_fault_element(self, fault: FaultElement):
        """Append a fault and update the relations matrix accordingly."""
        self._fault_elements.append(fault)
        self._fault_relations = self._generate_default_relations()

    def set_fault_relation(self, younger_idx: int, older_idx: int, value: bool):
        """Manually modify a fault-fault relation."""
        self._fault_relations[younger_idx, older_idx] = value

    def set_surface_points_df(self, df: pd.DataFrame):
        self._fault_surface_points_df = df

    def set_orientations_df(self, df: pd.DataFrame):
        self._fault_orientations_df = df

    def get_surface_points_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface points."""
        return self._fault_surface_points_df

    def get_orientations_df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame containing all fault surface orientations."""
        return self._fault_orientations_df

    def get_surface_points_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_surface_points_df is not None:
            return self._fault_surface_points_df[self._fault_surface_points_df["formation"] == name]
        return pd.DataFrame()

    def get_orientations_for_element(self, name: str) -> pd.DataFrame:
        if self._fault_orientations_df is not None:
            return self._fault_orientations_df[self._fault_orientations_df["formation"] == name]
        return pd.DataFrame()

    def set_domain_map(self, domain_map: np.ndarray):
        self._domain_map = domain_map

    def set_grid(self, grid: RegularGrid):
        """Set the grid for spatial context."""
        if not isinstance(grid, RegularGrid):
            raise ValueError("Grid must be an instance of RegularGrid.")
        self._grid = grid

    def describe_relations(self) -> List[str]:
        """Return a readable list of which faults offset which others."""
        descriptions = []
        names = [f.name for f in self._fault_elements]
        for y in range(len(names)):
            for o in range(len(names)):
                if self._fault_relations[y, o]:
                    descriptions.append(f"{names[y]} offsets {names[o]}")
        return descriptions

    def detailed_report(self):
        print("🧱 Fault Frame — Detailed Report")
        print("────────────────────────────────")
        print(f"• Number of faults: {len(self.fault_elements)}")

        if self._grid:
            print(f"• Grid extent: {self._grid.extent}")
            print(f"• Grid resolution: {self._grid.resolution}")
        else:
            print("• Grid: Not set")

        if self._fault_surface_points_df is not None:
            print(f"• Surface points: {len(self._fault_surface_points_df)} entries")
        else:
            print("• Surface points: None")

        if self._fault_orientations_df is not None:
            print(f"• Orientations: {len(self._fault_orientations_df)} entries\n")
        else:
            print("• Orientations: None\n")

        print("▶ Faults (youngest → oldest):")
        for fault in reversed(self.fault_elements):
            name = fault.name
            color = fault.color or "#888888"
            try:
                r, g, b = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                colored_name = f"\033[38;2;{r};{g};{b}m{name}\033[0m"
            except Exception:
                colored_name = name

            sp_count = len(self.get_surface_points_for_element(name))
            ori_count = len(self.get_orientations_for_element(name))

            print(f"  ├─ {colored_name}")
            print(f"  │   ├─ Surface points: {sp_count}")
            print(f"  │   └─ Orientations: {ori_count}")
        print("")


    def generate_fault_domains(self) -> None:
        """
        Interpolates all faults and generates a domain map across the model grid.
        Relies on fault.domain_mask being set by the interpolator_func.
        """
        if not self._grid:
            raise ValueError("Grid must be set before domain generation.")
        if self._fault_surface_points_df is None:
            raise ValueError("Fault surface points must be set.")

        # Initialize single-domain model
        domain_map = np.zeros(self._grid.resolution, dtype=int)
        domain_id_counter = 1

        temp_ids = []  # Track temporary domain IDs before remapping

        # Interpolate faults from youngest to oldest
        for i, fault in enumerate(reversed(self._fault_elements)):  # Youngest first
            name = fault.name

            # Extract surface point/orientation data for this fault
            points = self.get_surface_points_for_element(name)
            orientations = self.get_orientations_for_element(name)

            # points = self._fault_surface_points_df[
            #     self._fault_surface_points_df["formation"] == name
            #     ]
            # orientations = (
            #     self._fault_surface_orientations_df[
            #         self._fault_surface_orientations_df["formation"] == name
            #         ]
            #     if self._fault_surface_orientations_df is not None else pd.DataFrame()
            # )

            if points.empty:
                raise ValueError(f"❌ No surface points found for fault '{name}'.")

            # Run interpolation (sets scalar field, scalar value, mask internally)
            interpolate_group_universal_cokriging_for_faults(fault_frame.get_element_by_name(name),
                                                             grid,
                                                             fault_surface_points_df=points,
                                                             fault_orientations_points_df=orientations)

            if fault.get_domain_mask() is None:
                raise ValueError(f"❌ Interpolator did not set domain_mask for fault '{name}'.")

            fault_mask = fault.get_domain_mask()
            new_domain_map = domain_map.copy()

            # For each existing domain, split it if affected by this fault
            for existing_id in np.unique(domain_map):
                current_mask = domain_map == existing_id
                overlap = current_mask & fault_mask

                if np.any(overlap):
                    # Assign a temporary large ID
                    new_domain_map[overlap] = 9999 + domain_id_counter
                    temp_ids.append(9999 + domain_id_counter)
                    domain_id_counter += 1

            domain_map = new_domain_map

        # Remap domain IDs to consecutive values starting from 0
        unique_ids = np.unique(domain_map)
        remap = {old: new for new, old in enumerate(unique_ids)}
        remapped_map = np.vectorize(remap.get)(domain_map)
        self._domain_map = remapped_map


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


def build_fault_frame(
        fault_surface_points_df: pd.DataFrame,
        fault_orientations_df: pd.DataFrame,
        fault_names: list,  # youngest to oldest
        grid: RegularGrid,
        colors: list = None
) -> FaultFrame:
    """
    Build a FaultFrame from ordered fault names, surface data, and a grid.

    Args:
        fault_surface_points_df (pd.DataFrame): ['X', 'Y', 'Z', 'formation'].
        fault_orientations_df (pd.DataFrame): ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'formation'].
        fault_names (list): Fault names ordered from youngest to oldest.
        grid (RegularGrid): Model grid.
        colors (list, optional): Hex colors for faults, same order. Defaults to dark grey.

    Returns:
        FaultFrame
    """
    if colors is None:
        colors = ["#555555"] * len(fault_names)
    if len(colors) != len(fault_names):
        raise ValueError("Length of colors must match fault_names")

    fault_elements = []
    for name, color in reversed(list(zip(fault_names, colors))):  # oldest to youngest internally
        fault = FaultElement(name=name)
        fault.set_color(color)
        fault_elements.append(fault)

    fault_frame = FaultFrame(fault_elements=fault_elements)
    fault_frame.set_surface_points_df(fault_surface_points_df)
    fault_frame.set_orientations_df(fault_orientations_df)
    fault_frame.set_grid(grid)

    return fault_frame

def interpolate_group_universal_cokriging_for_faults(
        element: FaultElement,
        grid,
        fault_surface_points_df: pd.DataFrame,  # Only points relevant to this group
        fault_orientations_points_df=pd.DataFrame,  # Only orientations relevant to this group
) -> None:
    if fault_surface_points_df.empty:
        raise ValueError(f"No surface points provided for {element.name}")

    if fault_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for {element.name}")

    # 3. Convert to gempy
    surface_data = gp.data.surface_points.SurfacePointsTable.from_arrays(
        x=fault_surface_points_df.X.to_numpy(),
        y=fault_surface_points_df.Y.to_numpy(),
        z=fault_surface_points_df.Z.to_numpy(),
        names=fault_surface_points_df.formation.to_numpy(),
        nugget=np.zeros(len(fault_surface_points_df)),
        name_id_map=None)

    orientation_data = gp.data.orientations.OrientationsTable.from_arrays(
        x=fault_orientations_points_df.X.to_numpy(),
        y=fault_orientations_points_df.Y.to_numpy(),
        z=fault_orientations_points_df.Z.to_numpy(),
        G_x=fault_orientations_points_df.G_x.to_numpy(),
        G_y=fault_orientations_points_df.G_y.to_numpy(),
        G_z=fault_orientations_points_df.G_z.to_numpy(),
        names=fault_orientations_points_df.formation.to_numpy(),
        nugget=np.zeros(len(fault_orientations_points_df)),
        name_id_map=surface_data.name_id_map)

    gempy_structural_frame = gp.data.structural_frame.StructuralFrame.from_data_tables(surface_data, orientation_data)

    # Create a GeoModel instance
    geo_model = gp.create_geomodel(
        project_name="random",
        extent=grid.extent,
        resolution=grid.resolution,
        structural_frame=gempy_structural_frame
    )

    mapping = {"fault_group": [element.name]}

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object=mapping
    )

    # Compute the geological model
    gp.compute_model(geo_model)

    # Set scalar value for the fault element
    element.set_scalar_value(geo_model.solutions.raw_arrays.scalar_field_at_surface_points[0][0])

    # Set scalar field result in group (k3d1 is a numpy array with shape matching grid)
    element.set_scalar_field(geo_model.solutions.raw_arrays.scalar_field_matrix[0].reshape(grid.resolution).T)

    # Set the domain mask based on the scalar field
    element.set_domain_mask(element.scalar_field > element.scalar_value)

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
fault_colors = ["#FF0000", "#00FF00", "#00FF00"]

fault_frame = build_fault_frame(
    fault_surface_points_df=fault_surface_points_df,
    fault_orientations_df=fault_orientations_df,
    fault_names=fault_names,
    colors=fault_colors,
    grid=grid
)

#%%

fault_frame._fault_orientations_df

#%%

fault_frame.get_orientations_df()

#%%
fault_frame.detailed_report()


#%%
fault_frame.get_orientations_for_element("FaultA")

#%%

fault_frame.get_surface_points_for_element("FaultA")

#%%

fault_frame.generate_fault_domains()

#%%

fault_frame.detailed_report()

#%%

# Plot slice of the scalar field for FaultA
import matplotlib.pyplot as plt


# Reshape scalar field to match grid resolution
scalar_field_reshaped = fault_frame.get_element_by_name("FaultA")._scalar_field.reshape(grid.resolution)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(scalar_field_reshaped[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
# add contour lines for a single contour at the scalar value
ax.contour(scalar_field_reshaped[:,25,:],
           extent=grid.extent[:4],
           levels=[fault_frame.get_element_by_name("FaultA")._scalar_value],
           colors='red',
           linewidths=1.5)
# ax.colorbar(label='Scalar Field Value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

#%%



#%%

# Reshape scalar field to match grid resolution
mask_reshaped = fault_frame.get_element_by_name("FaultC")._mask.reshape(grid.resolution)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(mask_reshaped[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
# ax.colorbar(label='Scalar Field Value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

#%%

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(fault_frame.domain_map[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
# ax.colorbar(label='Scalar Field Value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

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

def generate_random_surface_points(n: int, formations: list[str]) -> pd.DataFrame:
    """
    Generate a DataFrame with n randomly scattered surface points within model extent.

    Args:
        n (int): Number of points to generate.
        formations (list of str): Formation names to assign to points (cycled if fewer than n).

    Returns:
        pd.DataFrame: DataFrame with columns X, Y, Z, formation.
    """
    data = {
        "X": np.random.uniform(0, 1000, n),
        "Y": np.random.uniform(0, 1000, n),
        "Z": np.random.uniform(0, 1000, n),
        "formation": [formations[i % len(formations)] for i in range(n)]
    }
    return pd.DataFrame(data)

#%%

# Example usage:
formations = ["UnitA", "UnitB", "UnitC"]
test_surface_points = generate_random_surface_points(100, formations)

#%%


# Assuming you have a StructuralFrame and FaultFrame
surface_points = assign_domain_ids_to_points(fault_frame.grid, fault_frame.domain_map, test_surface_points)
# orientations = assign_domain_ids_to_points(fault_frame.grid, fault_frame.domain_map, structural_frame.orientations)

#%%
surface_points.head()

#%%

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(fault_frame.domain_map[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
ax.contour(fault_frame.domain_map[:,25,:],levels=np.unique(fault_frame.domain_map), extent=grid.extent[:4], origin='lower', cmap='viridis')
ax.scatter(surface_points['X'], surface_points['Z'], c=surface_points['domain_id'], cmap='viridis', s=10, alpha=1)
ax.set_aspect("equal")
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

#%%

plt.scatter(surface_points['X'], surface_points['Z'], c=surface_points['domain_id'], cmap='viridis', s=10, alpha=1)
plt.colorbar()
plt.show()

#%%

# filter surface points df for domain id==3
domain_id = 3
filtered_points = surface_points[surface_points['domain_id'] == domain_id]
filtered_points.head()

#%%

surface_points.head()

#

# think about how to combine this with the structural frame
# how to combine and store scalar fields
# how to extract meshes if unit stretches over multiple fault blocks

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

from core.interpolator_components.interpolators_per_group import general
from core.visualization_components_new import visualize_structural_frame, plot_structural_slice

#%%

# Create a StructuralFrame
frame = general.build_structural_frame({"Top": ('UnitD', 'UnitC'),"Bot": ('UnitB', 'UnitA')},
                                        np.array([0, 1000, 0, 1000, 0, 1000]),
                                        np.array([50, 50, 50]),
                                        structural_surface_points_df,
                                        structural_orientations_df)
frame.detailed_report()


#%%

# Plot a slice of the structural model
plot_structural_slice(frame, axis='y', index=10, show_scalar_contours=True)

# Visualize the structural frame with options for surface meshes, points, and orientations
# visualize_structural_frame(frame, show_points=True, show_orientations=True, notebook=False, show=True)

#%%

frame["Top"].set_interpolation_method("Universal Co-Kriging")
frame["Bot"].set_interpolation_method("Universal Co-Kriging")

# frame["Top"].set_interpolation_method("Ordinary Kriging")
# frame["Bot"].set_interpolation_method("Ordinary Kriging")
#
# frame["Top"].configure_interpolation_params(range=10000, anisotropy_scaling_z=0.3)
# frame["Bot"].configure_interpolation_params(range=10000, anisotropy_scaling_z=0.3)

#%%

frame.detailed_report()

#%%

# Compute solution
frame, block = general.combined_interpolator(frame)

#%%

# Plot a slice of the structural model
plot_structural_slice(frame, lith_block=block, axis='y', index=0, show_scalar_contours=True)


#%%

# Visualize the structural frame with options for surface meshes, points, and orientations
visualize_structural_frame(frame, show_surface_meshes=True, show_points=True, show_orientations=True, notebook=False, show=True)

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

frame.structural_groups

#%%

def combine_group_scalar_fields(groups):
    # Assume each group has `scalar_field`, and elements have `scalar_value`
    combined = np.full(groups[0].scalar_field.shape, fill_value=-1.0)
    for group in groups:
        group_field = group.scalar_field
        for elem in group.structural_elements:
            value = elem.scalar_value
            mask = group_field >= value  # Or your logic
            combined[mask] = value
    return combined

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

for domain_id in domain_ids:
    print(f"🔎 Processing domain {domain_id}")

    # 2a. Filter surface points and orientations to this domain
    sp_in_domain = assign_domain_ids_to_points(grid, domain_map, surface_points)
    ori_in_domain = assign_domain_ids_to_points(grid, domain_map, orientations)

    sp_filtered = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
    ori_filtered = ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")

    # Plot domain map slice and surface points
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # ax.imshow(fault_frame.domain_map[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
    # ax.contour(fault_frame.domain_map[:, 25, :], levels=np.unique(fault_frame.domain_map), extent=grid.extent[:4],
    #            origin='lower', cmap='viridis')
    # ax.scatter(sp_in_domain['X'], sp_in_domain['Z'], c=sp_in_domain['domain_id'], cmap='viridis', s=10, alpha=1)
    # ax.set_aspect("equal")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.show()

    # 2b. Check data sufficiency per group
    for group in groups:
        element_names = [e.name for e in group.structural_elements]
        for name in element_names:
            sp_count = len(sp_filtered[sp_filtered["formation"] == name])
            if sp_count < 2:
                raise ValueError(f"❌ Not enough surface points for element '{name}' in domain {domain_id}")

    # 2c. Deepcopy group to avoid overwriting
    domain_groups = deepcopy(groups)

    print(domain_groups)
    # TODO: for some reason looks like this result uses all points, at least its not flat
    for group in domain_groups:
        print(group.name)
        interpolate_group_universal_cokriging(
            group=group,
            grid=frame.grid,
            group_surface_points_df=sp_filtered,
            group_orientations_points_df=ori_filtered,
        )
        # scatter plot of used points
        plt.scatter(sp_filtered['X'], sp_filtered['Z'], c=sp_filtered['formation'].astype('category').cat.codes, cmap='viridis', s=10, alpha=1)
        plt.title(f"Domain {domain_id} - Group {group.name} Surface Points")
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.show()

        # slice of scalar field
        # scalar_field = group.scalar_field.reshape(grid.resolution)
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.imshow(scalar_field[:, 25, :], extent=grid.extent[:4], origin='lower', cmap='viridis')
        # ax.scatter(sp_filtered['X'], sp_filtered['Z'], c=sp_filtered['formation'].astype('category').cat.codes, cmap='viridis', s=10, alpha=1)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # plt.title(f"Domain {domain_id} - Group {group.name} Scalar Field")
        # plt.show()



        # interpolate_group_ordinary_kriging(
        #     group=group,
        #     group_surface_points_df=sp_filtered,
        #     grid=frame.grid
        # )

    # 5. Create masks based on order of structural groups, scalar fields and scalar values
    set_scalar_masks(frame)

    print("Masking done")

    # 6. Create a combined result (lith_block) based on masks, scalar fields and scalar values
    lith_block = compute_lithology_block(frame)

    # Plot slice of the lith block
    lith_block_reshaped = lith_block.reshape(grid.resolution)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(lith_block_reshaped[:, 0, :], extent=grid.extent[:4], origin='lower', cmap='viridis')
    ax.scatter(sp_filtered['X'], sp_filtered['Z'], c=sp_filtered['formation'].astype('category').cat.codes, cmap='viridis', s=10, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.title(f"Domain {domain_id} Lith Block")
    plt.show()

    final_lith_blocks.append((domain_id, lith_block))

# 3️⃣ Combine all lith blocks into a single model
# You might resolve overlaps using the age of groups or just stack in order
final_model = np.full(grid.resolution, fill_value=-1)

for domain_id, lith_block in final_lith_blocks:
    mask = domain_map == domain_id
    final_model[mask] = lith_block[mask]

print("✅ Final lithology model constructed.")


#%%

final_model.shape

# plot slice of final model
import matplotlib.pyplot as plt

# Reshape final model to match grid resolution
final_model_reshaped = final_model.reshape(grid.resolution)
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(final_model_reshaped[:,25,:], extent=grid.extent[:4], origin='lower', cmap='viridis')
# add contour lines for a single contour at the scalar value
ax.contour(final_model_reshaped[:,25,:],
           extent=grid.extent[:4],
           levels=np.unique(final_model_reshaped),
           colors='red',
           linewidths=1.5)
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.show()

#%%