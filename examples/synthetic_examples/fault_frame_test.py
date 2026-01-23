from typing import List, Optional
from pydantic import BaseModel, PrivateAttr
import numpy as np
import pandas as pd
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid
import gempy as gp

from core.utility.surface_mesh_extraction import marching_cubes_new, marching_cubes_per_element

from core.visualization_components_new import visualize_fault_frame

from concepts.archive.objects import StructuralFrame


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
    _domain_masks: dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

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

    @property
    def domain_masks(self) -> dict[int, np.ndarray]:
        """Boolean masks for each final domain ID."""
        return self._domain_masks

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

    def check_fault_crosscuts_via_isovalue_bands(
            self,
            thickness_world: float | None = None,
            voxels: float = 1.0,
            use_gradient: bool = True,
    ) -> None:
        """
        Detect cross-cutting faults by overlapping 'isovalue bands' around each fault's own scalar isovalue.

        For each fault i with scalar field φ_i and isovalue L_i (fault.scalar_value):
            band_i = |φ_i - L_i| <= tol_scalar_i

        If any voxel satisfies band_i & band_j for i!=j, the faults crosscut.

        Parameters
        ----------
        thickness_world : float | None
            Desired half-thickness (in world units, e.g. meters) of each isovalue band.
            If None, it is computed as `voxels * min(grid.spacing)`.
        voxels : float
            If `thickness_world` is None, use this many voxels (based on min spacing) as the half-thickness.
        use_gradient : bool
            If True (recommended), convert the world thickness to scalar tolerance per-fault using
            that fault's median gradient magnitude: tol_scalar_i = thickness_world * median(|∇φ_i|).
            If False, assumes φ is approximately a signed distance function and uses tol_scalar_i = thickness_world.

        Raises
        ------
        ValueError
            If any pair of faults' bands overlap (i.e., cross-cut is detected).
        """

        if self._grid is None:
            raise ValueError("FaultFrame grid must be set.")
        spacing = getattr(self._grid, "spacing", None)
        if spacing is None:
            raise ValueError("Grid.spacing must be defined to compute band thickness.")

        # Determine band thickness in world units (meters)
        if thickness_world is None:
            thickness_world = float(voxels) * float(np.min(spacing))

        # Collect faults that have scalar fields and an isovalue
        faults = []
        for f in self._fault_elements:
            field = getattr(f, "scalar_field", None)
            level = getattr(f, "scalar_value", None)
            if field is None or level is None:
                continue
            if not isinstance(field, np.ndarray) or field.size == 0:
                continue
            faults.append((f.name, field, float(level)))

        if len(faults) < 2:
            return  # nothing to compare

        # Compute per-fault scalar tolerances from world thickness
        tol_scalar = []
        for nm, fld, _ in faults:
            if use_gradient:
                # gradient in scalar units per meter along each axis
                gx, gy, gz = np.gradient(fld, *spacing, edge_order=1)
                grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
                med = float(np.nanmedian(grad_mag)) if np.isfinite(grad_mag).any() else 0.0
                # guard against tiny gradients
                if med <= 1e-12:
                    med = 1e-12
                tol_scalar.append(thickness_world * med)
            else:
                # assume φ is approx. signed distance
                tol_scalar.append(thickness_world)

        # Build boolean bands once
        bands = []
        for (nm, fld, level), ts in zip(faults, tol_scalar):
            band = np.abs(fld - level) <= ts
            bands.append((nm, band))

        # Pairwise overlap test
        for i in range(len(bands)):
            name_i, band_i = bands[i]
            if not band_i.any():
                continue
            for j in range(i + 1, len(bands)):
                name_j, band_j = bands[j]
                if not band_j.any():
                    continue
                if np.any(band_i & band_j):
                    raise ValueError(f"❌ Fault '{name_i}' crosscuts fault '{name_j}' (isovalue-band overlap).")

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

            # Extract surface point/orientation input_data for this fault
            points = self.get_surface_points_for_element(name)
            orientations = self.get_orientations_for_element(name)

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

        # Remap domain IDs to consecutive values starting from 0
        unique_ids = np.unique(domain_map)
        remap = {old: new for new, old in enumerate(unique_ids)}
        remapped_map = np.vectorize(remap.get)(domain_map)
        self._domain_map = remapped_map

        #  Store per-domain masks
        self._domain_masks = {}
        for uid in np.unique(remapped_map):
            self._domain_masks[uid] = remapped_map == uid

        # Extrac surfaces meshes for faults
        for i, fault in enumerate(reversed(self._fault_elements)):
            vertices, edges = marching_cubes_new(fault.scalar_field.T,
                                                 [fault.scalar_value],
                                                 grid.spacing,
                                                 grid.extent)

            fault.set_vertices(vertices[0])
            fault.set_edges(edges[0])

        # 🔎 After all faults are processed
        self.check_fault_crosscuts_via_isovalue_bands()


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
    Build a FaultFrame from ordered fault names, surface input_data, and a grid.

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
    resolution=(50, 50, 50)  # Example resolution
)

fault_names = ["FaultC", "FaultA", "FaultB"]  # FaultB is younger than FaultA
fault_colors = ["#A9A9A9", "#A9A9A9", "#A9A9A9"]

fault_frame = build_fault_frame(
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

from core.structuralmodeling_components.interpolators_per_group import general
from core.visualization_components_new import visualize_structural_frame, plot_structural_slice

#%%

# Create a StructuralFrame
frame = general.build_structural_frame({"Top": ('UnitD', 'UnitC'), "Bot": ('UnitB', 'UnitA')},
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

from core.structuralmodeling_components.interpolators_per_group.universal_cokriging_per_group import \
    interpolate_group_universal_cokriging

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


# Preparation: Assign domain IDs to surface points and orientations
sp_in_domain = assign_domain_ids_to_points(grid, domain_map, surface_points)
ori_in_domain = assign_domain_ids_to_points(grid, domain_map, orientations)

# TODO: This is where I need to set the storage options
scalar_fields_per_domain = {}
scalar_values_per_domain = {}
masks_per_domain = dict()  # {domain_id: {group_name: lith_mask}}

for domain_id in domain_ids:
    print(f"🔎 Processing domain {domain_id}")

    # Filter input input_data for this domain
    sp_filtered = sp_in_domain[sp_in_domain["domain_id"] == domain_id].drop(columns="domain_id")
    ori_filtered = ori_in_domain[ori_in_domain["domain_id"] == domain_id].drop(columns="domain_id")

    # Check sufficient points per element
    for group in groups:
        for elem in group.structural_elements:
            count = len(sp_filtered[sp_filtered["formation"] == elem.name])
            if count < 2:
                raise ValueError(f"❌ Not enough surface points for '{elem.name}' in domain {domain_id}")

    # Copy groups to avoid mutating original
    domain_groups = deepcopy(groups)

    # Initialize storage for this domain
    scalar_fields_per_domain[domain_id] = {}
    scalar_values_per_domain[domain_id] = {}

    # Inside the loop for each domain:
    masks_per_domain[domain_id] = {}

    for group in domain_groups:
        interpolate_group_universal_cokriging(
            group=group,
            grid=frame.grid,
            group_surface_points_df=sp_filtered,
            group_orientations_points_df=ori_filtered,
        )

        # Store scalar field
        scalar_fields_per_domain[domain_id][group.name] = group.scalar_field

        if group.name not in scalar_values_per_domain[domain_id]:
            scalar_values_per_domain[domain_id][group.name] = {}

        for elem in group.structural_elements:
            scalar_values_per_domain[domain_id][group.name][elem.name] = elem.scalar_value

masks_per_domain = set_scalar_masks_per_domain(
    structural_frame=frame,
    scalar_fields_per_domain=scalar_fields_per_domain,
    scalar_values_per_domain=scalar_values_per_domain,
)

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

import numpy as np


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
# Check for input_data in each fault domain/group (what happens if no input_data for group in domain)

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
