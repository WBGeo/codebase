import typing
import re
import gmsh
import meshio
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from typing import Sequence, Tuple, List, Mapping, Optional, Dict, Set, Any
from numpy.typing import NDArray
from core.object_components import StructuralModelResults, ExtentData, MeshResults
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import create_surface_grid, import_surfaces, fragment_surfaces
from core.meshing_components.explicit.unstructured.create_clean_surface import data_prepration
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType, wbgeo_type, wbgeo_inspector, \
  InspectorHelper
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d

from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType




def point_on_line_segment(pt: Tuple[float, float, float], p1: Tuple[float, float, float], p2: Tuple[float, float, float],
    tol: float = 1e-6) -> bool:
    """
    Check whether a point is collinear with a line defined by two points.

    This function verifies that `pt` lies on the *infinite line* passing
    through `p1` and `p2` within a given tolerance. It does NOT check whether
    the point lies between `p1` and `p2` (i.e., inside the line segment).

    Args:
    pt (tuple(float, float, float)): Point to be tested.
    p1, p2 (tuple(float, float, float)): Two distinct points defining the line.
    tol (float, optional): Numerical tolerance used for collinearity checks.

    Returns:
    bool: True if the point is collinear with the line (within tolerance), False otherwise.
    """
    pt = np.array(pt, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    ratios = []

    for i in range(3):
        delta = p2[i] - p1[i]

        if abs(delta) > tol:
            ratio = (pt[i] - p1[i]) / delta
            ratios.append(ratio)
        else:
            # Line is (almost) constant in this coordinate
            if abs(pt[i] - p1[i]) > tol:
                return False

    # If only one (or zero) ratio is available, the line is axis-aligned
    if len(ratios) <= 1:
        return True

    return max(ratios) - min(ratios) < tol



def _check_extent_format(extent: List[float]) -> None:
    """
    Validate the format of a spatial extent.

    Args:
    extent (List[float]): Spatial extent defined as: [xmin, xmax, ymin, ymax, zmin, zmax]

    Raises:
    ValueError: If extent is None or does not contain exactly 6 values.
    """
    if extent is None:
        raise ValueError("Extent is None")

    if len(extent) != 6:
        raise ValueError(
            f"Extent must be [xmin, xmax, ymin, ymax, zmin, zmax], got {extent}"
        )


def point_inside_extent(point: Tuple[float, float, float], extent: List[float]) -> bool:
    """
    Check whether a 3D point lies inside or on the boundary of an extent.

    Args:
    point (tuple(float, float, float)): The point coordinates (x, y, z).
    extent (List[float]): Spatial extent defined as:[xmin, xmax, ymin, ymax, zmin, zmax]

    Returns
    bool : True if the point is inside or on the boundary of the extent, False otherwise.
    """
    x, y, z = point
    xmin, xmax, ymin, ymax, zmin, zmax = extent

    return (
        xmin <= x <= xmax and
        ymin <= y <= ymax and
        zmin <= z <= zmax
    )



def assert_point_inside_extent(point: Tuple[float, float, float], extent: List[float], obj_type: str = "",
    obj_id: Optional[int] = None, point_idx: Optional[int] = None) -> None:
    """
    Assert that a 3D point lies inside or on the boundary of a given extent.

    Args:
    point (tuple(float, float, float)): The (x, y, z) coordinates of the point to check.
    extent (List[float]): Model extent in the form [xmin, xmax, ymin, ymax, zmin, zmax].
    obj_type (str, optional): Type of object being checked (e.g., "Well", "Plane", "Fault").
    obj_id (int, optional): Identifier of the object (typically 1-based for user-facing messages).
    point_idx (int, optional): Zero-based index of the point within the object. Reported as 1-based
        in error messages for clarity.

    Raises:
    ValueError: If the point lies outside the given extent.
    """

    if not point_inside_extent(point, extent):
        msg: str = f"{obj_type}"

        if obj_id is not None:
            msg += f" '{obj_id}'"

        if point_idx is not None:
            msg += f", point #{point_idx + 1}"

        msg += f" at {tuple(point)} is outside extent {extent}"
        raise ValueError(msg)


def validate_wells(wells: List[Tuple[float, ...]], extent: List[float]) -> None:
    """
    Validate that all well points lie inside or on the boundary of the model extent.
    Each well is defined as a flat sequence of coordinates
    [x1, y1, z1, x2, y2, z2, ...], representing a polyline in 3D space.
    All points belonging to every well are checked individually.

    Args:
    wells List[Tuple[float, ...]]: Collection of wells, where each well is given as a flat sequenceof 3D coordinates (x, y, z).
    extent (List[float]): Model extent in the form [xmin, xmax, ymin, ymax, zmin, zmax].

    Raises:
    ValueError: If any point of any well lies outside the given extent.
            The error message specifies:
              - the well number, and
              - the point index within that well (1-based), and
              - the offending point coordinates.
    """

    _check_extent_format(extent)

    for i, well in enumerate(wells):

        # Convert flat coordinate list to (N, 3) array
        points: NDArray[np.float64] = np.asarray(well, dtype=float).reshape(-1, 3)

        for p_idx, (x, y, z) in enumerate(points):
            assert_point_inside_extent(
                (x, y, z),
                extent,
                obj_type="Well",
                obj_id=i + 1,       # user-facing well index
                point_idx=p_idx,    # user-facing point index handled inside helper
            )



def validate_sources(sources: List[Tuple[float, float, float]], extent: List[float]) -> None:
    """
    Validate that all point sources lie inside or on the boundary of the model extent.
    Each source is defined by a single 3D coordinate (x, y, z).
    All sources are checked individually against the model extent.

    Args:
    sources (List[Tuple[float, float, float]]): Collection of point sources, each defined by its (x, y, z) coordinates.
    extent (List[float]): Model extent in the form [xmin, xmax, ymin, ymax, zmin, zmax].

    Raises:
    ValueError: If any source lies outside the given extent.
            The error message specifies:
              - the source number (1-based), and
              - the offending point coordinates.
    """

    _check_extent_format(extent)

    for i, (x, y, z) in enumerate(sources):
        assert_point_inside_extent(
            (x, y, z),
            extent,
            obj_type="Source",
            obj_id=i + 1,  # user-facing index
        )



def validate_shafts(shafts: List[Dict[str, Tuple[float, ...]]], extent: List[float],) -> None:
    """
    Validate if shaft center lies inside or on the model extent.

    Args:
    shafts (List[Dict[str, Tuple[float, ...]]]): Collection of shaft definitions. Each shaft must be a dictionary with keys:
              - "center": (x, y, z)
              - "axis": (ax, ay, az)
              - "radius": float
    extent (List[float]): Model extent in the form[xmin, xmax, ymin, ymax, zmin, zmax].

    Raises:
    ValueError: If a shaft has a center outside the model extent.
    """

    _check_extent_format(extent)

    xmin, xmax, ymin, ymax, zmin, zmax = extent

    def assert_center_inside(
        point: Tuple[float, float, float],
        shaft_id: int,
    ) -> None:
        """
        Assert that a shaft center lies inside or on the model extent.
        """
        x, y, z = point
        if not (
            xmin <= x <= xmax and
            ymin <= y <= ymax and
            zmin <= z <= zmax
        ):
            raise ValueError(
                f"Shaft '{shaft_id}' center {tuple(point)} is outside extent {extent}"
            )

    for i, shaft in enumerate(shafts):

        center: NDArray[np.float64] = np.asarray(shaft["center"], dtype=float)
        axis:   NDArray[np.float64] = np.asarray(shaft["axis"],   dtype=float)
        radius: float = float(shaft["radius"])

        # ---- extent check (center only) ----
        assert_center_inside(tuple(center), i + 1)



def fit_plane(points: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit a best-fit plane to a set of 3D points using Singular Value Decomposition (SVD).

    The plane is defined in point–normal form: (x - p0) · n = 0
    where:
      - p0 is the centroid of the points,
      - n is the unit normal vector corresponding to the smallest singular value.

    Args:
    points (np.ndarray): Array of shape (N, 3) containing 3D point coordinates.

    Returns:
    normal (np.ndarray): Unit normal vector of the fitted plane, shape (3,).
    point_on_plane (np.ndarray): A point on the plane (the centroid), shape (3,).

    Raises:
    ValueError: If the input array does not have shape (N, 3).
    """

    pts: NDArray[np.float64] = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"fit_plane expects array of shape (N, 3), got {pts.shape}"
        )

    centroid: NDArray[np.float64] = pts.mean(axis=0)

    # SVD of centered coordinates
    _, _, vh = np.linalg.svd(pts - centroid)
    normal: NDArray[np.float64] = vh[-1]  # smallest singular value direction

    return normal / np.linalg.norm(normal), centroid



def max_point_plane_distance( points: NDArray[np.float64], normal: NDArray[np.float64], point_on_plane: NDArray[np.float64]) -> float:
    """
    Compute the maximum absolute distance of a set of points from a plane.
    The distance is computed as |(x - p0) · n|

    Args:
    points (np.ndarray): Array of shape (N, 3) containing 3D point coordinates.
    normal (np.ndarray): Unit normal vector of the plane, shape (3,).
    point_on_plane (np.ndarray): A point on the plane, shape (3,).

    Returns:
    float: Maximum absolute distance of the points to the plane.
    """

    pts: NDArray[np.float64] = np.asarray(points, dtype=float)
    dists: NDArray[np.float64] = np.abs((pts - point_on_plane) @ normal)
    return float(dists.max())




def validate_planes(planes: List[Tuple[float, ...]], extent:List[float], tol: float = 1e-8) -> None:
    """
    Validate planar surfaces defined by point sets.
    Validation rules:
      1) All points must lie inside or on the model extent.
      2) All points must be coplanar within a given tolerance.
    Plane indexing in error messages is 1-based (e.g., Plane #2 refers to the second plane in the input).

    Args:
    planes (iterable): Collection of planes. Each plane may be provided as:
              - a flat sequence (x1, y1, z1, x2, y2, z2, ...), or
              - a dict with a "coords" key storing the same flat sequence.
    extent (List[float]): Model extent in the form: [xmin, xmax, ymin, ymax, zmin, zmax].
    tol (float, optional): Maximum allowed deviation from planarity. Defaults to 1e-8.

    Raises:
    ValueError: If any plane:
              - contains points outside the extent,
              - or is not planar within the given tolerance.
    """

    # ---- extent unpacking ----
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    xmin, xmax, ymin, ymax, zmin, zmax = extent

    def assert_point_inside(point: Tuple[float, float, float], plane_no: int) -> None:
        """
        Assert that a plane point lies inside or on the model extent.

        Args:
        point (tuple(float, float, float)): Point coordinates (x, y, z).
        plane_no (int): 1-based plane index for user-facing error messages.

        Raises:
        ValueError: If the point lies outside the extent.
        """
        x: float
        y: float
        z: float
        x, y, z = point

        if not (
            xmin <= x <= xmax and
            ymin <= y <= ymax and
            zmin <= z <= zmax
        ):
            raise ValueError(
                f"Plane #{plane_no}, point {tuple(point)} "
                f"is outside extent {extent}"
            )

    # iterate over planes
    idx: int
    plane: List[Tuple[float, ...]]

    for idx, plane in enumerate(planes):
        plane_no: int = idx + 1

        # extract coordinates
        coords: Optional[Sequence[float]]

        if isinstance(plane, dict):
            coords = plane.get("coords")
        else:
            coords = plane


        # reshape into (N, 3)
        pts: NDArray[np.float64] = (
            np.asarray(coords, dtype=float).reshape(-1, 3)
        )

        # extent check
        p: NDArray[np.float64]
        for p in pts:
            assert_point_inside(
                (float(p[0]), float(p[1]), float(p[2])),
                plane_no,
            )

        # coplanarity check
        normal: NDArray[np.float64]
        p0: NDArray[np.float64]
        normal, p0 = fit_plane(pts)

        max_dev: float = max_point_plane_distance(pts, normal, p0)

        if max_dev > tol:
            raise ValueError(
                f"Plane #{plane_no} is NOT planar "
                f"(max deviation = {max_dev:.2e}, tol = {tol:.1e})"
            )




def validate_ellipses(ellipses: List[Dict], extent: Optional[List[float]] = None,) -> None:
    """
    Validate ellipses already loaded as dictionaries.

    Validation rules:
      1) 'center' and 'radii' must exist; radii must be positive
      2) Extent: ellipse must lie fully inside the model extent (conservative bounding sphere)

    Args:
        ellipses: list of dicts, each with 'center' and 'radii'
        extent: [xmin, xmax, ymin, ymax, zmin, zmax]

    Raises:
        ValueError if any ellipse fails validation
    """
    for idx, e in enumerate(ellipses, start=1):
        # --- Required fields ---
        if "center" not in e or "radii" not in e:
            raise ValueError(f"Ellipse #{idx} must have 'center' and 'radii' keys.")

        cx, cy, cz = e["center"]
        r1, r2 = e["radii"]

        if not all(np.isfinite([cx, cy, cz, r1, r2])):
            raise ValueError(f"Ellipse #{idx}: center and radii must be finite numbers.")
        if r1 <= 0 or r2 <= 0:
            raise ValueError(f"Ellipse #{idx}: radii must be positive.")

        # --- Extent check ---
        if extent is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = extent
            R = max(r1, r2)
            if not (
                xmin <= cx - R and cx + R <= xmax and
                ymin <= cy - R and cy + R <= ymax and
                zmin <= cz - R and cz + R <= zmax
            ):
                raise ValueError(
                    f"Ellipse #{idx}: center ({cx},{cy},{cz}) with max radius {R} "
                    f"extends outside extent {extent}"
                )




def validate_triangulation( points: np.ndarray, extent: Tuple[float, float, float, float, float, float],
    raise_error: bool = True) -> np.ndarray:
    """
    Validate that triangulation points lie inside a given spatial extent.

    Args:
        points (np.ndarray):
            Array of shape (n_points, 3) containing triangulation coordinates.

        extent (Tuple[float, float, float, float, float, float]):
            Bounding box defined as:
            (xmin, xmax, ymin, ymax, zmin, zmax)

        raise_error (bool, optional):
            If True, raises ValueError when points fall outside extent.
            If False, returns only valid points.


    Raises:
        ValueError:
            If points lie outside the extent and raise_error=True.
    """

    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    xmin, xmax, ymin, ymax, zmin, zmax = extent

    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )

    outside_points = points[~mask]

    if outside_points.size > 0:
        msg = (
            f"{outside_points.shape[0]} triangulation points lie outside the extent.\n"
            f"Extent: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}, zmin={zmin}, zmax={zmax}\n"
            f"First few invalid points:\n{outside_points[:5]}"
        )

        if raise_error:
            raise ValueError(msg)
        else:
            print("Warning:", msg)
            return points[mask]




def mesh_generator(ov: List[Tuple[int, int]], tagsss: List[int], extent: List[float], wells: List[Tuple[float, ...]], well_tags: Optional[List[int]],
    source_tag: Optional[List[int]], shaft_tags: Optional[List[int]], shaft_to_child_fragments: Dict[int, List[int]],  tri_surface_tags, tri_surface_to_child_fragments,
    grid_litho: pd.DataFrame, mesh_size: float = 30.0, curve_mesh_size: float = 5.0):

  """
    Generate an unstructured 3D tetrahedral mesh using Gmsh and export it
    in a meshio-compatible format.

    The function meshes 3D volumes using Gmsh, preserves embedded points, lines (wells), and surfaces (faults, triangulations),
    separates shaft-related tetrahedra from regular geological volumes, assigns lithology to tetrahedral blocks using nearest-neighbor
        classification against a geological grid, merges tetrahedra with identical lithology into larger blocks, and returns a cleaned mesh ready for export or simulation.

    Args:
        ov (List[Tuple[int, int]]): Gmsh entities defined as (dimension, tag), typically volume entities resulting from prior fragmentation.
        tagsss (List[int]): Physical group tags assigned to the main geological surfaces (faults or layers) + extra planes after fragmentation.
        wells (List[Tuple[float, ...]]): Well trajectories defined as flat coordinate tuples: (x1, y1, z1, x2, y2, z2, ...).
        well_tags (List[int] or None): Gmsh physical group tags associated with wells (dimension = 1).
        source_tag (List[int] or None): Physical group tags of embedded source points (dimension = 0).
        shaft_tags (List[int] or None): Physical group tags identifying shaft volumes (dimension = 3).
        shaft_to_child_fragments (Dict[int, List[int]]): Mapping from each shaft volume tag to the list of child volume tags
            produced by Boolean fragmentation. Used to isolate shaft tetrahedra from regular geological volume tetrahedra.
        tri_surface_tags (List[int] or None): Physical group tags assigned to triangulated surfaces (generated from external triangulation input). These are preserved
            as 2D triangle elements.
        tri_surface_to_child_fragments (Dict[int, List[int]]): Mapping from each parent triangulated surface tag to its child
            surface tags created during fragmentation. Used to correctly collect all triangle elements belonging to triangulated surfaces.
        grid_litho (pandas.DataFrame): Geological reference grid with four columns: x coordinate, y coordinate, z coordinate and
            lithology number. Used to assign lithology to tetrahedral elements via nearest-neighbor classification.
        mesh_size (float, optional): Global target mesh size for Gmsh (default: 20.0).
        curve_mesh_size (float, optional): Mesh refinement factor based on curvature (Mesh.MeshSizeFromCurvature, default: 5.0).

    Returns:
        nodes (numpy.ndarray): Array of mesh node coordinates with shape (n_nodes, 3).

        new_cells (List[meshio.CellBlock]): Mesh elements grouped by type. Tetrahedral elements are grouped
            by lithology, while: geological surfaces, triangulated surfaces, wells (lines), and source points,
            are preserved as separate element blocks.
  """


  # Extract volumes tags
  volumes: List[int] = [tag for dim, tag in ov if dim == 3]
  x_min, x_max, y_min, y_max, z_min, z_max = extent

  # Add physical groups for volumes only
  for i, tag in enumerate(volumes):
      gmsh.model.addPhysicalGroup(3, [tag], i + 1)
      gmsh.model.setPhysicalName(3, i + 1, f"Volume {i + 1}")
      print((3, i + 1, f"Volume {i + 1}"), 'physical')
  gmsh.model.occ.synchronize()
  gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)

  gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
  #gmsh.model.mesh.removeDuplicateNodes()

  # Generate 3D mesh
  if source_tag:
      for volume_tag in volumes:
          for s_tag in source_tag:
              gmsh.model.mesh.embed(0, [s_tag], 3, volume_tag)

  # specify a global mesh size and mesh the partitioned model:
  gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)


  gmsh.option.setNumber("General.Verbosity", 4)
  gmsh.model.occ.synchronize()
  #gmsh.option.setNumber("Geometry.Tolerance", 1e-4)  # default is 1e-6
  gmsh.model.mesh.generate(3)
  #gmsh.write("mesh.msh")       # save to file

  gmsh.model.occ.synchronize()
# NODES
  node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
  nodes = node_coords_flat.reshape(-1, 3)

  tag2idx: Dict[int, int] = {tag: i for i, tag in enumerate(node_tags)}

# ELEMENT TYPE MAP
  gmsh_to_meshio: Dict[int, Tuple[str, int]] = {
    15: ("vertex", 1),     # points
    1:  ("line", 2),       # linear lines
    2:  ("triangle", 3),
    4:  ("tetra", 4),
  }

  cells: List[meshio.CellBlock] = []
  cell_sets: Dict[str, List[int]] = {}

  import time

  t0 = time.perf_counter()
# =================================================
# FAST EXTRACTION (all dims, keeps cell_sets)
# =================================================
  def extract_all_fast():
    # Precompute: entity → physical name
    entity_to_phys = {}

    for dim, phys_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_tag) or f"phys_{dim}_{phys_tag}"
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
        for ent in entities:
            entity_to_phys[(dim, ent)] = name

    # Loop once over dimensions
    for dim in [3, 2, 1, 0]:
        entities = gmsh.model.getEntities(dim)

        for dim_e, ent in entities:
            key = (dim_e, ent)

            if key not in entity_to_phys:
                continue

            name = entity_to_phys[key]
            block_ids = []

            elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim_e, ent)

            for etype, enodes in zip(elem_types, elem_node_tags):
                if etype not in gmsh_to_meshio:
                    continue

                cell_type, n_nodes = gmsh_to_meshio[etype]

                conn = enodes.reshape(-1, n_nodes)

                # 🔥 FAST mapping instead of np.vectorize
                conn = np.array([tag2idx[t] for t in conn.ravel()]).reshape(conn.shape)

                cells.append(meshio.CellBlock(cell_type, conn))
                block_ids.append(len(cells) - 1)

            if block_ids:
                if name not in cell_sets:
                    cell_sets[name] = []
                cell_sets[name].extend(block_ids)
  extract_all_fast()
  t1 = time.perf_counter()


  t2=time.perf_counter()
# SPECIAL HANDLING FOR LINES (WELLS)
  line_blocks = defaultdict(list)

  for _, phys_tag in gmsh.model.getPhysicalGroups(1):
    name = gmsh.model.getPhysicalName(1, phys_tag) or f"phys_1_{phys_tag}"
    entity_tags = gmsh.model.getEntitiesForPhysicalGroup(1, phys_tag)

    for ent in entity_tags:
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(1, ent)

        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype not in gmsh_to_meshio:
                continue

            cell_type, n_nodes = gmsh_to_meshio[etype]
            conn = enodes.reshape(-1, n_nodes)
            conn = np.vectorize(tag2idx.get)(conn)

            line_blocks[name].append(conn)

# Merge ALL line segments per physical group
  for name, conns in line_blocks.items():
    merged = np.vstack(conns)
    cell_type = "line" if merged.shape[1] == 2 else "line3"

    cells.append(meshio.CellBlock(cell_type, merged))
    cell_sets[name] = [len(cells) - 1]
  t3=time.perf_counter()

  t4=time.perf_counter()

# -----------------------------
# FILTER UNUSED NODES (FAST VERSION)
# -----------------------------
# Step 1: Find all used node indices
  used = np.unique(np.concatenate([cb.data.ravel() for cb in cells]))

# Step 2: Build a direct lookup array
  max_idx = used[-1]  # largest node tag
  lookup = np.full(max_idx + 1, -1, dtype=int)
  lookup[used] = np.arange(len(used))

# Step 3: Subset node coordinates
  nodes = nodes[used]

# Step 4: Remap cell connectivity using fast NumPy indexing
  cells = [
    meshio.CellBlock(cb.type, lookup[cb.data])
    for cb in cells
]

# Step 5: Build mesh
  mesh_model = meshio.Mesh(
    points=nodes,
    cells=cells,
    cell_sets=cell_sets
)

  t5=time.perf_counter()

  t6=time.perf_counter()

# Inspect mesh blocks
  for block in mesh_model.cells:
    print(f"Cell type: {block.type}, Number of cells: {len(block.data)}")
  cells: List[meshio.CellBlock] = mesh_model.cells
  nodes: NDArray[np.float64] = mesh_model.points
  tetra_blocks_all: List[meshio.CellBlock]  = [block for block in cells if block.type == "tetra"]
  print(f"Number of tetrahedral blocks: {len(tetra_blocks_all)}")
# -------------------------------------------------
# SHAFTS (separate tetra blocks like wells)
# -------------------------------------------------
  print("Extracting shaft and regular tetra blocks")
  shaft_blocks_dict: Dict[int, List[meshio.CellBlock]] = {}
  regular_blocks: List[meshio.CellBlock] = []
# initialize shaft storage
  shaft_tag_set = set(shaft_tags)
  for tag in shaft_tag_set:
    shaft_blocks_dict[tag] = []
  t7=time.perf_counter()

  t8=time.perf_counter()
# -------------------------------------------------
# SINGLE LOOP over cell_sets
# -------------------------------------------------
  for name, block_ids in cell_sets.items():
    for bid in block_ids:
        block = cells[bid]
        # only tetrahedral elements
        if block.type not in {"tetra", "tetra10"}:
            continue
        assigned_to_shaft = False
        # -----------------------------
        # CHECK SHAFT TAGS (3500+i)
        # -----------------------------
        for shaft_tag in shaft_tag_set:
            if str(shaft_tag) in name or name.endswith(str(shaft_tag)):
                shaft_blocks_dict[shaft_tag].append(block)
                assigned_to_shaft = True
                break
        # -----------------------------
        # REGULAR VOLUMES
        # -----------------------------
        if not assigned_to_shaft:
            regular_blocks.append(block)
# -------------------------------------------------
# MERGE SHAFT BLOCKS (per shaft)
# -------------------------------------------------
  merged_tetra_blocks_sorted: List[meshio.CellBlock] = []
  for shaft_tag, shaft_blocks in shaft_blocks_dict.items():
    if not shaft_blocks:
        continue
    # collect connectivity
    all_data = [b.data for b in shaft_blocks if len(b.data) > 0]
    if not all_data:
        continue
    merged_data = np.vstack(all_data)
    merged_tetra_blocks_sorted.append(
        meshio.CellBlock("tetra", merged_data)
    )
    print(f"  Shaft {shaft_tag}: merged {len(merged_data)} tetra elements")
# -------------------------------------------------
# DEBUG PRINT
# -------------------------------------------------
  for shaft_tag, blocks in shaft_blocks_dict.items():
    if blocks:
        print(f"  Shaft {shaft_tag}: {len(blocks)} tetra blocks")
    else:
        print(f"  Shaft {shaft_tag}: not found")
  print(f"  Regular volume blocks: {len(regular_blocks)}")
  t9=time.perf_counter()

  t10=time.perf_counter()
# -------------------------------------------------
# Lithology assignment (ONLY regular blocks)
# -------------------------------------------------
  grid_coords: NDArray[np.float64] = grid_litho.iloc[:, :3].to_numpy()
  grid_litho_values: NDArray[np.int64] = grid_litho.iloc[:, 3].to_numpy()
  tree = cKDTree(grid_coords)
  threshold_ratio: float = 0.70
  lithology_numbers: List[int] = []
# -------------------------------------------------
# Assign lithology per regular block
# -------------------------------------------------
  for block in regular_blocks:
    node_ids: NDArray[np.int64] = np.unique(block.data)
    node_coords: NDArray[np.float64] = nodes[node_ids]
    _, nearest_idx = tree.query(node_coords)
    node_litho: NDArray[np.int64] = grid_litho_values[nearest_idx]
    unique_vals, counts = np.unique(node_litho, return_counts=True)
    max_idx: int = np.argmax(counts)
    if counts[max_idx] / len(node_ids) >= threshold_ratio:
        lithology_numbers.append(int(unique_vals[max_idx]))
    else:
        centroid: NDArray[np.float64] = node_coords.mean(axis=0)
        distances = np.linalg.norm(node_coords - centroid, axis=1)
        nearest_sample_idx = np.argmin(distances)
        lithology_numbers.append(int(node_litho[nearest_sample_idx]))
  t11=time.perf_counter()
  t12=time.perf_counter()
# -------------------------------------------------
# Merge geological blocks by lithology
# -------------------------------------------------
  litho_to_blocks: Dict[int, List[meshio.CellBlock]] = defaultdict(list)
  for lith, block in zip(lithology_numbers, regular_blocks):
    litho_to_blocks[lith].append(block)
  merged_regular_blocks: List[meshio.CellBlock] = []
  for lith, blocks in litho_to_blocks.items():
    merged_data: NDArray[np.int64] = np.vstack(
        [b.data for b in blocks if len(b.data) > 0]
    )
    merged_regular_blocks.append(meshio.CellBlock("tetra", merged_data))
# -------------------------------------------------
# Sort regular blocks by depth
# -------------------------------------------------
# Compute block centroids for sorting
  block_centroids: List[Tuple[int, float, float]] = []
  for i, block in enumerate(merged_regular_blocks):
    node_ids: NDArray[np.int64] = np.unique(block.data)
    centroid = nodes[node_ids].mean(axis=0)  # (x, y, z)
    block_centroids.append((i, centroid[2], centroid[0]))  # (index, z, x)

# Sort by z ascending, then x descending
  sorted_indices: List[int] = [
    i for i, _, _ in sorted(block_centroids, key=lambda t: (t[1], -t[2]))
]

# Reorder blocks
  sorted_regular_blocks: List[meshio.CellBlock] = [
    merged_regular_blocks[i] for i in sorted_indices
]
  t13=time.perf_counter()

  t14=time.perf_counter()
#-------------------------------------------------
# FINAL MERGE: regular + shafts
# -------------------------------------------------
  merged_tetra_blocks_sorted = []
# 1. regular lithology blocks (sorted)
  merged_tetra_blocks_sorted.extend(sorted_regular_blocks)
# 2. shaft blocks (already merged earlier!)
  for shaft_tag, shaft_blocks in shaft_blocks_dict.items():
    if not shaft_blocks:
        continue
    all_data = [b.data for b in shaft_blocks if len(b.data) > 0]
    if not all_data:
        continue
    merged_data = np.vstack(all_data)
    merged_tetra_blocks_sorted.append(
        meshio.CellBlock("tetra", merged_data)
    )
    print(f"  Shaft {shaft_tag}: merged {len(merged_data)} tetra elements")
# -------------------------------------------------
# Rebuild final cell list
# -------------------------------------------------
  cells_n: List[meshio.CellBlock] = [
    block for block in cells if block.type not in {"tetra"}
  ]
  cells_n.extend(merged_tetra_blocks_sorted)

  new_cells: List[meshio.CellBlock] = [
    block for block in cells_n
    if block.type not in {"line", "triangle", "vertex"}
]
  print("Extracting wells, sources, fault surfaces, and triangulated surfaces (single loop)")
  found_any = False
  t15=time.perf_counter()
  t16=time.perf_counter()

# -----------------------------
# Detect fault tags dynamically from cell names
# -----------------------------
  fault_tags_set = set()
  for name in cell_sets:
    numbers_in_name = [int(m) for m in re.findall(r'\d+', name)]
    for n in numbers_in_name:
        if n >= 1000 and n % 1000 == 0:
            fault_tags_set.add(n)

  fault_tags = sorted(fault_tags_set)
  fault_tri_blocks = {tag: [] for tag in fault_tags}
  print(f"Detected fault tags: {fault_tags}")

# -----------------------------
# Triangulated surfaces
  tri_surface_blocks = []

# Wells (1060–1070)
  well_blocks = {tag: [] for tag in well_tags if 1060 <= tag <= 1070}

# Sources (points)
  vertex_blocks: List[meshio.CellBlock] = []

# -----------------------------
# Single loop over cell_sets
# -----------------------------
  for name, block_ids in cell_sets.items():
    # Extract numbers from name for precise matching
    numbers_in_name = [int(n) for n in re.findall(r'\d+', name)]

    for bid in block_ids:
        block = cells[bid]

        # -----------------------------
        # TRIANGULATED SURFACE
        # -----------------------------
        if tri_surface_tags and "TRIANGULATED_SURFACE" in name:
            if block.type == "triangle":
                tri_surface_blocks.append(block.data)

        # -----------------------------
        # FAULT SURFACES (exact tag match)
        # -----------------------------
        for tag in fault_tags:
            if tag in numbers_in_name and block.type == "triangle":
                fault_tri_blocks[tag].append(block)

        # -----------------------------
        # WELLS (exact tag match)
        # -----------------------------
        for tag in well_blocks:
            if tag in numbers_in_name and block.type in {"line"}:
                well_blocks[tag].append(block)

        # -----------------------------
        # SOURCES (points)
        # -----------------------------
        if block.type == "vertex":
            vertex_blocks.append(block)

  t17=time.perf_counter()
  t18=time.perf_counter()

# -----------------------------
# ADD FAULT SURFACES (merge by tag, filter outside)
# -----------------------------
  for tag, blocks in fault_tri_blocks.items():
    if not blocks:
        print(f"  Fault tag {tag} not found")
        continue

    # collect triangles that are inside the model extent
    kept_triangles = []

    for b in blocks:
        if len(b.data) == 0:
            continue

        # Get node coordinates for each triangle
        tri_coords = nodes[b.data]
        # Compute centroid of each triangle
        centroids = tri_coords.mean(axis=1) if tri_coords.ndim == 3 else tri_coords

        # Check if centroid is inside the bounding box
        inside_mask = (
            (centroids[:, 0] >= x_min) & (centroids[:, 0] <= x_max) &
            (centroids[:, 1] >= y_min) & (centroids[:, 1] <= y_max) &
            (centroids[:, 2] >= z_min) & (centroids[:, 2] <= z_max)
        )

        # Keep only triangles inside
        kept_triangles.append(b.data[inside_mask])

    # Merge all kept triangles
    if kept_triangles:
        merged_data = np.vstack(kept_triangles)
        new_cells.append(meshio.CellBlock("triangle", merged_data))
        print(f"  Fault tag {tag}: merged {len(merged_data)} triangles from {len(blocks)} blocks after filtering")
        found_any = True
    else:
        print(f"  Fault tag {tag}: no triangles inside bounding box after filtering")

# -----------------------------
# ADD TRIANGULATED SURFACES (merge into single block)
# -----------------------------
  if tri_surface_blocks:
    merged_data = np.vstack(tri_surface_blocks)
    new_cells.append(meshio.CellBlock("triangle", merged_data))
    print(f"  Triangulated surface: {len(merged_data)} triangles")
    found_any = True
  t19=time.perf_counter()
  t20=time.perf_counter()

# -----------------------------
# ADD WELLS
# -----------------------------
  for tag, blocks in well_blocks.items():
    if blocks:
        new_cells.extend(blocks)
        print(f"  Well tag {tag} added ({len(blocks)} line blocks)")
        found_any = True
    else:
        print(f"  Well tag {tag} not found")
# -----------------------------
# ADD SOURCES
# -----------------------------
  if vertex_blocks:
    new_cells.extend(vertex_blocks)
    print(f"  Added {len(vertex_blocks)} source point blocks")
    found_any = True
# -----------------------------
# FINAL CHECK
# -----------------------------
  if not found_any:
    print("  No surfaces, wells, or sources found")
  t21=time.perf_counter()


  print(f"Extraction: {t1 - t0:.4f} s")
  print(f"Handeling lines after extraction:    {t3 - t2:.4f} s")
  print(f"Filter unused nodes and create meshio:      {t5 - t4:.4f} s")
  print(f"Shaft sepration: {t7 - t6:.4f} s")
  print(f"regular grid sepration:    {t9 - t8:.4f} s")
  print(f"Lithology assign:      {t11 - t10:.4f} s")
  print(f"Merge Litho: {t13 - t12:.4f} s")
  print(f"Final merge +rebluid:    {t15 - t14:.4f} s")
  print(f"Fiault + wells ...:      {t17 - t16:.4f} s")
  print(f"Add faults:      {t19 - t18:.4f} s")
  print(f"Add wells:      {t21 - t20:.4f} s")

#---------------
  if new_cells:
    return nodes, new_cells
  else:
    return nodes, cells


WellData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='well_list', color='aqua', identifier='mesh::WellListData', controlled='Table|x3')]
SourcesData = typing.Annotated[List[Tuple[float, float, float]], AnnotatedScriptType(name='sources', color='aqua', identifier='mesh::SourcesData', controlled='Table|3|X|Y|Z')]
ShaftData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='shaft_list', color='aqua', identifier='mesh::ShaftListData', controlled='Table|x7')]

PlaneData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='plane_list', color='aqua', identifier='mesh::PlaneListData', controlled='Table|x12')]
EllipseData = typing.Annotated[List[Dict[str, typing.Any]], AnnotatedScriptType( name='ellipse_list', color='aqua', identifier='mesh::EllipseListData' )]
TriangulationData = typing.Annotated[List[Tuple[float, float, float]], AnnotatedScriptType( name='triangulation', color='aqua',
        identifier='mesh::TriangulationData',controlled='Table|3|X|Y|Z')]

# the file must end with "wells.csv", e.g., "example_wells.csv", etc.
WellCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::well_csv', controlled='RemoteFile|endswith=wells.csv')]

@wbgeo_component(description='Loads a well from a wells.CSV file',
                 title='Load Well',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='wbgeo::meshing_load_well_from_csv',  # a unique identifier
                 return_name='Wells',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature

def load_wells_from_csv(well_file: WellCSVDataType, key_hierarchical: bool = False) ->  WellData:
    """
    Load well trajectories from a CSV file.

    Returns:
        If key_hierarchical=False:
            List[Tuple] → [(x1,y1,z1,x2,y2,z2,...), ...]
        If key_hierarchical=True:
            Dict[int, List[Tuple]] → {1: [...], 2: [...]}
    """

    named_well_data = {}

    with open(well_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = [s.strip() for s in line.split(",")]

            if key_hierarchical:
                if len(parts) != 5:
                    raise ValueError("Invalid well line (expected 5 columns: id,x,y,z,key)")
                well_id, x, y, z, key = parts
                key = int(key)
                well_key_id = (well_id, key)
            else:
                if len(parts) != 4:
                    if len(parts) != 4:
                        raise ValueError("Invalid well line (expected 4 columns: id,x,y,z)")
                well_id, x, y, z = parts
                well_key_id = well_id
                key = None

            try:
                x = float(x)
                y = float(y)
                z = float(z)
            except ValueError:
                raise ValueError("Non-numeric value in well definition")

            if well_key_id not in named_well_data:
                named_well_data[well_key_id] = []

            named_well_data[well_key_id].append((x, y, z))

    # Massages
    incorrect = [str(wid) if not isinstance(wid, str) else wid
                 for wid, pts in named_well_data.items() if len(pts) < 2]

    if incorrect:
        raise ValueError(f"Some well(s) {incorrect} are missing their second point")

    flattened_wells = {
        wid: tuple(c for pt in pts for c in pt)
        for wid, pts in named_well_data.items()
    }

    if not key_hierarchical:
        return list(flattened_wells.values())

    wells_by_key = {1: [], 2: []}
    for (wid, k), coords in flattened_wells.items():
        wells_by_key[k].append(coords)

    return wells_by_key


# the file must end with "sources.csv", e.g., "example_sources.csv", etc.
SourceCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::source_csv', controlled='RemoteFile|endswith=sources.csv')]

@wbgeo_component(description='Loads a source from a sources.CSV file',
                 title='Load Source',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='wbgeo::meshing_load_source_from_csv',  # a unique identifier
                 return_name='Sources',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def load_sources_from_csv(source_file: SourceCSVDataType) -> SourcesData:
  """
    Load point sources from a CSV file.
    The CSV file must contain exactly one point per source and follow the format: id, x, y, z

    Notes:
        - Lines starting with `#` are ignored.
        - Multiple rows with the same `id` are NOT allowed.
        - Each source must be defined by exactly one 3D point.
        - The output format is a list of flat coordinate tuples:
          [(x, y, z), ...]

    Args:
        source_file (str): Path to the CSV file containing source definitions.

    Returns:
        SourcesData: A list of sources, where each source is represented as a flat uple of coordinates (x, y, z).

    Raises:
        ValueError:
            - If a row does not contain exactly four columns.
            - If x, y, or z values are missing.
            - If any coordinate value is non-numeric.
            - If a source ID appears more than once in the file.
  """
  # format of the csv is: id, x,y,z\n
  named_source_data: Mapping[str, List[Tuple[float]]] = {}
  # load csv file
  with open(source_file, 'r') as f:
    for line in f.readlines():
      if line.startswith('#'): continue
      parts = [s.strip() for s in line.split(",")]
      # CHECK NUMBER OF COLUMNS FIRST
      if len(parts) != 4:
          raise ValueError(
                f"Invalid source line (expected 4 columns: id,x,y,z): {line}"
                )
      source_id, source_x, source_y, source_z = parts
      #check empty values (for instance x, y or z is missing)
      if not all([source_x, source_y, source_z]):
            raise ValueError(
                f"(missing x, y, or z value): {line.strip()}"
            )
        # check for non numeric values
      try:
          values = [float(source_x), float(source_y), float(source_z)]
      except ValueError:
            raise ValueError(f"Non-numeric value in source definition: {line.strip()}")
      if source_id not in named_source_data:
        named_source_data[source_id] = []
      named_source_data[source_id].append([float(source_x), float(source_y), float(source_z)])
  # ensure that we have only one point per source
  if any(True for source in named_source_data.values() if len(source) > 1):
    incorrect_sources = [source_id for source_id, source_data in named_source_data.items() if
                       len(source_data) > 1]
    raise ValueError(f"Some source(s) {incorrect_sources} has more than one coordinates")
  # format right now is {key: [(x,y,z)]} -> map it to [x1, y1, z1] for each source
  return [tuple([coordinate for source_group in source_data for coordinate in source_group]) for source_data
          in named_source_data.values()]



EllipseCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::ellipse_csv', controlled='RemoteFile|endswith=ellipses.csv')]

@wbgeo_component(
    description='Loads ellipses from an ellipses.CSV file',
    title='Load Ellipse',
    color='#cc9999',
    border_color='#000000',
    group='Meshing',
    identifier='wbgeo::meshing_load_ellipse_from_csv',
    return_name='Ellipses',
    is_object_type=True,
)
def load_ellipses_from_csv(ellipse_file: EllipseCSVDataType) -> EllipseData:
    """
    Load ellipses from a CSV file and return them as a list of dictionaries.

    The CSV file should contain one ellipse per row, with optional ID columns.
    Each row must contain at least the required parameters for defining an ellipse.

    Format:
        - Optional ID column (first column) is automatically detected and ignored if present.
        - Required fields: cx, cy, cz, r1, r2
            - cx, cy, cz: coordinates of the ellipse center
            - r1: semi-major axis length
            - r2: semi-minor axis length
        - Optional fields (if present):
            - angle1, angle2: start and end angles in radians for a partial ellipse (default: 0.0, 2*pi)
            - zAxis: normal vector of the ellipse plane (default: [0, 0, 1])
            - xAxis: x-axis direction of the ellipse plane (optional; auto-generated if not provided)

    Example CSV rows:
        1, 100, 200, 50, 30           # ellipse with center (100,200,50) and radii 30,50
        2, 150, 250, 60, 40, 0, 3.14  # ellipse with angles
        3, 200, 300, 70, 50, 0, 6.28, 0,0,1  # ellipse with zAxis
        4, 250, 350, 80, 60, 0, 6.28, 0,0,1, 1,0,0  # ellipse with zAxis and xAxis

    Args:
        ellipse_file (str): Path to the CSV file containing ellipse definitions.

    Returns:
        EllipseData: A list of ellipses, each represented as a dictionary with keys:
            - 'center': tuple[float, float, float] - center coordinates
            - 'radii': tuple[float, float] - semi-major and semi-minor axes
            - 'angle1': float (optional) - start angle in radians
            - 'angle2': float (optional) - end angle in radians
            - 'zAxis': list[float] (optional) - normal vector
            - 'xAxis': list[float] (optional) - x-axis vector

    Raises:
        ValueError: If a row has an invalid number of columns, missing numeric values,
                    or contains non-numeric data in required fields.
    """

    ellipses = []

    with open(ellipse_file, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            # Detect optional ID column
            if len(parts) in (6, 8, 11, 14):
                parts = parts[1:]

            if len(parts) < 5:
                raise ValueError(f"Line {line_number}: too few columns ({len(parts)}).")

            # --- Required fields ---
            try:
                cx = float(parts[0])
                cy = float(parts[1])
                cz = float(parts[2])
                r1 = float(parts[3])
                r2 = float(parts[4])
            except ValueError:
                raise ValueError(f"Line {line_number}: invalid numeric value in required fields")

            ellipse = dict(center=(cx, cy, cz), radii=(r1, r2))
            index = 5

            # --- Optional angles ---
            if len(parts) >= index + 2:
                angle1 = float(parts[index]) if parts[index] else 0.0
                angle2 = float(parts[index+1]) if parts[index+1] else 2*np.pi
                ellipse["angle1"] = angle1
                ellipse["angle2"] = angle2
                index += 2
            else:
                ellipse["angle1"] = 0.0
                ellipse["angle2"] = 2*np.pi

            # --- Optional zAxis ---
            if len(parts) >= index + 3:
                try:
                    zAxis = [float(parts[index]), float(parts[index+1]), float(parts[index+2])]
                except ValueError:
                    raise ValueError(f"Line {line_number}: zAxis must contain 3 numeric values")
                if len(zAxis) != 3 or not all(np.isfinite(zAxis)):
                    raise ValueError(f"Line {line_number}: zAxis must be 3 finite numbers")
                ellipse["zAxis"] = zAxis
                index += 3
            else:
                ellipse["zAxis"] = [0, 0, 1]

            # --- Optional xAxis ---
            if len(parts) >= index + 3:
                try:
                    xAxis = [float(parts[index]), float(parts[index+1]), float(parts[index+2])]
                except ValueError:
                    raise ValueError(f"Line {line_number}: xAxis must contain 3 numeric values")
                if len(xAxis) != 3 or not all(np.isfinite(xAxis)):
                    raise ValueError(f"Line {line_number}: xAxis must be 3 finite numbers")
                ellipse["xAxis"] = xAxis

            ellipses.append(ellipse)

    return ellipses




# the file must end with "shaftss.csv", e.g., "example_shaftss.csv", etc.
ShaftCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::shaft_csv', controlled='RemoteFile|endswith=shafts.csv')]

@wbgeo_component(description='Loads a shatf from a shafts.CSV file',
                 title='Load Shaft',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='wbgeo::meshing_load_shaft_from_csv',  # a unique identifier
                 return_name='Shafts',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def load_shafts_from_csv(shaft_file: ShaftCSVDataType) -> ShaftData:
    """
    Load shaft definitions from a CSV file.
    Each shaft is defined by a single row with the following format: id, cx, cy, cz, ax, ay, az, radius
    where:
        - (cx, cy, cz) is the shaft center point
        - (ax, ay, az) is the shaft axis direction vector
        - radius is the shaft radius

    Notes:
        - Lines starting with `#` are ignored.
        - Each shaft ID must appear exactly once in the file.
        - All numeric values must be present and valid floating-point numbers.
        - The output format is a list of flat numeric tuples: [(cx, cy, cz, ax, ay, az, radius), ...]

    Args:
        shaft_file (str): Path to the CSV file containing shaft definitions.

    Returns:
        ShaftData: A list of shafts, where each shaft is represented as a flat tuple (cx, cy, cz, ax, ay, az, radius).

    Raises:
        ValueError:
            - If a row does not contain exactly eight columns.
            - If any required numeric value is missing.
            - If any value cannot be converted to float.
            - If a shaft ID is defined more than once.
    """
    named_shaft_data = {}

    with open(shaft_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            parts = [s.strip() for s in line.split(",")]

            if len(parts) != 8:
                raise ValueError(
                    f"Invalid shaft line (expected 8 columns): {line.strip()}"
                )

            shaft_id, cx, cy, cz, ax, ay, az, rad = parts

            # check missing values
            numeric_fields = {
                "cx": cx, "cy": cy, "cz": cz,
                "ax": ax, "ay": ay, "az": az,
                "radius": rad
            }

            missing = [k for k, v in numeric_fields.items() if v == ""]
            if missing:
                raise ValueError(
                    f"Missing value(s) {missing} in shaft definition"
                )

            # check numeric values
            try:
                values = [
                    float(cx), float(cy), float(cz),
                    float(ax), float(ay), float(az),
                    float(rad)
                ]
            except ValueError:
                raise ValueError(
                    f"Non-numeric value in shaft definition: {line.strip()}"
                )

            # DUPLICATE ID CHECK
            if shaft_id in named_shaft_data:
                raise ValueError(
                    f"Duplicate shaft definition for shaft ID '{shaft_id}'"
                )

            named_shaft_data[shaft_id] = values

    return [tuple(v) for v in named_shaft_data.values()]



# the file must end with "planes.csv", e.g., "example_planes.csv", etc.
PlaneCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::plane_csv', controlled='RemoteFile|endswith=planes.csv')]

@wbgeo_component(description='Loads a plane from a planes.CSV file',
                 title='Load Plane',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='wbgeo::meshing_load_plane_from_csv',  # a unique identifier
                 return_name='planes',  # the name for the returned-port
                 is_object_type=True,
                 )  # inputs are handled via the method signature
def load_planes_from_csv(plane_file: PlaneCSVDataType) -> PlaneData:
  """
    Load plane definitions from a CSV file.
    Each plane is defined by multiple rows with the same plane ID. Each row provides a 3D point belonging to that plane: id, x, y, z

    Notes:
        - Lines starting with `#` are ignored.
        - Multiple rows with the same `id` belong to the same plane.
        - Each plane must be defined by at least four points.
        - All coordinate values must be valid floating-point numbers.
        - The output format is a list of flat coordinate tuples: [(x1, y1, z1, x2, y2, z2, ..., xN, yN, zN), ...]

    Args:
        plane_file (str): Path to the CSV file containing plane point definitions.

    Returns:
        PlaneData: A list of planes, where each plane is represented as a flat tuple containing the concatenated coordinates of its defining points.

    Raises:
        ValueError:
            - If a row does not contain exactly four columns (id, x, y, z).
            - If any coordinate value is missing.
            - If any coordinate value cannot be converted to float.
            - If a plane is defined with fewer than four points.
  """
  # format of the csv is: id, x,y,z\n
  named_plane_data: Mapping[str, List[Tuple[float]]] = {}
  # load csv file
  with open(plane_file, 'r') as f:
    for line in f.readlines():
      if line.startswith('#'): continue
      parts = [s.strip() for s in line.split(",")]
      # CHECK NUMBER OF COLUMNS FIRST
      if len(parts) != 4:
          print(len(parts), parts, 'partsss')
          raise ValueError(
                f"Invalid plane line (expected 4 columns: id,x,y,z): {line}"
                )
      plane_id, plane_x, plane_y, plane_z = parts
      #check empty values (for instance x, y or z is missing)
      if not all([plane_x, plane_y, plane_z]):
            raise ValueError(
                f"(missing x, y, or z value): {line.strip()}"
            )
      # check for non numeric values
      try:
          values = [float(plane_x), float(plane_y), float(plane_z)]
      except ValueError:
            raise ValueError(f"Non-numeric value in plane definition: {line.strip()}")
      if plane_id not in named_plane_data:
        named_plane_data[plane_id] = []
      named_plane_data[plane_id].append([float(plane_x), float(plane_y), float(plane_z)])
  # ensure that we have at least 2 points per plane
  if any(True for plane in named_plane_data.values() if len(plane) < 4):
    incorrect_planes = [plane_id for plane_id, plane_data in named_plane_data.items() if
                       len(plane_data) < 4]
    raise ValueError(f"Some plane(s) {incorrect_planes} are missing required number of points (minimum 4)")
  # format right now is {key: [(x,y,z)]} -> map it to [x1, y1, z1, ..., xi, yi, zi] for each plane
  return [tuple([coordinate for plane_group in plane_data for coordinate in plane_group]) for plane_data
          in named_plane_data.values()]


TriangulationsPlanesData = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::triangulations_planes_csv',
        controlled='RemoteFile|endswith=.csv')]

@wbgeo_component(
    description='Load planes coordinates from a CSV to do triangulations',
    title='Load Triangulations Planes',
    color='#cc9999',
    border_color='#000000',
    group='Meshing',
    identifier='wbgeo::meshing_load_triangulations_planes_from_csv',
    return_name='triangulations_planes',
    is_object_type=True,
)
def load_triangulations_planes_from_csv(csv_file: TriangulationsPlanesData) -> TriangulationData:
    """
    Load plane coordinates from a CSV file for triangulations.

    The CSV must have columns: x, y, z. Lines starting with '#' are ignored.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        np.ndarray: Array of 3D points as 'triangulations_planes' (shape Nx3).

    Raises:
        ValueError: If a line has missing or non-numeric values.
    """
    points = []

    with open(csv_file, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # ignore empty lines and comments
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Line {line_number}: expected 3 columns (x,y,z), got {len(parts)}: {line}")
            try:
                x, y, z = map(float, parts)
            except ValueError:
                raise ValueError(f"Line {line_number}: non-numeric value found: {line}")
            points.append([x, y, z])

    if not points:
        raise ValueError("No points loaded from the CSV file.")

    return np.array(points, dtype=np.float64)




# Register this function as a component
@wbgeo_component(description='Provides unstructured mesh',
                 title='Create Unstructured Mesh',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='create_unstructured_mesh_data',  # a unique identifier
                 return_name='Mesh',  # the name for the returned-port
                 )  # inputs are handled via the method signature

def create_unstructured_mesh_data(
    geomodel_result: StructuralModelResults,
    wells: Optional[WellData] = None,
    sources: Optional[SourcesData] = None,
    shafts: Optional[ShaftData] = None,
    extra_planes: Optional[PlaneData] = None,
    ellipses: Optional[EllipseData] = None,
    triangulations: Optional[TriangulationsPlanesData] = None,
    tolerance: float = 50,
    mesh_size: float = 30,
    curve_mesh_size: float = 5,
    DISTANCE_THRESHOLD: float = 50,
    PROJECTION_THRESHOLD: float = 60,
    EXTRUSION_FACTOR: float = 100,
    z_threshold: float = 10,
    extent: Optional[ExtentData] = None,
    smooth: float = 1e-5 ) -> MeshResults:
    """
    Generates an unstructured geological mesh using a geomodel and additional structures
    such as wells, sources, shafts, and extra planes. It performs surface cleaning,
    fragmentation, and meshing using GMSH and returns the final MeshData object.

    Args:
        input_data (InputData): Input data object containing surface points, orientations, mapping, faults, and extent.
        geomodel_result (object): Output object from the geomodel interpolation, e.g. from `universal_cokriging_interpolator`.
        wells (list of tuples): Each tuple contains coordinates defining the top (, middel) and bottom of a well (x1, y1, z1, x2, y2, z2).
        sources (list of tuples): Each tuple contains coordinates (x, y, z) of a point source.
        shafts (list of tuples): List of coordinates for the centers of mine shaft cylinders (x, y, z), direction vectors (dx, dy, dz)
                                 for the axes of mine shaft cylinders and radii for the mine shaft cylinders.
        extra_planes (list of tuples): Each tuple contains coordinates of 4 corners (12 values) defining an extra plane.
        ellipses (list of dicts): Each dictionary defines an ellipse with keys:
            - 'center': (x, y, z) coordinates of the ellipse center
            - 'radii': (r1, r2) the semi-major and semi-minor axes
            - 'angle1' (optional): start angle in radians (default=0.0)
            - 'angle2' (optional): end angle in radians (default=2*pi)
            - 'zAxis' (optional): z-axis vector of the ellipse plane (default=[0,0,1])
            - 'xAxis' (optional): x-axis vector of the ellipse plane (if not provided, auto-generated)
        triangulations (np.ndarray or list of tuples, optional): Array of 3D points [(x, y, z), ...] representing additional
                                   plane coordinates used for surface triangulation. If provided, these points are used to generate Delaunay
        tolerance (float): Distance threshold to identify boarder of mesh.
        mesh_size (int): Default mesh size for surface and volume meshing (default is 30).
        curve_mesh_size (int): Mesh size applied to curves (default is 5).
        DISTANCE_THRESHOLD (float): Maximum distance used to filter overlapping points between surfaces.
        PROJECTION_THRESHOLD (float): Distance threshold for projecting points when calculating extrusion.
        EXTRUSION_FACTOR (float): Factor that scales extrusion distance.
        z_threshold (float): Threshold for determining whether two surfaces on either side of a fault are close in elevation.
        extent (list): extent of mesh (min_x, max_x, min_y,max_y, min_z, max_z)
        smooth (float): smoothness factor for interpolation of surfaces
    Returns:
        MeshResults: An instance of the MeshResults class.
    """

    wells = wells or []

    sources = sources or []
    shafts = shafts or []
    extra_planes = extra_planes or []
    ellipses = ellipses or []
    if triangulations is None:
        triangulations = []
    mine_shafts: List[Dict[str, Tuple[float, ...]]] = []

    if len(shafts) != 0:
        for i, shaft in enumerate(shafts):
            cx, cy, cz, ax, ay, az, r = shaft

            mine_shafts.append({
                "center": (cx, cy, cz),
                "axis": (ax, ay, az),
                "radius": r
            })
    else:
      mine_shafts = []

    # extract extent of model
    extent_arr: NDArray[np.float64]
    if extent is None or len(extent) == 0:
        extent_arr = np.asarray(geomodel_result.structural_frame.grid.extent, dtype=float)
    else:
        extent_arr = np.asarray(extent, dtype=float)

    # check if engeeing objects are inside the extent
    validate_wells(wells, extent_arr)
    validate_sources(sources, extent_arr)
    validate_shafts(mine_shafts, extent_arr)
    validate_planes(extra_planes, extent_arr)
    validate_ellipses(ellipses, extent_arr)
    if len(triangulations) > 0:
        triangulations = np.asarray(triangulations, dtype=np.float64)
        validate_triangulation(triangulations, extent_arr)
    # Initialize GMSH
    gmsh.initialize()
    # Surface preparation
    cleaned_surfaces: List[NDArray[np.float64]]
    ref_surface_indices: Dict[int, int]
    grid_litho: pd.DataFrame
    cleaned_surfaces, ref_surface_indices , grid_litho = data_prepration(geomodel_result, DISTANCE_THRESHOLD = DISTANCE_THRESHOLD, PROJECTION_THRESHOLD = PROJECTION_THRESHOLD,
                                                                                EXTRUSION_FACTOR = EXTRUSION_FACTOR, z_threshold = z_threshold)

     # Surface interpolation
    interpolated_s: List[NDArray[np.float64]] = create_surface_grid(cleaned_surfaces, smooth=smooth)

    # Import surfaces
    surfaces_original: List[int]
    bounds: Tuple[float, float, float, float, float, float]
    surfaces_orginal, bounds = import_surfaces(interpolated_s, extent_arr, tolerance=tolerance)

    gmsh.model.occ.synchronize()


    surfaces=surfaces_orginal.copy()
    # fragmentation
    ov: List[Tuple[int, int]]
    tagssss: List[int]
    well_tags: List[int]
    shaft_tags: List[int]
    shaft_to_child_fragments: Dict[int, List[int]]
    source_tag: List[int]
    ov,ovv, tagssss, well_tags, shaft_tags,shaft_to_child_fragments,  tri_surface_tags, tri_surface_to_child_fragments, source_tag = fragment_surfaces(surfaces, bounds, ref_surface_indices,wells,
                                extra_planes, sources, mine_shafts,  ellipses=ellipses, triangulations = triangulations, mesh_size=mesh_size,curve_mesh_size=curve_mesh_size )

    # Mesh generation
    nodes: NDArray[np.float64]
    cells: List[meshio.CellBlock]
    nodes, cells = mesh_generator(ov, tagssss, extent_arr , wells, well_tags, source_tag, shaft_tags, shaft_to_child_fragments,  tri_surface_tags, tri_surface_to_child_fragments, grid_litho, mesh_size= mesh_size, curve_mesh_size=curve_mesh_size )
    print("\n[FINAL DEBUG] Line blocks in new_cells:")

    gmsh.finalize()
    # return a MeshData instance
    return MeshResults(elements=cells,
                       nodes=nodes,
                       )


@wbgeo_component(identifier='wbgeo::inspect_unstructure_mesh_3d',
                 title='Plot Unstructured Mesh in 3D',
                 description='...')
@wbgeo_inspector()
def inspect_unstructured_mesh_3d(
    structural_model_result: StructuralModelResults, mesh: MeshResults, _inspector: InspectorHelper):
  plot_mesh_3d(mesh, structural_model_result, "surface", True,)
