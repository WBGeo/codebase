import logging
import typing
import re
from pathlib import Path
import gmsh

logger = logging.getLogger(__name__)
import meshio
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from typing import Sequence, Tuple, List, Mapping, Optional, Dict, Set, Any
from numpy.typing import NDArray
from core.object_components import StructuralModelResults, ExtentData
from core.object_components import MeshResults, MeshType
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import create_surface_grid, import_surfaces, fragment_surfaces
from core.meshing_components.explicit.unstructured.create_clean_surface import data_preparation
from core.meshing_components.explicit.unstructured.domain_geometry import (
    domain_aabb, domain_local_coordinates, normalize_domain, point_inside_domain, points_inside_domain
)
from core.meshing_components.explicit.unstructured.refinement_mesh import (Refinement, LinearWellRefinement, FunctionWellRefinement,
                                                               LinearSourceRefinement,FunctionSourceRefinement,)
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from enum import Enum, StrEnum


MESH_ENGINEERING_COLOR = '#99b3cc'
MESH_ENGINEERING_GROUP = 'Engineering Objects'

class LithoMappingMode(StrEnum):
    NONE = "none"
    MANUAL = "manual"
    # Per-block lithology vote taken on each raw block's individual *cell
    # centroids* against grid_litho. Cell centroids are guaranteed-interior
    # points, so a fault-adjacent block's vote isn't confused by nodes
    # sitting exactly on a block boundary -- exactly where a fault plane
    # tends to be -- unlike AUTOMATIC_CORNERS. The default: generally more
    # accurate overall, especially near faults/unconformities, though the
    # accuracy gap between the two modes only really shows up when the
    # structural grid is coarse relative to the mesh -- neither mode fully
    # fixes that mismatch on its own (see _check_lithology_mapping_reliability,
    # which warns about it directly).
    AUTOMATIC_CENTERS = "automatic_centers"
    # Per-block lithology vote taken on each raw block's *node* coordinates
    # against grid_litho. Nodes sit on block boundaries -- exactly where a
    # fault plane tends to be -- so a fault-adjacent block's node vote can
    # be genuinely ambiguous regardless of how good the voting algorithm
    # is. Kept for comparison/backwards compatibility; AUTOMATIC_CENTERS is
    # recommended instead (see its own comment above for the comparison).
    AUTOMATIC_CORNERS = "automatic_corners"

LithoMappingModeType = typing.Annotated[
    LithoMappingMode,
    AnnotatedScriptType(
        name="LithoMappingMode",
        identifier="LithoMappingModeType",
        controlled="Select|automatic_centers|automatic_corners|manual|none"
    ),
]


def _check_lithology_mapping_reliability(
    litho_to_blocks: "Dict[int, List[meshio.CellBlock]]",
    vote_purities_by_lithology: "Dict[int, List[float]]",
    grid_litho_values: "NDArray[np.int64]",
    grid_coords: "NDArray[np.float64]",
    mesh_size: float,
    mode_name: str,
) -> None:
    """
    Log warnings for signs that mapping_litho="automatic_centers"/"automatic_corners"'s
    per-block lithology mapping may be unreliable for this run. Both modes
    are a geometric nearest-neighbor vote against the structural model's
    grid_litho, which degrades when that grid is coarse relative to the
    mesh -- each structural grid point then "owns" a large physical
    region, so many mesh cells/nodes snap to the same nearest point
    regardless of which side of a real boundary (e.g. a fault) they're
    actually on, degrading accuracy well below the near-perfect results
    seen at matched resolution. None of these checks can fix a resolution
    mismatch, only flag it.
    """
    voxel_sizes = []
    for axis in range(3):
        vals = np.unique(grid_coords[:, axis])
        if len(vals) > 1:
            voxel_sizes.append(float(np.diff(vals).min()))

    if voxel_sizes:
        voxel_size = float(np.mean(voxel_sizes))
        if voxel_size > 1.5 * mesh_size:
            logger.warning(
                "mapping_litho=%r: structural model grid voxel size (~%.1f) is "
                "more than 1.5x the mesh_size (%.1f). Lithology mapping accuracy "
                "degrades when the structural grid is coarse relative to the mesh, "
                "especially near faults/unconformities -- consider a finer "
                "structural model grid resolution if mapped lithology boundaries "
                "look unreliable.",
                mode_name, voxel_size, mesh_size,
            )

    n_expected = int(len(np.unique(grid_litho_values)))
    n_final = len(litho_to_blocks)
    if n_final > n_expected:
        logger.warning(
            "mapping_litho=%r: produced %d merged mesh blocks for %d distinct "
            "lithologies in the structural model -- some lithology's raw blocks "
            "likely got inconsistent votes and failed to merge into one block. "
            "Known failure mode near faults/unconformities and/or when the "
            "structural grid is coarse relative to the mesh.",
            mode_name, n_final, n_expected,
        )

    for lith, purities in sorted(vote_purities_by_lithology.items()):
        mean_purity = float(np.mean(purities))
        if mean_purity < 0.70:
            logger.warning(
                "mapping_litho=%r: lithology %d's block(s) averaged only %.0f%% "
                "vote agreement (fraction of a block's own nodes/cells whose "
                "nearest structural-grid point actually agreed with the lithology "
                "assigned to the whole block) -- this lithology's mesh region may "
                "be unreliable.",
                mode_name, lith, mean_purity * 100,
            )


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


def _check_extent_format(extent) -> None:
    """Validate either a legacy 6-value extent or an 8-corner oriented box."""
    normalize_domain(extent)


def point_inside_extent(point: Tuple[float, float, float], extent) -> bool:
    """Return True when a point is inside the legacy or 8-corner domain."""
    return point_inside_domain(point, extent)

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


def validate_sources(sources, extent) -> None:
    """Validate point sources for a normal or hierarchical data set."""
    _check_extent_format(extent)
    groups = sources.values() if isinstance(sources, dict) else (sources,)
    for group in groups:
        for i, source in enumerate(group):
            if len(source) != 3:
                raise ValueError(f"Source '{i + 1}' must contain exactly 3 coordinates (x, y, z).")
            x, y, z = map(float, source)
            if not np.all(np.isfinite([x, y, z])):
                raise ValueError(f"Source '{i + 1}' contains non-finite coordinates.")
            assert_point_inside_extent((x, y, z), extent, obj_type="Source", obj_id=i + 1)

def validate_shafts(shafts, extent) -> None:
    """Validate shaft centers/radii/axes for normal or hierarchical data."""
    _check_extent_format(extent)
    groups = shafts.values() if isinstance(shafts, dict) else (shafts,)
    for group in groups:
        for i, shaft in enumerate(group):
            # create_unstructured_mesh_data converts flat shaft tuples to dicts;
            # accepting both here makes validation safe in either call path.
            if isinstance(shaft, dict):
                center = np.asarray(shaft["center"], dtype=float)
                axis = np.asarray(shaft["axis"], dtype=float)
                radius = float(shaft["radius"])
            else:
                vals = np.asarray(shaft, dtype=float).reshape(-1)
                if vals.size != 7:
                    raise ValueError(f"Shaft '{i + 1}' must contain 7 values (cx,cy,cz,ax,ay,az,radius).")
                center, axis, radius = vals[:3], vals[3:6], float(vals[6])
            if center.shape != (3,) or axis.shape != (3,) or not np.all(np.isfinite(np.r_[center, axis, radius])):
                raise ValueError(f"Shaft '{i + 1}' contains invalid/non-finite values.")
            if np.linalg.norm(axis) == 0.0:
                raise ValueError(f"Shaft '{i + 1}' axis must be non-zero.")
            if radius <= 0.0:
                raise ValueError(f"Shaft '{i + 1}' radius must be positive.")
            assert_point_inside_extent(tuple(center), extent, obj_type="Shaft", obj_id=i + 1)

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

    _check_extent_format(extent)

    if isinstance(planes, dict):
        for group in planes.values():
            validate_planes(group, extent, tol=tol)
        return

    def assert_point_inside(point: Tuple[float, float, float], plane_no: int) -> None:
        if not point_inside_domain(point, extent):
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
        ValueError: if any ellipse fails validation
    """

    if isinstance(ellipses, dict):
        for group in ellipses.values():
            validate_ellipses(group, extent)
        return

    for idx, e in enumerate(ellipses, start=1):

        # --- Required fields ---
        if "center" not in e or "radii" not in e:
            raise ValueError(f"Ellipse #{idx} must have 'center' and 'radii' keys.")

        cx, cy, cz = e["center"]
        r1, r2 = e["radii"]

        if not all(np.isfinite([cx, cy, cz, r1, r2])):
            raise ValueError(
                f"Ellipse #{idx}: center and radii must be finite numbers."
            )

        if r1 <= 0 or r2 <= 0:
            raise ValueError(f"Ellipse #{idx}: radii must be positive.")

        # --- Extent check ---
        if extent is not None:

            R = max(r1, r2)
            kind, data = normalize_domain(extent)
            if kind == "extent":
                xmin, xmax, ymin, ymax, zmin, zmax = data
                inside = (
                    xmin <= cx - R and cx + R <= xmax and
                    ymin <= cy - R and cy + R <= ymax and
                    zmin <= cz - R and cz + R <= zmax
                )
            else:
                origin = data[0]
                basis = np.column_stack((data[1]-data[0], data[3]-data[0], data[4]-data[0]))
                inv_basis = np.linalg.inv(basis)
                q = inv_basis @ (np.array([cx, cy, cz]) - origin)
                # A radius-R sphere around the ellipse is conservative.  In local
                # coordinates its half-width along coordinate i is R*||row_i(B^-1)||.
                dq = R * np.linalg.norm(inv_basis, axis=1)
                inside = bool(np.all(q - dq >= -1e-9) and np.all(q + dq <= 1.0 + 1e-9))
            if not inside:
                raise ValueError(
                    f"Ellipse #{idx}: center ({cx},{cy},{cz}) with max radius {R} "
                    f"extends outside extent {extent}"
                )



def validate_triangulation(points: np.ndarray, extent: Tuple[float, float, float, float, float, float],
    raise_error: bool = True) -> np.ndarray:
    """
    Validate that triangulation points lie inside a given spatial extent.
    Supports points with shape:
        (n_points, 3) -> [x, y, z]
        (n_points, 4) -> [x, y, z, id]

    Args:
        points:
            Array containing triangulation coordinates.

        extent:
            (xmin, xmax, ymin, ymax, zmin, zmax)

        raise_error:
            If True, raises error for invalid points.
            If False, filters them out.

    Returns:
        np.ndarray:
            Valid points (same shape as input).
    """

    if isinstance(points, dict):
        return {k: validate_triangulation(np.asarray(v, dtype=float), extent, raise_error) for k, v in points.items()}

    if not isinstance(points, np.ndarray):
        points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, 3) or (n_points, 4)")

    if points.shape[1] not in (3, 4):
        raise ValueError("points must have 3 or 4 columns (x,y,z[,id])")


    xyz = points[:, :3]  # only spatial coordinates
    mask = points_inside_domain(xyz, extent)

    outside_points = points[~mask]

    if outside_points.size > 0:

        msg = (
            f"{outside_points.shape[0]} triangulation points lie outside the extent.\n"
            f"Domain: {np.asarray(extent).tolist()}\n"
            f"First few invalid points:\n{outside_points[:5]}"
        )

        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning("%s", msg)
            return points[mask]

    return points


def mesh_generator(ov: List[Tuple[int, int]], extent: List[float], well_tags: Optional[List[int]],
    source_tag: Optional[List[int]], shaft_tags: Optional[List[int]], tri_group_tags: Optional[List[int]], tri_surface_tags: Optional[List[int]],
    grid_litho: pd.DataFrame, mesh_size: float = 20.0, curve_mesh_size: float = 5.0,
    boundary_tags: Optional[List[int]] = None, gmsh_flag: bool = False, mapping_litho: LithoMappingModeType = LithoMappingMode.AUTOMATIC_CENTERS.value, merge_file: Optional[str] = None) -> MeshResults:

  """
    Generate an unstructured 3D tetrahedral mesh using Gmsh and export it
    in a meshio-compatible format.
    The function meshes 3D volumes using Gmsh, preserves embedded points, lines (wells), and surfaces (faults, triangulations),
    separates shaft-related tetrahedra from regular geological volumes, assigns lithology to tetrahedral blocks using nearest-neighbor
        classification against a geological grid, merges tetrahedra with identical lithology into larger blocks, and returns a cleaned mesh ready for export or simulation.

    Args:
        ov (List[Tuple[int, int]]): Gmsh entities defined as (dimension, tag), typically volume entities resulting from prior fragmentation.
        well_tags (List[int] or None): Gmsh physical group tags associated with wells (dimension = 1).
        source_tag (List[int] or None): Physical group tags of embedded source points (dimension = 0).
        shaft_tags (List[int] or None): Physical group tags identifying shaft volumes (dimension = 3).
        tri_group_tags (List[int]): Physical group tags assigned to triangulated *groups* (intermediate grouping before surface tagging).
        tri_surface_tags (List[int]): Physical group tags assigned to triangulated surfaces created from input point sets.
        grid_litho (pandas.DataFrame): Geological reference grid with four columns: x coordinate, y coordinate, z coordinate and
            lithology number. Used to assign lithology to tetrahedral elements via nearest-neighbor classification.
        mesh_size (float, optional): Global target mesh size for Gmsh (default: 20.0).
        boundary_tags (List(int)): List of tags of outer boundary surfaces to be preserved as seprate element blocks
        curve_mesh_size (float, optional): Mesh refinement factor based on curvature (Mesh.MeshSizeFromCurvature, default: 5.0).
        gmsh_flag (bool):  If True, enables writing the GMSH original mesh to the disk (default=False)
        mapping_litho (str): Controls lithological processing mode.
            Options:
                - "automatic_centers" : apply lithological mapping from structural models to the
                  mesh, voting on each raw block's cell centroids (default -- generally more
                  reliable near a fault plane than "automatic_corners", since cell centroids
                  are guaranteed-interior points, unlike nodes which sit on block boundaries)
                - "automatic_corners" : same as "automatic_centers" but votes on each raw
                  block's node coordinates instead; kept for comparison/backwards compatibility
                - "none"  : skip lithological mapping
                - "manual"   : merge lithology blocks using information provided by user
     merge_file (str, optional): Directory path containing merge definition file.
            Used only when mapping_litho="manual".
    Returns:
        nodes (numpy.ndarray): Array of mesh node coordinates with shape (n_nodes, 3).

        new_cells (List[meshio.CellBlock]): Mesh elements grouped by type. Tetrahedral elements are grouped
            by lithology, while: geological surfaces, triangulated surfaces, wells (lines), and source points,
            are preserved as separate element blocks.
  """

  ####################
  # Extract extent
  ###################
  volumes: List[int] = [tag for dim, tag in ov if dim == 3]

  x_min, x_max, y_min, y_max, z_min, z_max = extent

  ######################################
  # Add physical groups for volumes only
  ######################################
  for i, tag in enumerate(volumes):

    gmsh.model.addPhysicalGroup(3, [tag], i + 1)
    gmsh.model.setPhysicalName(3, i + 1, f"Volume {i + 1}")

    logger.debug("Volume %d physical tag added", i + 1)

  gmsh.model.occ.synchronize()

##############################
# Embed sources into all volumes
##############################
  if source_tag:

    for volume_tag in volumes:

        for s_tag in source_tag:

            gmsh.model.mesh.embed(0, [s_tag], 3, volume_tag)

  ############################################################
  # Specify global mesh size and mesh the partitioned model
  ############################################################
  gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)

  gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

  gmsh.option.setNumber("General.Verbosity", 4)

  gmsh.model.occ.synchronize()

  ###############################################
  # Embed triangulated surfaces into all volumes
  ###############################################
  if tri_surface_tags:

    for volume_tag in volumes:

        gmsh.model.mesh.embed(
            2,                  # surface dimension
            tri_surface_tags,   # surface entity tags
            3,                  # target dimension
            volume_tag
        )

  ##############################
  # Generate 3D mesh and save
  ##############################
  gmsh.model.mesh.generate(3)

  if gmsh_flag:
    gmsh.write("mesh.msh")

  gmsh.model.occ.synchronize()


  #########################
  # Convert to meshio
  #########################
  # NODES
  node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
  nodes = node_coords_flat.reshape(-1, 3)

  tag2idx: Dict[int, int] = {
    tag: i for i, tag in enumerate(node_tags)
  }

  # ELEMENT TYPE MAP
  gmsh_to_meshio: Dict[int, Tuple[str, int]] = {
    15: ("vertex", 1),
    1:  ("line", 2),
    2:  ("triangle", 3),
    4:  ("tetra", 4),
  }


  cells: List[meshio.CellBlock] = []

  cell_sets: Dict[str, List[int]] = {}

  cell_data: Dict[str, List[np.ndarray]] = {
    "gmsh:physical": [],
    "gmsh:geometrical": [],
  }


  #########################################################
  # FAST EXTRACTION (all dims, keeps cell_sets + cell_data)
  #########################################################
  def extract_all_fast():

    # entity -> physical mapping
    entity_to_phys = {}
    entity_to_phys_tag = {}

    for dim, phys_tag in gmsh.model.getPhysicalGroups():

        name = gmsh.model.getPhysicalName(dim, phys_tag)

        if not name:
            name = f"phys_{dim}_{phys_tag}"

        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

        for ent in entities:
            entity_to_phys[(dim, ent)] = name
            entity_to_phys_tag[(dim, ent)] = phys_tag


    # Loop over all entities
    for dim in [3, 2, 1, 0]:

        entities = gmsh.model.getEntities(dim)

        for dim_e, ent in entities:

            key = (dim_e, ent)

            if key not in entity_to_phys:
                continue

            name = entity_to_phys[key]
            phys_tag = entity_to_phys_tag[key]

            block_ids = []

            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim_e, ent)

            for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):

                if etype not in gmsh_to_meshio:
                    continue

                cell_type, n_nodes = gmsh_to_meshio[etype]

                conn = enodes.reshape(-1, n_nodes)

                # FAST node mapping
                conn = np.array(
                    [tag2idx[t] for t in conn.ravel()],
                    dtype=np.int64
                ).reshape(conn.shape)

                block = meshio.CellBlock(cell_type, conn)
                cells.append(block)

                block_id = len(cells) - 1
                block_ids.append(block_id)

                n_cells = len(conn)

                cell_data["gmsh:physical"].append(
                    np.full(n_cells, phys_tag, dtype=np.int32)
                )

                cell_data["gmsh:geometrical"].append(
                    np.full(n_cells, ent, dtype=np.int32)
                )

            if block_ids:
                if name not in cell_sets:
                    cell_sets[name] = []
                cell_sets[name].extend(block_ids)

  extract_all_fast()

  ###################################
  # SPECIAL HANDLING FOR LINES (WELLS)
  ###################################
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


  ###########################################
  # Merge ALL line segments per physical group
  ###########################################
  for name, conns in line_blocks.items():

    merged = np.vstack(conns)

    cell_type = "line" if merged.shape[1] == 2 else "line3"

    block = meshio.CellBlock(cell_type, merged)

    cells.append(block)

    block_id = len(cells) - 1

    cell_sets[name] = [block_id]

    # Recover physical tag from name
    phys_tag = None

    for dim, ptag in gmsh.model.getPhysicalGroups(1):

        pname = gmsh.model.getPhysicalName(1, ptag)

        if pname == name:
            phys_tag = ptag
            break

    if phys_tag is None:
        phys_tag = -1

    # Add matching cell_data
    n_cells = len(merged)

    cell_data["gmsh:physical"].append(
        np.full(n_cells, phys_tag, dtype=np.int32)
    )

    cell_data["gmsh:geometrical"].append(
        np.full(n_cells, phys_tag, dtype=np.int32)
    )


  #######################
  # FILTER UNUSED NODES
  #####################

  # Find all used node indices
  used = np.unique(np.concatenate([cb.data.ravel() for cb in cells]))

  #Build a direct lookup array
  max_idx = used[-1]
  lookup = np.full(max_idx + 1, -1, dtype=int)
  lookup[used] = np.arange(len(used))

  # Subset node coordinates
  nodes = nodes[used]

  # Remap connectivity
  new_cells_tmp = []
  new_cell_data = {
    "gmsh:physical": [],
    "gmsh:geometrical": []
  }

  for i, cb in enumerate(cells):

    remapped = lookup[cb.data]

    new_cells_tmp.append(
        meshio.CellBlock(cb.type, remapped)
    )

    new_cell_data["gmsh:physical"].append(
        cell_data["gmsh:physical"][i]
    )

    new_cell_data["gmsh:geometrical"].append(
        cell_data["gmsh:geometrical"][i]
    )

  cells = new_cells_tmp
  cell_data = new_cell_data


  # Build mesh
  mesh_model = meshio.Mesh(
    points=nodes,
    cells=cells,
    cell_sets=cell_sets,
    cell_data=cell_data
  )


  #####################
  # Inspect mesh blocks
  ####################
  for block in mesh_model.cells:
    logger.debug("Cell type: %s, Number of cells: %d", block.type, len(block.data))

  cells: List[meshio.CellBlock] = mesh_model.cells
  nodes: NDArray[np.float64] = mesh_model.points

  tetra_blocks_all: List[meshio.CellBlock] = [
    block for block in cells if block.type == "tetra"]

  logger.debug("Number of tetrahedral blocks: %d", len(tetra_blocks_all))


# #######################################
# SHAFTS (separate tetra blocks like wells)
##########################################
  logger.debug("Extracting shaft and regular tetra blocks")

  shaft_blocks_dict: Dict[int, List[meshio.CellBlock]] = {}
  regular_blocks: List[meshio.CellBlock] = []

  # initialize shaft storage
  shaft_tag_set = set(shaft_tags)

  for tag in shaft_tag_set:
    shaft_blocks_dict[tag] = []


  ############################
  # SINGLE LOOP over cell_sets
  ############################
  for name, block_ids in cell_sets.items():

    for bid in block_ids:

        block = cells[bid]

        # only tetrahedral elements
        if block.type not in {"tetra", "tetra10"}:
            continue

        assigned_to_shaft = False

        # CHECK SHAFT TAGS (3500+i)
        for shaft_tag in shaft_tag_set:
            if str(shaft_tag) in name or name.endswith(str(shaft_tag)):
                shaft_blocks_dict[shaft_tag].append(block)
                assigned_to_shaft = True
                break

        # REGULAR VOLUMES
        if not assigned_to_shaft:
            regular_blocks.append(block)


  ################################
  # MERGE SHAFT BLOCKS (per shaft)
  ################################
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

    logger.debug("Shaft %s: merged %d tetra elements", shaft_tag, len(merged_data))


  ##############
  # DEBUG PRINT
  #############
  for shaft_tag, blocks in shaft_blocks_dict.items():
    if blocks:
        logger.debug("Shaft %s: %d tetra blocks", shaft_tag, len(blocks))
    else:
        logger.debug("Shaft %s: not found", shaft_tag)

  logger.debug("Regular volume blocks: %d", len(regular_blocks))


  #####################
  # SORT REGULAR BLOCKS
  ####################
  block_centroids: List[Tuple[int, float, float]] = []

  for i, block in enumerate(regular_blocks):

    node_ids: NDArray[np.int64] = np.unique(block.data)

    centroid = nodes[node_ids].mean(axis=0)  # (x, y, z)

    block_centroids.append((i, centroid[2], centroid[0]))  # (index, z, x)


  # Sort by z ascending, then x descending
  sorted_indices: List[int] = [
    i for i, _, _ in sorted(block_centroids, key=lambda t: (t[1], -t[2]))
]


  # Reorder blocks
  regular_blocks: List[meshio.CellBlock] = [
    regular_blocks[i] for i in sorted_indices
]

  # Populated by the AUTOMATIC_CENTERS/AUTOMATIC_CORNERS branches below (by object identity,
  # not index -- new_cells gets more non-lithology blocks appended to it
  # after this point: shafts, wells, sources, fault surfaces, triangulated
  # surfaces, boundary surfaces). Used right before this function returns
  # to build cell_data["block_id"], so mesh_generator's caller can trust
  # the lithology mapping directly instead of having to re-derive it.
  # MANUAL/NONE don't populate this -- MANUAL groups by user-supplied
  # indices with no inherent lithology-id semantics, and NONE explicitly
  # skips lithology mapping.
  litho_block_id_by_identity: Dict[int, int] = {}

  if mapping_litho == LithoMappingMode.AUTOMATIC_CORNERS:

    # Lithology assignment (ONLY regular blocks)
    grid_coords: NDArray[np.float64] = grid_litho.iloc[:, :3].to_numpy()
    grid_litho_values: NDArray[np.int64] = grid_litho.iloc[:, 3].to_numpy()

    tree = cKDTree(grid_coords)

    threshold_ratio: float = 0.60
    lithology_numbers: List[int] = []
    vote_purities: List[float] = []

    # Assign lithology per regular block
    for block in regular_blocks:

        node_ids: NDArray[np.int64] = np.unique(block.data)
        node_coords: NDArray[np.float64] = nodes[node_ids]

        _, nearest_idx = tree.query(node_coords)

        node_litho: NDArray[np.int64] = grid_litho_values[nearest_idx]

        unique_vals, counts = np.unique(node_litho, return_counts=True)

        max_idx: int = np.argmax(counts)
        vote_purities.append(float(counts[max_idx] / len(node_ids)))

        if counts[max_idx] / len(node_ids) >= threshold_ratio:
            lithology_numbers.append(int(unique_vals[max_idx]))

        else:
            centroid: NDArray[np.float64] = node_coords.mean(axis=0)

            distances = np.linalg.norm(node_coords - centroid, axis=1)

            nearest_sample_idx = np.argmin(distances)

            lithology_numbers.append(int(node_litho[nearest_sample_idx]))


    # Merge geological blocks by lithology
    litho_to_blocks: Dict[int, List[meshio.CellBlock]] = defaultdict(list)
    vote_purities_by_lithology: Dict[int, List[float]] = defaultdict(list)

    for lith, block, purity in zip(lithology_numbers, regular_blocks, vote_purities):
        litho_to_blocks[lith].append(block)
        vote_purities_by_lithology[lith].append(purity)

    _check_lithology_mapping_reliability(
        litho_to_blocks, vote_purities_by_lithology,
        grid_litho_values, grid_coords, mesh_size, "automatic_corners",
    )

    merged_regular_blocks: List[meshio.CellBlock] = []
    merged_regular_lith_ids: List[int] = []

    for lith, blocks in sorted(litho_to_blocks.items()):

        merged_data: NDArray[np.int64] = np.vstack(
            [b.data for b in blocks if len(b.data) > 0]
        )

        merged_regular_blocks.append(
            meshio.CellBlock("tetra", merged_data)
        )
        merged_regular_lith_ids.append(lith)


    # Sort regular blocks by depth
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

    # Propagate the real lithology id for each of these blocks out to
    # cell_data["block_id"] (see the identity-based lookup right before
    # this function's return) -- by object identity, so this survives
    # regardless of how many more non-lithology blocks (wells, sources,
    # fault surfaces, ...) get appended to new_cells afterward.
    for block, lith in zip(
        sorted_regular_blocks, [merged_regular_lith_ids[i] for i in sorted_indices]
    ):
        litho_block_id_by_identity[id(block)] = lith


    # FINAL MERGE: regular + shafts
    merged_tetra_blocks_sorted = []

    # regular lithology blocks (sorted)
    merged_tetra_blocks_sorted.extend(sorted_regular_blocks)

    # shaft blocks (already merged earlier!)
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

        logger.debug("Shaft %s: merged %d tetra elements", shaft_tag, len(merged_data))


  elif mapping_litho == LithoMappingMode.AUTOMATIC_CENTERS:

    # Same overall structure as AUTOMATIC_CORNERS (per-block vote -> merge
    # by lithology -> sort by depth), but the vote itself is taken on each
    # block's individual *cell centroids* (guaranteed-interior points)
    # instead of its *node* coordinates (which sit on block boundaries --
    # exactly where a fault plane tends to be, and can give a genuinely
    # ambiguous nearest-neighbor vote no matter how good the algorithm is).
    grid_coords: NDArray[np.float64] = grid_litho.iloc[:, :3].to_numpy()
    grid_litho_values: NDArray[np.int64] = grid_litho.iloc[:, 3].to_numpy()

    tree = cKDTree(grid_coords)

    lithology_numbers: List[int] = []
    vote_purities: List[float] = []

    for block in regular_blocks:

        cell_centroids: NDArray[np.float64] = nodes[block.data].mean(axis=1)

        _, nearest_idx = tree.query(cell_centroids)

        cell_litho: NDArray[np.int64] = grid_litho_values[nearest_idx]

        unique_vals, counts = np.unique(cell_litho, return_counts=True)

        max_idx: int = np.argmax(counts)
        vote_purities.append(float(counts[max_idx] / len(cell_centroids)))

        lithology_numbers.append(int(unique_vals[max_idx]))


    # Merge geological blocks by lithology
    litho_to_blocks: Dict[int, List[meshio.CellBlock]] = defaultdict(list)
    vote_purities_by_lithology: Dict[int, List[float]] = defaultdict(list)

    for lith, block, purity in zip(lithology_numbers, regular_blocks, vote_purities):
        litho_to_blocks[lith].append(block)
        vote_purities_by_lithology[lith].append(purity)

    _check_lithology_mapping_reliability(
        litho_to_blocks, vote_purities_by_lithology,
        grid_litho_values, grid_coords, mesh_size, "automatic_centers",
    )

    merged_regular_blocks: List[meshio.CellBlock] = []
    merged_regular_lith_ids: List[int] = []

    for lith, blocks in sorted(litho_to_blocks.items()):

        merged_data: NDArray[np.int64] = np.vstack(
            [b.data for b in blocks if len(b.data) > 0]
        )

        merged_regular_blocks.append(
            meshio.CellBlock("tetra", merged_data)
        )
        merged_regular_lith_ids.append(lith)


    # Sort regular blocks by depth
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

    # Propagate the real lithology id for each of these blocks out to
    # cell_data["block_id"] -- see the matching comment in the AUTOMATIC_CORNERS branch.
    for block, lith in zip(
        sorted_regular_blocks, [merged_regular_lith_ids[i] for i in sorted_indices]
    ):
        litho_block_id_by_identity[id(block)] = lith


    # FINAL MERGE: regular + shafts
    merged_tetra_blocks_sorted = []

    # regular lithology blocks (sorted)
    merged_tetra_blocks_sorted.extend(sorted_regular_blocks)

    # shaft blocks (already merged earlier!)
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

        logger.debug("Shaft %s: merged %d tetra elements", shaft_tag, len(merged_data))


  elif mapping_litho == LithoMappingMode.MANUAL:

    if merge_file is None:
        raise ValueError(
            "mapping_litho='merge' requires merge_file"
        )

    # Read block groups
    block_groups = []

    with open(merge_file, "r") as f:

        for line in f:

            line = line.split("#")[0].strip()

            if not line:
                continue

            tokens = re.split(r"[,\s]+", line)

            group = [int(x) - 1 for x in tokens if x != ""]
            block_groups.append(group)


    # VALIDATION 1: number of groups vs blocks
    n_blocks_in_file = sum(len(g) for g in block_groups)

    if n_blocks_in_file != len(regular_blocks):
        raise ValueError(
            f"Merge file contains {n_blocks_in_file} block IDs "
            f"but mesh has {len(regular_blocks)} regular blocks."
        )


    # VALIDATION 2: flatten all values
    all_vals = [v for g in block_groups for v in g]


    # VALIDATION 3: duplicates check
    unique_vals = set(all_vals)

    if len(all_vals) != len(unique_vals):
        duplicates = [
            x for x in unique_vals
            if all_vals.count(x) > 1
        ]

        raise ValueError(
            f"Duplicate block IDs found in merge file : {duplicates}"
        )


    # FINAL: mapping is valid
    logger.info("Merge file validation passed")


    # Merge blocks according to grouping
    merged_regular_blocks = []

    for group in block_groups:

        merged_data_list = []

        for bid in group:

            block = regular_blocks[bid]

            if len(block.data) > 0:
                merged_data_list.append(block.data)

        if len(merged_data_list) == 0:
            continue

        merged_data = np.vstack(merged_data_list)

        # IMPORTANT: keep same element type as original blocks
        cell_type = regular_blocks[group[0]].type

        merged_regular_blocks.append(
            meshio.CellBlock(cell_type, merged_data)
        )


    # Sort regular blocks by depth
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


    # FINAL MERGE: regular + shafts
    merged_tetra_blocks_sorted = []

    # regular lithology blocks (sorted)
    merged_tetra_blocks_sorted.extend(sorted_regular_blocks)

    # shaft blocks (already merged earlier!)
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

        logger.debug("Shaft %s: merged %d tetra elements", shaft_tag, len(merged_data))


  elif mapping_litho == LithoMappingMode.NONE:

    # FINAL MERGE: regular + shafts
    merged_tetra_blocks_sorted = []

    # 1. regular lithology blocks (sorted)
    merged_tetra_blocks_sorted.extend(regular_blocks)

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

        logger.debug("Shaft %s: merged %d tetra elements", shaft_tag, len(merged_data))


  #######################
  # Rebuild final cell list
  #########################
  cells_n: List[meshio.CellBlock] = [
    block for block in cells
    if block.type not in {"tetra"}
  ]

  cells_n.extend(merged_tetra_blocks_sorted)

  new_cells: List[meshio.CellBlock] = [
    block for block in cells_n
    if block.type not in {"line", "triangle", "vertex"}
  ]

  logger.debug("Extracting wells, sources, fault surfaces, and triangulated surfaces")

  found_any = False


  ###############################################
  # Detect fault tags dynamically from cell names
  # #############################################
  fault_tags_set = set()

  for name in cell_sets:

    numbers_in_name = [int(m) for m in re.findall(r"\d+", name)]

    for n in numbers_in_name:

        if n >= 1000 and n % 1000 == 0:
            fault_tags_set.add(n)

  fault_tags = sorted(fault_tags_set)

  fault_tri_blocks = {tag: [] for tag in fault_tags}

  logger.debug("Detected fault tags: %s", fault_tags)

  # Triangulated surfaces
  tri_surface_blocks = defaultdict(list)

   # Wells (1060–1070)
  well_blocks = {tag: [] for tag in well_tags if 1060 <= tag <= 1070}

# Sources (points)
  vertex_blocks: List[meshio.CellBlock] = []


  ###############################
  # Single loop over cell_sets
  ###############################
  for name, block_ids in cell_sets.items():

    # Extract numbers from name
    numbers_in_name = [
        int(n) for n in re.findall(r'\d+', name)
    ]

    for bid in block_ids:

        block = cells[bid]

        # TRIANGULATED SURFACES
        phys = cell_data["gmsh:physical"][bid]

        if len(phys) == 0:
            continue

        phys_tag = phys[0]

        if block.type == "triangle" and phys_tag in tri_group_tags:
            tri_surface_blocks[phys_tag].append(block.data)

        # FAULT SURFACES
        for tag in fault_tags:

            if (
                tag in numbers_in_name
                and block.type == "triangle"
            ):

                fault_tri_blocks[tag].append(block)

        # WELLS
        for tag in well_blocks:

            if (
                tag in numbers_in_name
                and block.type == "line"
            ):

                well_blocks[tag].append(block)

        # SOURCES
        if block.type == "vertex":

            vertex_blocks.append(block)

  #################################################
  # ADD FAULT SURFACES (merge by tag, filter outside)
  #################################################
  for tag, blocks in fault_tri_blocks.items():
    if not blocks:
        logger.debug("Fault tag %s not found", tag)
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
        logger.debug("Fault tag %s: merged %d triangles from %d blocks", tag, len(merged_data), len(blocks))
        found_any = True
    else:
        logger.debug("Fault tag %s: no triangles inside bounding box after filtering", tag)

  ###########################
  # ADD TRIANGULATED SURFACES
  ###########################
  if tri_surface_blocks:

    for phys_tag, blocks in tri_surface_blocks.items():

        if len(blocks) == 0:
            continue

        merged_data = np.vstack(blocks)

        new_cells.append(
            meshio.CellBlock("triangle", merged_data)
        )

        logger.debug("Triangulated surface %s: %d triangles", phys_tag, len(merged_data))

        found_any = True

  ############
  # ADD WELLS
  ############
  for tag, blocks in well_blocks.items():

    if blocks:
        new_cells.extend(blocks)
        logger.debug("Well tag %s added (%d line blocks)", tag, len(blocks))
        found_any = True

    else:
        logger.debug("Well tag %s not found", tag)


  #############
  # ADD SOURCES
  #############
  if vertex_blocks:

    new_cells.extend(vertex_blocks)
    logger.debug("Added %d source point blocks", len(vertex_blocks))
    found_any = True

  ########################
  # ADD BOUNDARY SURFACES
  ########################

  boundary_blocks = {}

  if boundary_tags:

    for btag in boundary_tags:

        name = gmsh.model.getPhysicalName(2, btag)

        block_ids = cell_sets.get(name, [])

        tri_list = []

        for bid in block_ids:

            block = cells[bid]

            if block.type != "triangle":
                continue

            if len(block.data) == 0:
                continue

            tri_list.append(block.data)

        if tri_list:

            merged = np.vstack(tri_list)

            boundary_blocks[name] = meshio.CellBlock(
                "triangle",
                merged,
            )

  for name, block in boundary_blocks.items():

    new_cells.append(block)

    logger.debug("%s merged (%d triangles)", name, len(block.data))

    found_any = True


  # Propagate the real per-block lithology id (AUTOMATIC_CENTERS/AUTOMATIC_CORNERS only --
  # see litho_block_id_by_identity above) out to cell_data["block_id"],
  # matching the convention create_structured_mesh_data/
  # create_implicit_structured_mesh already use, so callers can trust the
  # mapping directly instead of re-deriving it. Sentinel -1 for every block
  # that isn't a lithology block (shafts, wells, sources, fault surfaces,
  # triangulated/boundary surfaces). Left out entirely for MANUAL/NONE --
  # litho_block_id_by_identity stays empty for both, so there's nothing
  # meaningful to propagate.
  if litho_block_id_by_identity and new_cells:
    cell_data["block_id"] = [
        np.full(len(block.data), litho_block_id_by_identity.get(id(block), -1), dtype=int)
        for block in new_cells
    ]

  # FINAL CHECK
  if not found_any:
    logger.debug("No surfaces, wells, or sources found")

  if new_cells:
    return nodes, new_cells, cell_data
  else:
    return nodes, cells, cell_data


WellData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='well_list', color='aqua', identifier='mesh::WellListData', controlled='Table|x3')]

SourcesData = typing.Annotated[typing.Union[List[Tuple[float, float, float]], Dict[int, List[Tuple[float, float, float]]]], AnnotatedScriptType(name='sources', color='aqua', identifier='mesh::SourcesData', controlled='Table|3|X|Y|Z')]

ShaftData = typing.Annotated[typing.Union[List[Tuple[float, ...]], Dict[int, List[Tuple[float, ...]]]], AnnotatedScriptType(name='shaft_list', color='aqua', identifier='mesh::ShaftListData', controlled='Table|x7')]

PlaneData = typing.Annotated[typing.Union[List[Tuple[float, ...]], Dict[int, List[Tuple[float, ...]]]], AnnotatedScriptType(name='plane_list', color='aqua', identifier='mesh::PlaneListData', controlled='Table|x12')]

EllipseData = typing.Annotated[typing.Union[List[Dict[str, typing.Any]], Dict[int, List[Dict[str, typing.Any]]]], AnnotatedScriptType( name='ellipse_list', color='aqua', identifier='mesh::EllipseListData' )]

TriangulationData = typing.Annotated[typing.Union[NDArray[np.float64], Dict[int, NDArray[np.float64]]], AnnotatedScriptType( name='triangulation', color='aqua',
        identifier='mesh::TriangulationData',controlled='Table|3|X|Y|Z')]

# the file must end with "wells.csv", e.g., "example_wells.csv", etc.
WellCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::well_csv', controlled='RemoteFile|endswith=wells.csv')]

@wbgeo_component(description='Loads a well from a wells.CSV file',
                 title='Load Well',  # The title shown in the GUI
                 color=MESH_ENGINEERING_COLOR,  # the color of the components
                 border_color='#000000',  # and its border color
                 group=MESH_ENGINEERING_GROUP,
                 identifier='wbgeo::meshing_load_well_from_csv',  # a unique identifier
                 return_name='Wells',  # the name for the returned-port
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

            # Parse depending on format
            if key_hierarchical:

                if len(parts) != 5:
                    raise ValueError(
                        "Invalid well line (expected 5 columns: id,x,y,z,key)"
                    )

                well_id, x, y, z, key = parts
                key = int(key)

                well_key_id = (well_id, key)

            else:

                if len(parts) != 4:
                    raise ValueError(
                        "Invalid well line (expected 4 columns: id,x,y,z)"
                    )

                well_id, x, y, z = parts
                well_key_id = well_id
                key = None

            # Numeric validation
            try:
                x = float(x)
                y = float(y)
                z = float(z)

            except ValueError:
                raise ValueError("Non-numeric value in well definition")

            # Store
            if well_key_id not in named_well_data:
                named_well_data[well_key_id] = []

            named_well_data[well_key_id].append((x, y, z))


    # Validate completeness
    incorrect = [
        str(wid) if not isinstance(wid, str) else wid
        for wid, pts in named_well_data.items()
        if len(pts) < 2
    ]

    if incorrect:
        raise ValueError(
            f"Some well(s) {incorrect} are missing their second point"
        )


    # Flatten wells
    flattened_wells = {
        wid: tuple(c for pt in pts for c in pt)
        for wid, pts in named_well_data.items()
    }


    # Return format
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
                 color=MESH_ENGINEERING_COLOR,  # the color of the components
                 border_color='#000000',  # and its border color
                 group=MESH_ENGINEERING_GROUP,
                 identifier='wbgeo::meshing_load_source_from_csv',  # a unique identifier
                 return_name='Sources',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def load_sources_from_csv(source_file: SourceCSVDataType, key_hierarchical: bool = False) -> SourcesData:
    """Load sources. Hierarchical CSV format is: id,x,y,z,key (key must be 1 or 2)."""
    named = {}
    with open(source_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            expected = 5 if key_hierarchical else 4
            if len(parts) != expected:
                raise ValueError(f"Line {line_no}: expected {expected} columns ({'id,x,y,z,key' if key_hierarchical else 'id,x,y,z'}).")
            source_id, *rest = parts
            try:
                x, y, z = map(float, rest[:3])
                key = int(rest[3]) if key_hierarchical else None
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: non-numeric source value: {line.strip()}") from exc
            if not np.all(np.isfinite([x, y, z])):
                raise ValueError(f"Line {line_no}: source coordinates must be finite.")
            if key_hierarchical and key not in (1, 2):
                raise ValueError(f"Line {line_no}: hierarchical key must be 1 or 2, got {key}.")
            ident = (source_id, key) if key_hierarchical else source_id
            if ident in named:
                raise ValueError(f"Duplicate source definition for source ID '{source_id}'" + (f" and key {key}" if key_hierarchical else ""))
            named[ident] = (x, y, z)
    if not key_hierarchical:
        return list(named.values())
    out = {1: [], 2: []}
    for (_, key), value in named.items():
        out[key].append(value)
    return out


# the file must end with "ellipses.csv", e.g., "example_ellipses.csv", etc.
EllipseCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::ellipse_csv', controlled='RemoteFile|endswith=ellipses.csv')]

@wbgeo_component(
    description='Loads ellipses from an ellipses.CSV file',
    title='Load Ellipse',
    color=MESH_ENGINEERING_COLOR,  # the color of the components
    border_color='#000000',  # and its border color
    group=MESH_ENGINEERING_GROUP,
    identifier='wbgeo::meshing_load_ellipse_from_csv',
    return_name='Ellipses',
)
def load_ellipses_from_csv(ellipse_file: EllipseCSVDataType, key_hierarchical: bool = False) -> EllipseData:
    """Load ellipses; hierarchical rows append a final key (1 or 2)."""
    ellipses_by_key = {1: [], 2: []}
    ellipses = []
    with open(ellipse_file, "r") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            key = None
            if key_hierarchical:
                if len(parts) < 6:
                    raise ValueError(f"Line {line_number}: hierarchical ellipse row is missing its key.")
                try:
                    key = int(parts[-1])
                except ValueError as exc:
                    raise ValueError(f"Line {line_number}: ellipse hierarchy key must be integer 1 or 2.") from exc
                if key not in (1, 2):
                    raise ValueError(f"Line {line_number}: hierarchical key must be 1 or 2, got {key}.")
                parts = parts[:-1]

            # Existing accepted layouts: 5/7/10/13 values without ID,
            # or 6/8/11/14 with an ID in the first column.
            if len(parts) in (6, 8, 11, 14):
                parts = parts[1:]
            if len(parts) not in (5, 7, 10, 13):
                raise ValueError(f"Line {line_number}: invalid ellipse column count ({len(parts)} after removing ID/key).")
            try:
                cx, cy, cz, r1, r2 = map(float, parts[:5])
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: invalid numeric value in ellipse required fields.") from exc
            e = {"center": (cx, cy, cz), "radii": (r1, r2), "angle1": 0.0, "angle2": 2*np.pi, "zAxis": [0.0, 0.0, 1.0]}
            idx = 5
            try:
                if len(parts) >= 7:
                    e["angle1"] = float(parts[idx]) if parts[idx] else 0.0
                    e["angle2"] = float(parts[idx+1]) if parts[idx+1] else 2*np.pi
                    idx += 2
                if len(parts) >= idx + 3:
                    e["zAxis"] = [float(v) for v in parts[idx:idx+3]]
                    idx += 3
                if len(parts) >= idx + 3:
                    e["xAxis"] = [float(v) for v in parts[idx:idx+3]]
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: invalid numeric value in optional ellipse fields.") from exc
            validate_ellipses([e], None)
            (ellipses_by_key[key] if key_hierarchical else ellipses).append(e)
    return ellipses_by_key if key_hierarchical else ellipses


# the file must end with "shaftss.csv", e.g., "example_shaftss.csv", etc.
ShaftCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::shaft_csv', controlled='RemoteFile|endswith=shafts.csv')]

@wbgeo_component(description='Loads a shaft from a shafts.CSV file',
                 title='Load Shaft',  # The title shown in the GUI
                 color=MESH_ENGINEERING_COLOR,  # the color of the components
                 border_color='#000000',  # and its border color
                 group=MESH_ENGINEERING_GROUP,
                 identifier='wbgeo::meshing_load_shaft_from_csv',  # a unique identifier
                 return_name='Shafts',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def load_shafts_from_csv(shaft_file: ShaftCSVDataType, key_hierarchical: bool = False) -> ShaftData:
    """Load shafts. Hierarchical CSV format is: id,cx,cy,cz,ax,ay,az,radius,key."""
    named = {}
    with open(shaft_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            expected = 9 if key_hierarchical else 8
            if len(parts) != expected:
                raise ValueError(f"Line {line_no}: expected {expected} shaft columns.")
            shaft_id, *rest = parts
            try:
                vals = tuple(map(float, rest[:7]))
                key = int(rest[7]) if key_hierarchical else None
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: non-numeric shaft value: {line.strip()}") from exc
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Line {line_no}: shaft values must be finite.")
            if np.linalg.norm(vals[3:6]) == 0.0:
                raise ValueError(f"Line {line_no}: shaft axis must be non-zero.")
            if vals[6] <= 0.0:
                raise ValueError(f"Line {line_no}: shaft radius must be positive.")
            if key_hierarchical and key not in (1, 2):
                raise ValueError(f"Line {line_no}: hierarchical key must be 1 or 2, got {key}.")
            ident = (shaft_id, key) if key_hierarchical else shaft_id
            if ident in named:
                raise ValueError(f"Duplicate shaft definition for shaft ID '{shaft_id}'" + (f" and key {key}" if key_hierarchical else ""))
            named[ident] = vals
    if not key_hierarchical:
        return list(named.values())
    out = {1: [], 2: []}
    for (_, key), value in named.items():
        out[key].append(value)
    return out

# the file must end with "planes.csv", e.g., "example_planes.csv", etc.
PlaneCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::plane_csv', controlled='RemoteFile|endswith=planes.csv')]

@wbgeo_component(description='Loads a plane from a planes.CSV file',
                 title='Load Plane',  # The title shown in the GUI
                 color=MESH_ENGINEERING_COLOR,  # the color of the components
                 border_color='#000000',  # and its border color
                 group=MESH_ENGINEERING_GROUP,
                 identifier='wbgeo::meshing_load_plane_from_csv',  # a unique identifier
                 return_name='planes',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def load_planes_from_csv(plane_file: PlaneCSVDataType, key_hierarchical: bool = False) -> PlaneData:
    """Load extra planes. Hierarchical CSV format is: id,x,y,z,key."""
    named = {}
    with open(plane_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            expected = 5 if key_hierarchical else 4
            if len(parts) != expected:
                raise ValueError(f"Line {line_no}: expected {expected} plane columns.")
            plane_id, *rest = parts
            try:
                point = tuple(map(float, rest[:3]))
                key = int(rest[3]) if key_hierarchical else None
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: non-numeric plane value: {line.strip()}") from exc
            if not np.all(np.isfinite(point)):
                raise ValueError(f"Line {line_no}: plane coordinates must be finite.")
            if key_hierarchical and key not in (1, 2):
                raise ValueError(f"Line {line_no}: hierarchical key must be 1 or 2, got {key}.")
            ident = (plane_id, key) if key_hierarchical else plane_id
            named.setdefault(ident, []).append(point)
    bad = [str(k) for k, pts in named.items() if len(pts) < 4]
    if bad:
        raise ValueError(f"Some plane(s) {bad} have fewer than 4 points.")
    flattened = {k: tuple(c for pt in pts for c in pt) for k, pts in named.items()}
    if not key_hierarchical:
        return list(flattened.values())
    out = {1: [], 2: []}
    for (_, key), value in flattened.items():
        out[key].append(value)
    return out

TriangulationsPlanesData = typing.Annotated[typing.Union[str, typing.List[str]], AnnotatedScriptType(name='path',
        color='aqua', identifier='wbgeo::triangulations_planes_csv', controlled='RemoteFile|endswith=.csv')]

@wbgeo_component(
    description='Load planes coordinates from a CSV to do triangulations',
    title='Load Triangulations Planes',
    color=MESH_ENGINEERING_COLOR,  # the color of the components
    border_color='#000000',  # and its border color
    group=MESH_ENGINEERING_GROUP,
    identifier='wbgeo::meshing_load_triangulations_planes_from_csv',
    return_name='triangulations_planes',
)
def load_triangulations_planes_from_csv(csv_file: TriangulationsPlanesData, key_hierarchical: bool = False) -> TriangulationData:
    """Load triangulation points. Hierarchical CSV rows are x,y,z,key.

    The fourth output column remains the triangulation/surface id (file number),
    so the hierarchy key is used only for splitting and is not passed downstream.
    """
    csv_files = [csv_file] if isinstance(csv_file, str) else list(csv_file)
    grouped = {1: [], 2: []}
    all_points = []
    for file_id, file_path in enumerate(csv_files, 1):
        with open(file_path, "r") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                expected = 4 if key_hierarchical else 3
                if len(parts) != expected:
                    raise ValueError(f"{file_path} line {line_number}: expected {expected} columns ({'x,y,z,key' if key_hierarchical else 'x,y,z'}), got {len(parts)}")
                try:
                    x, y, z = map(float, parts[:3])
                    key = int(parts[3]) if key_hierarchical else None
                except ValueError as exc:
                    raise ValueError(f"{file_path} line {line_number}: non-numeric value found: {line}") from exc
                if not np.all(np.isfinite([x, y, z])):
                    raise ValueError(f"{file_path} line {line_number}: coordinates must be finite.")
                if key_hierarchical and key not in (1, 2):
                    raise ValueError(f"{file_path} line {line_number}: hierarchical key must be 1 or 2, got {key}.")
                row = [x, y, z, file_id]
                (grouped[key] if key_hierarchical else all_points).append(row)
    if key_hierarchical:
        return {k: np.asarray(v, dtype=np.float64).reshape((-1, 4)) for k, v in grouped.items()}
    if not all_points:
        raise ValueError("No points loaded from the CSV file(s).")
    return np.asarray(all_points, dtype=np.float64)

def classify_boundary_nodes(nodes: NDArray[np.float64], extent, tol_ratio: float = 1e-4) -> Dict[str, List[int]]:
    """Classify nodes on the six local faces of either supported domain form."""
    nodes = np.asarray(nodes, dtype=float)
    local = np.asarray(domain_local_coordinates(nodes, extent), dtype=float)
    boundary_groups = {
        "left": [], "right": [], "front": [], "back": [], "bottom": [], "top": []
    }
    tests = (
        ("left", 0, 0.0), ("right", 0, 1.0),
        ("front", 1, 0.0), ("back", 1, 1.0),
        ("bottom", 2, 0.0), ("top", 2, 1.0),
    )
    for name, axis, target in tests:
        boundary_groups[name] = np.flatnonzero(np.abs(local[:, axis] - target) <= tol_ratio).tolist()
    return boundary_groups

def build_point_sets(boundary_groups: Dict[str, List[int]], n_nodes: int) -> Dict[str, NDArray[np.int64]]:
    point_sets = {}

    for name, node_ids in boundary_groups.items():

        if len(node_ids) == 0:
            continue

        arr = np.asarray(node_ids, dtype=int)

        if np.any(arr >= n_nodes):
            raise ValueError(f"{name} contains invalid node index")

        point_sets[name] = arr

    return point_sets



def meshit_coordinate_transform(
    nodes,
    angle_deg,
    translate=False,
    xt=None,
    yt=None,
    zt=None,
):
    """
    Rotate mesh coordinates in the XY plane and optionally translate the
    transformed mesh to a specified location.

    The rotation is performed about the horizontal center of the mesh,
    calculated from the X and Y coordinate bounds. Before rotation, this
    center is shifted to (0, 0). The Z coordinates are not affected by
    the rotation.

    When ``translate=False``, only the XY rotation is applied and the
    original Z coordinates are preserved.

    When ``translate=True``, the rotated mesh can additionally be
    positioned using ``xt``, ``yt``, and ``zt``. The X and Y coordinates
    are shifted by ``xt`` and ``yt``. In the Z direction, the mesh is
    centered using its Z bounds and then shifted to ``zt``.

    If translation is enabled but ``xt``, ``yt``, and ``zt`` are omitted,
    they default to zero. In this case, the transformed mesh is centered
    at (0, 0, 0). If any translation coordinate is provided, all three
    coordinates must be provided.

    Parameters
    ----------
    nodes : array_like
        Mesh node coordinates with shape (N, 3), where the columns
        represent X, Y, and Z coordinates.

    angle_deg : float
        Counterclockwise rotation angle in degrees about the Z axis.

    translate : bool, optional
        If False, apply rotation only and preserve the original Z
        coordinates. If True, additionally reposition the transformed
        mesh. Default is False.

    xt : float, optional
        Target X coordinate of the transformed mesh center.
        Used only when ``translate=True``. Default is 0.0 when omitted.

    yt : float, optional
        Target Y coordinate of the transformed mesh center.
        Used only when ``translate=True``. Default is 0.0 when omitted.

    zt : float, optional
        Target Z coordinate of the transformed mesh center.
        Used only when ``translate=True``. Default is 0.0 when omitted.

    Returns
    -------
    numpy.ndarray
        Transformed mesh coordinates with shape (N, 3).

    Raises
    ------
    ValueError
        If only some of ``xt``, ``yt``, and ``zt`` are provided.
        When translation coordinates are specified, all three must
        be provided together.

    Notes
    -----
    For rotation only, the transformation is:

        X = x - xc
        Y = y - yc

        x' = X*cos(theta) - Y*sin(theta)
        y' = X*sin(theta) + Y*cos(theta)
        z' = z

    where (xc, yc) is the center of the mesh XY bounding box.

    For rotation with translation:

        x' = X*cos(theta) - Y*sin(theta) + xt
        y' = X*sin(theta) + Y*cos(theta) + yt
        z' = z - zc + zt

    where (xc, yc, zc) is the center of the mesh bounding box.

    Examples
    --------
    Rotate the mesh by 30 degrees while preserving the original
    Z coordinates:

        transformed = meshit_coordinate_transform(
            nodes,
            angle_deg=30.0
        )

    Rotate the mesh and place its center at (100, 200, 50):

        transformed = meshit_coordinate_transform(
            nodes,
            angle_deg=30.0,
            translate=True,
            xt=100.0,
            yt=200.0,
            zt=50.0
        )
    """
    pts = np.asarray(nodes, dtype=np.float64).copy()

    xc = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    yc = 0.5 * (pts[:, 1].min() + pts[:, 1].max())

    # This reproduces the ORIGINAL rotation behavior
    X = pts[:, 0] - xc
    Y = pts[:, 1] - yc

    theta = np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)

    transformed = pts.copy()

    transformed[:, 0] = X * c - Y * s
    transformed[:, 1] = X * s + Y * c

    if translate:
        if any(v is not None for v in (xt, yt, zt)) and \
           not all(v is not None for v in (xt, yt, zt)):
            raise ValueError(
                "xt, yt, and zt must be provided together."
            )

        tx = 0.0 if xt is None else float(xt)
        ty = 0.0 if yt is None else float(yt)
        tz = 0.0 if zt is None else float(zt)

        zc = 0.5 * (pts[:, 2].min() + pts[:, 2].max())

        transformed[:, 0] += tx
        transformed[:, 1] += ty
        transformed[:, 2] = pts[:, 2] - zc + tz

    else:
        # EXACTLY like original code
        transformed[:, 2] = pts[:, 2]

    return transformed


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
    geomodel_result: Optional[StructuralModelResults] = None,
    surface_csv_dir: Optional[str] = None,
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
    extent_z: Optional[Tuple[float, float]] = None,
    smooth: float = 1e-5,
    gmsh_flag: bool = False,
    mapping_litho: LithoMappingModeType = LithoMappingMode.AUTOMATIC_CENTERS.value,
    merge_file: Optional[str] = None,
    refinement: Optional[Refinement] = None,
    rotation_angle: Optional[float] = None,
    rotation_angle_translate: Optional[float] = None,
    xt: Optional[float] = None,
    yt: Optional[float] = None,
    zt: Optional[float] = None,

) -> MeshResults:
    """
    Generates an unstructured geological mesh using a geomodel and additional structures
    such as wells, sources, shafts, and extra planes. It performs surface cleaning,
    fragmentation, and meshing using GMSH and returns the final MeshData object.

    Args:
        geomodel_result (StructuralModelResults, optional): Structural-model input. Leave None when using surface_csv_dir.\n        surface_csv_dir (str, optional): Directory of CSV geological surfaces with x,y,z columns. Files whose names start with 'fault' (for example fault1.csv) are treated as reference/fault surfaces. CSV mode bypasses structural-model surface cleaning/removal/extension and uses mapping_litho='none'.\n        rotation_angle (float, optional): Rotation angle in degrees in the XY plane. The mesh center in X and Y is shifted to (0, 0) before rotation; the original Z coordinates are preserved.
        rotation_angle_translate (float, optional): Rotation angle in degrees in the XY plane with additional translation. The mesh center is used as the reference point. If xt/yt/zt are omitted, the transformed mesh is centered at (0, 0, 0).
        xt, yt, zt (float, optional): Target coordinates of the transformed mesh center, used only with rotation_angle_translate. If specified, all three values must be provided together.
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
        extent: Optional domain. Either [xmin, xmax, ymin, ymax, zmin, zmax] for the legacy axis-aligned box, or 8 ordered (x, y, z) corners for an oriented box. If omitted, structural-model mode uses the structural-model extent and CSV-surface mode uses the combined min/max x, y, z values of all CSV surfaces.
        extent_z: Optional absolute Z limits [z_min, z_max] applied only when extent is omitted. For example [-9, 5] replaces the inferred Z limits with zmin=-9 and zmax=5 while preserving the inferred X/Y limits. An explicit extent always takes precedence.
        smooth (float): smoothness factor for interpolation of surfaces
        gmsh_flag (bool):  If True, enables writing the GMSH original mesh to the disk (default=False)
        mapping_litho (str): Controls lithological processing mode.
            Options:
                - "automatic_centers" : apply lithological mapping from structural models to the
                  mesh, voting on each raw block's cell centroids (default -- generally more
                  reliable near a fault plane than "automatic_corners", since cell centroids
                  are guaranteed-interior points, unlike nodes which sit on block boundaries)
                - "automatic_corners" : same as "automatic_centers" but votes on each raw
                  block's node coordinates instead; kept for comparison/backwards compatibility
                - "none"  : skip lithological mapping
                - "manual"   : merge lithology blocks using information provided by user
        merge_file (str, optional): Directory path containing merge definition file.
            Used only when mapping_litho="manual".
        refinement (Refinement, optional): Adaptive mesh refinement settings. Supports local refinement around wells, sources, faults, ellipses,
            and triangulated surfaces through GMSH background mesh fields.
        Rotation_angle (float, optional): Angle in degree to rotate the mesh around the z-axis. If None, no rotation is applied.

    Returns:
        MeshResults: An instance of the MeshResults class.
    """

    # normalize
    if isinstance(mapping_litho, str):
        try:
            mapping_litho = LithoMappingMode(mapping_litho.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mapping_litho='{mapping_litho}'. "
                f"Valid options are: {[m.value for m in LithoMappingMode]}"
            )

    # Input route validation.
    csv_mode = surface_csv_dir is not None
    if csv_mode and geomodel_result is not None:
        raise ValueError(
            "Use either geomodel_result or surface_csv_dir, not both."
        )
    if not csv_mode and geomodel_result is None:
        raise ValueError(
            "Either geomodel_result or surface_csv_dir must be supplied."
        )

    # CSV surfaces do not contain a structural lithology grid, so lithology
    # mapping is intentionally disabled for this input route.
    if csv_mode:
        if mapping_litho != LithoMappingMode.NONE:
            logger.info(
                "surface_csv_dir mode forces mapping_litho='none' "
                "(requested %r).", mapping_litho.value
            )
        mapping_litho = LithoMappingMode.NONE
        if merge_file is not None:
            raise ValueError(
                "merge_file cannot be used with surface_csv_dir because "
                "CSV-surface mode uses mapping_litho='none'."
            )

    # enforce merge_file rules
    if mapping_litho == LithoMappingMode.MANUAL:
        if merge_file is None:
            raise ValueError(
                "manual requires a csv file in which block ids are listed"
            )

    elif mapping_litho == LithoMappingMode.NONE:
        if merge_file is not None:
            raise ValueError(
                "none does not require merge_file"
            )

    elif mapping_litho == LithoMappingMode.AUTOMATIC_CENTERS:
        if merge_file is not None:
            raise ValueError(
                "automatic_centers does not require merge_file"
            )

    elif mapping_litho == LithoMappingMode.AUTOMATIC_CORNERS:
        if merge_file is not None:
            raise ValueError(
                "automatic_corners does not require merge_file"
            )


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
            mine_shafts.append(
                {
                    "center": (cx, cy, cz),
                    "axis": (ax, ay, az),
                    "radius": r,
                }
            )
    else:
        mine_shafts = []

    ########################
    # extract extent of model
    ########################
    if extent_z is not None:
        if len(extent_z) != 2:
            raise ValueError(
                "extent_z must contain exactly two values: [z_min, z_max]."
            )
        extent_z_min, extent_z_max = (float(v) for v in extent_z)
        if not np.isfinite(extent_z_min) or not np.isfinite(extent_z_max):
            raise ValueError("extent_z values must be finite.")
        if extent_z_min >= extent_z_max:
            raise ValueError("extent_z requires z_min < z_max.")
    else:
        extent_z_min = extent_z_max = None

    # ``extent`` is backward compatible:
    #   * 6 values -> legacy axis-aligned box
    #   * 8 (x,y,z) corners -> oriented parallelepiped
    if extent is None or len(extent) == 0:
        if csv_mode:
            # Same default pattern as structural-model mode: infer the domain
            # from the geological input when no explicit extent is supplied.
            # For CSV surfaces this is the combined x/y/z min/max of all files.
            csv_root = Path(surface_csv_dir).expanduser()
            if not csv_root.is_dir():
                raise ValueError(
                    f"surface_csv_dir is not a directory: {surface_csv_dir}"
                )

            csv_files = sorted(
                p for p in csv_root.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".csv"
                and not p.name.startswith(".")
                and not p.name.startswith("._")
            )
            if not csv_files:
                raise ValueError(
                    f"No CSV surface files found in {surface_csv_dir}"
                )

            xyz_min = np.array([np.inf, np.inf, np.inf], dtype=float)
            xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
            n_valid_points = 0

            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                lower_to_original = {
                    str(c).strip().lower(): c for c in df.columns
                }
                missing = [
                    axis for axis in ("x", "y", "z")
                    if axis not in lower_to_original
                ]
                if missing:
                    raise ValueError(
                        f"{csv_path.name} must contain x, y, z columns; "
                        f"missing {missing}."
                    )

                xyz = df[
                    [
                        lower_to_original["x"],
                        lower_to_original["y"],
                        lower_to_original["z"],
                    ]
                ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

                xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
                if xyz.size == 0:
                    continue

                xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
                xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
                n_valid_points += xyz.shape[0]

            if n_valid_points == 0:
                raise ValueError(
                    "No finite x, y, z surface coordinates were found in "
                    f"{surface_csv_dir}."
                )

            domain_spec = np.array(
                [
                    xyz_min[0], xyz_max[0],
                    xyz_min[1], xyz_max[1],
                    xyz_min[2], xyz_max[2],
                ],
                dtype=float,
            )
        else:
            domain_spec = np.asarray(
                geomodel_result.structural_frame.grid.extent, dtype=float
            )

        if extent_z is not None:
            domain_spec = np.asarray(domain_spec, dtype=float).copy()
            domain_spec[4] = extent_z_min
            domain_spec[5] = extent_z_max
    else:
        # An explicitly supplied extent has priority and is never enlarged.
        domain_spec = np.asarray(extent, dtype=float)
    normalize_domain(domain_spec)
    # Some legacy surface interpolation/import code still needs an axis-aligned
    # bounding range.  This AABB is NOT used to create the final 8-corner volume.
    extent_arr = domain_aabb(domain_spec)

    #################################################
    # check if engineering objects are inside extent
    #################################################
    validate_wells(wells, domain_spec)
    validate_sources(sources, domain_spec)
    validate_shafts(mine_shafts, domain_spec)
    validate_planes(extra_planes, domain_spec)
    validate_ellipses(ellipses, domain_spec)

    if len(triangulations) > 0:
        triangulations = np.asarray(triangulations, dtype=np.float64)
        validate_triangulation(triangulations, domain_spec)

    # Initialize GMSH
    try:
        gmsh.initialize()

        ############################################
        # Get data from structural model and clean it
        ############################################
        cleaned_surfaces: List[Tuple[str | int, NDArray[np.float64]]]
        ref_surface_indices: Dict[str | int, int]
        grid_litho: pd.DataFrame

        if csv_mode:
            # CSV mode deliberately bypasses data_preparation(): no overlap
            # removal, fault-side cleaning, projection, or surface extension.
            csv_root = Path(surface_csv_dir).expanduser()
            if not csv_root.is_dir():
                raise ValueError(
                    f"surface_csv_dir is not a directory: {surface_csv_dir}"
                )

            csv_files = sorted(
                p for p in csv_root.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".csv"
                and not p.name.startswith(".")
                and not p.name.startswith("._")
            )
            if not csv_files:
                raise ValueError(
                    f"No CSV surface files found in {surface_csv_dir}"
                )

            cleaned_surfaces = []
            ref_surface_indices = {}

            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                lower_to_original = {
                    str(c).strip().lower(): c for c in df.columns
                }
                missing = [
                    axis for axis in ("x", "y", "z")
                    if axis not in lower_to_original
                ]
                if missing:
                    raise ValueError(
                        f"{csv_path.name} must contain x, y, z columns; "
                        f"missing {missing}."
                    )

                xyz = df[
                    [
                        lower_to_original["x"],
                        lower_to_original["y"],
                        lower_to_original["z"],
                    ]
                ].to_numpy(dtype=float)

                finite = np.all(np.isfinite(xyz), axis=1)
                xyz = xyz[finite]
                if len(xyz) < 3:
                    raise ValueError(
                        f"{csv_path.name} has fewer than 3 valid x,y,z points."
                    )

                # IMPORTANT: create_surface_grid() expects exactly
                # (surface_id, points), not extra metadata.
                surface_id = csv_path.name
                surface_index = len(cleaned_surfaces)
                cleaned_surfaces.append((surface_id, xyz))

                # Match structural-model data_preparation(): reference/fault
                # surfaces are stored separately as name -> cleaned-surface index.
                if csv_path.stem.lower().startswith("fault"):
                    ref_surface_indices[surface_id] = surface_index
                    logger.info(
                        "CSV fault reference surface %s added at index %d",
                        surface_id, surface_index,
                    )

            # No structural lithology exists in CSV mode.
            grid_litho = pd.DataFrame(
                columns=["x", "y", "z", "lithology"]
            )

            logger.info(
                "Loaded %d CSV surfaces (%d fault/reference surfaces): %s",
                len(cleaned_surfaces),
                len(ref_surface_indices),
                list(ref_surface_indices.keys()),
            )
        else:
            cleaned_surfaces, ref_surface_indices, grid_litho = data_preparation(
                geomodel_result,
                DISTANCE_THRESHOLD=DISTANCE_THRESHOLD,
                PROJECTION_THRESHOLD=PROJECTION_THRESHOLD,
                EXTRUSION_FACTOR=EXTRUSION_FACTOR,
                z_threshold=z_threshold,
            )

        ########################
        # Surface interpolation
        ########################
        interpolated_s: List[NDArray[np.float64]] = create_surface_grid(cleaned_surfaces, smooth=smooth,)

        ##################
        # Import surfaces
        ##################
        surfaces_original: List[int]
        bounds: Tuple[float, float, float, float, float, float]

        surfaces_original, bounds = import_surfaces(
            interpolated_s,
            extent_arr,
            tolerance=tolerance,
        )

        gmsh.model.occ.synchronize()

        #################
        # Fragmentation
        #################
        surfaces = surfaces_original.copy()

        ov: List[Tuple[int, int]]
        well_tags: List[int]
        shaft_tags: List[int]
        tri_group_tags: List[int]
        tri_surface_tags: List[int]
        source_tag: List[int]
        boundary_tags: List[int]

        fragment_domain = bounds if normalize_domain(domain_spec)[0] == "extent" else domain_spec
        ov, well_tags, shaft_tags, tri_group_tags, tri_surface_tags, source_tag, boundary_tags = fragment_surfaces(surfaces,
            fragment_domain, ref_surface_indices, wells, extra_planes, sources, mine_shafts, ellipses=ellipses,
            triangulations=triangulations, mesh_size=mesh_size, curve_mesh_size=curve_mesh_size,refinement=refinement,
        )

        #################
        # Mesh generation
        #################
        nodes: NDArray[np.float64]
        cells: List[meshio.CellBlock]

        nodes, cells, cell_data = mesh_generator(ov, extent_arr, well_tags, source_tag, shaft_tags, tri_group_tags,
            tri_surface_tags, grid_litho, mesh_size=mesh_size, curve_mesh_size=curve_mesh_size,
            boundary_tags=boundary_tags, gmsh_flag=gmsh_flag, mapping_litho=mapping_litho, merge_file=merge_file,
        )
        # 1. Original WBGeo/Gmsh coordinates
        x_min, y_min, z_min = nodes.min(axis=0)
        x_max, y_max, z_max = nodes.max(axis=0)
        nodes_global = nodes.copy()
        extent_mesh = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

        # 2. Identify physical boundaries using original geometry
        boundary_domain = extent_mesh if normalize_domain(domain_spec)[0] == "extent" else domain_spec
        boundary_groups = classify_boundary_nodes(nodes_global, boundary_domain)
        point_sets = build_point_sets(boundary_groups, len(nodes_global))

        # 3. Transform ALL node coordinates to the new coordinate system

        if rotation_angle is not None and rotation_angle_translate is not None:
            raise ValueError(
                "Use either rotation_angle (pure rotation) or "
                "rotation_angle_translate (rotation plus centre translation), not both."
            )

        if rotation_angle is not None:
            if any(v is not None for v in (xt, yt, zt)):
                raise ValueError(
                    "xt/yt/zt are only valid with "
                    "rotation_angle_translate."
                )
            nodes_local = meshit_coordinate_transform(
                nodes_global,
                angle_deg=rotation_angle,
                translate=False,
            )

        elif rotation_angle_translate is not None:
            nodes_local = meshit_coordinate_transform(
                nodes_global,
                angle_deg=rotation_angle_translate,
                translate=True,
                xt=xt,
                yt=yt,
                zt=zt,
            )

        elif any(v is not None for v in (xt, yt, zt)):
            raise ValueError(
                "xt/yt/zt require rotation_angle_translate."
            )

    finally:
        gmsh.finalize()




    return MeshResults(
        elements=cells,
        nodes=nodes_local if 'nodes_local' in locals() else nodes,
        point_sets=point_sets,
        cell_data=cell_data,
        mesh_type=MeshType.UNSTRUCTURED,
)
