import logging
import typing
import re
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
from core.meshing_components.explicit.unstructured.refinement_mesh import (Refinement, LinearWellRefinement, FunctionWellRefinement,
                                                               LinearSourceRefinement,FunctionSourceRefinement,)
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from enum import Enum, StrEnum


MESH_ENGINEERING_COLOR = '#99b3cc'
MESH_ENGINEERING_GROUP = 'Engineering Objects'

class LithoMappingMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    MANUAL = "manual"
    # Experimental: same per-block lithology assignment as AUTO, but voting
    # on each raw block's individual *cell centroids* against grid_litho
    # instead of its *node* coordinates. Nodes sit on block boundaries --
    # exactly where a fault plane tends to be -- so a fault-adjacent
    # block's node vote can be genuinely ambiguous regardless of how good
    # the voting algorithm is; cell centroids are guaranteed-interior
    # points, so they don't have that failure mode. Purely additive/opt-in
    # -- does not change AUTO's existing behavior.
    #
    # Verified on model2 (deliberately mismatched resolution: structural
    # grid voxel ~3x the mesh_size, to stress-test this): overall
    # cell-level accuracy against ground truth was 73.3% for
    # AUTOMATIC_DEV vs. 68.1% for AUTO (~16% relative reduction in
    # misclassified cells). Not a clean win on every lithology individually
    # (one lithology was actually more accurate under AUTO), but better
    # overall, and produced fewer/less-severe low-confidence blocks (1
    # warning at 68% vs. 2 warnings at 59% each). At matched
    # structural-grid/mesh resolution, both methods were equivalent
    # (clean 5-blocks-for-5-lithologies either way) -- the accuracy gap
    # specifically shows up when the structural grid is coarse relative to
    # the mesh, which neither method can fully fix (see
    # _check_lithology_mapping_reliability, which warns about this
    # directly). See project_meshing_lithology_mapping memory for the full
    # investigation.
    AUTOMATIC_DEV = "automatic_dev"

LithoMappingModeType = typing.Annotated[
    LithoMappingMode,
    AnnotatedScriptType(
        name="LithoMappingMode",
        identifier="LithoMappingModeType",
        controlled="Select|auto|none|manual|automatic_dev"
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
    Log warnings for signs that mapping_litho="auto"/"automatic_dev"'s
    per-block lithology mapping may be unreliable for this run. Both modes
    are a geometric nearest-neighbor vote against the structural model's
    grid_litho, which degrades when that grid is coarse relative to the
    mesh -- each structural grid point then "owns" a large physical
    region, so many mesh cells/nodes snap to the same nearest point
    regardless of which side of a real boundary (e.g. a fault) they're
    actually on. Verified empirically (model2, deliberately mismatched
    resolution -- structural grid voxel ~3x the mesh_size): overall
    cell-level accuracy against ground truth dropped to ~68-73%, with one
    lithology as low as ~33% -- vs. near-perfect at matched resolution.
    None of these checks can fix a resolution mismatch, only flag it.
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

    def assert_center_inside( point: Tuple[float, float, float], shaft_id: int,) -> None:
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


        #extent check (center only)
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
        ValueError: if any ellipse fails validation
    """

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

    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")

    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, 3) or (n_points, 4)")

    if points.shape[1] not in (3, 4):
        raise ValueError("points must have 3 or 4 columns (x,y,z[,id])")


    xmin, xmax, ymin, ymax, zmin, zmax = extent

    xyz = points[:, :3]  # only spatial coordinates

    mask = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )

    outside_points = points[~mask]

    if outside_points.size > 0:

        msg = (
            f"{outside_points.shape[0]} triangulation points lie outside the extent.\n"
            f"Extent: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}, "
            f"zmin={zmin}, zmax={zmax}\n"
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
    boundary_tags: Optional[List[int]] = None, gmsh_flag: bool = False, mapping_litho: LithoMappingModeType = LithoMappingMode.AUTO.value, merge_file: Optional[str] = None) -> MeshResults:

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
                - "auto"     : apply lithological mapping from structural models to the mesh (default)
                - "none"  : skip lithological mapping
                - "manual"   : merge lithology blocks using information provided by user
                - "automatic_dev" : experimental -- same as "auto" but votes on each
                  raw block's cell centroids instead of its node coordinates, which
                  are more reliable near a fault plane (nodes sit on block
                  boundaries, exactly where a fault tends to be; cell centroids are
                  guaranteed-interior points)
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

  # Populated by the AUTO/AUTOMATIC_DEV branches below (by object identity,
  # not index -- new_cells gets more non-lithology blocks appended to it
  # after this point: shafts, wells, sources, fault surfaces, triangulated
  # surfaces, boundary surfaces). Used right before this function returns
  # to build cell_data["block_id"], so mesh_generator's caller can trust
  # the lithology mapping directly instead of having to re-derive it (see
  # project_meshing_lithology_mapping memory). MANUAL/NONE don't populate
  # this -- MANUAL groups by user-supplied indices with no inherent
  # lithology-id semantics, and NONE explicitly skips lithology mapping.
  litho_block_id_by_identity: Dict[int, int] = {}

  if mapping_litho == LithoMappingMode.AUTO:

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
        grid_litho_values, grid_coords, mesh_size, "auto",
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


  elif mapping_litho == LithoMappingMode.AUTOMATIC_DEV:

    # Same overall structure as AUTO (per-block vote -> merge by lithology
    # -> sort by depth), but the vote itself is taken on each block's
    # individual *cell centroids* (guaranteed-interior points) instead of
    # its *node* coordinates (which sit on block boundaries -- exactly
    # where a fault plane tends to be, and can give a genuinely ambiguous
    # nearest-neighbor vote no matter how good the algorithm is).
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
        grid_litho_values, grid_coords, mesh_size, "automatic_dev",
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
    # cell_data["block_id"] -- see the matching comment in the AUTO branch.
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


  # Propagate the real per-block lithology id (AUTO/AUTOMATIC_DEV only --
  # see litho_block_id_by_identity above) out to cell_data["block_id"],
  # matching the convention create_structured_mesh_data/
  # create_implicit_structured_mesh already use, so callers can trust the
  # mapping directly instead of re-deriving it (see
  # project_meshing_lithology_mapping memory). Sentinel -1 for every block
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

    # format of the csv is: id, x, y, z\n
    named_source_data: Mapping[str, List[Tuple[float]]] = {}

    # load csv file
    with open(source_file, "r") as f:

        for line in f.readlines():

            if line.startswith("#"):
                continue

            parts = [s.strip() for s in line.split(",")]

            # CHECK NUMBER OF COLUMNS FIRST
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid source line (expected 4 columns: id,x,y,z): {line}"
                )

            source_id, source_x, source_y, source_z = parts

            # check empty values (for instance x, y or z is missing)
            if not all([source_x, source_y, source_z]):
                raise ValueError(
                    f"(missing x, y, or z value): {line.strip()}"
                )

            # check for non-numeric values
            try:
                values = [
                    float(source_x),
                    float(source_y),
                    float(source_z),
                ]
            except ValueError:
                raise ValueError(
                    f"Non-numeric value in source definition: {line.strip()}"
                )

            if source_id not in named_source_data:
                named_source_data[source_id] = []

            named_source_data[source_id].append(values)


    # ensure that we have only one point per source
    if any(len(source) > 1 for source in named_source_data.values()):
        incorrect_sources = [
            source_id
            for source_id, source_data in named_source_data.items()
            if len(source_data) > 1
        ]

        raise ValueError(
            f"Some source(s) {incorrect_sources} has more than one coordinates"
        )

    # format right now is {key: [(x,y,z)]}
    # -> map it to [x1, y1, z1] for each source
    return [
        tuple(coord for source_group in source_data for coord in source_group)
        for source_data in named_source_data.values()
    ]


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
                raise ValueError(
                    f"Line {line_number}: too few columns ({len(parts)})."
                )

            # -------------------
            # Required fields
            # -------------------
            try:
                cx = float(parts[0])
                cy = float(parts[1])
                cz = float(parts[2])
                r1 = float(parts[3])
                r2 = float(parts[4])

            except ValueError:
                raise ValueError(
                    f"Line {line_number}: invalid numeric value in required fields"
                )

            ellipse = {
                "center": (cx, cy, cz),
                "radii": (r1, r2),
            }

            index = 5

            # -------------------
            # Optional angles
            # -------------------
            if len(parts) >= index + 2:
                angle1 = float(parts[index]) if parts[index] else 0.0
                angle2 = float(parts[index + 1]) if parts[index + 1] else 2 * np.pi

                ellipse["angle1"] = angle1
                ellipse["angle2"] = angle2

                index += 2
            else:
                ellipse["angle1"] = 0.0
                ellipse["angle2"] = 2 * np.pi

            # -------------------
            # Optional zAxis
            # -------------------
            if len(parts) >= index + 3:
                try:
                    zAxis = [
                        float(parts[index]),
                        float(parts[index + 1]),
                        float(parts[index + 2]),
                    ]
                except ValueError:
                    raise ValueError(
                        f"Line {line_number}: zAxis must contain 3 numeric values"
                    )

                if not all(np.isfinite(zAxis)):
                    raise ValueError(
                        f"Line {line_number}: zAxis must be 3 finite numbers"
                    )

                ellipse["zAxis"] = zAxis
                index += 3
            else:
                ellipse["zAxis"] = [0.0, 0.0, 1.0]

            # -------------------
            # Optional xAxis
            # -------------------
            if len(parts) >= index + 3:
                try:
                    xAxis = [
                        float(parts[index]),
                        float(parts[index + 1]),
                        float(parts[index + 2]),
                    ]
                except ValueError:
                    raise ValueError(
                        f"Line {line_number}: xAxis must contain 3 numeric values"
                    )

                if not all(np.isfinite(xAxis)):
                    raise ValueError(
                        f"Line {line_number}: xAxis must be 3 finite numbers"
                    )

                ellipse["xAxis"] = xAxis

            ellipses.append(ellipse)

    return ellipses


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

    with open(shaft_file, "r") as f:

        for line in f.readlines():

            if line.startswith("#"):
                continue

            parts = [s.strip() for s in line.split(",")]

            if len(parts) != 8:
                raise ValueError(
                    f"Invalid shaft line (expected 8 columns): {line.strip()}"
                )

            shaft_id, cx, cy, cz, ax, ay, az, rad = parts

            # check missing values
            numeric_fields = {
                "cx": cx,
                "cy": cy,
                "cz": cz,
                "ax": ax,
                "ay": ay,
                "az": az,
                "radius": rad,
            }

            missing = [k for k, v in numeric_fields.items() if v == ""]

            if missing:
                raise ValueError(
                    f"Missing value(s) {missing} in shaft definition"
                )

            # check numeric values
            try:
                values = [
                    float(cx),
                    float(cy),
                    float(cz),
                    float(ax),
                    float(ay),
                    float(az),
                    float(rad),
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
                 color=MESH_ENGINEERING_COLOR,  # the color of the components
                 border_color='#000000',  # and its border color
                 group=MESH_ENGINEERING_GROUP,
                 identifier='wbgeo::meshing_load_plane_from_csv',  # a unique identifier
                 return_name='planes',  # the name for the returned-port
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

    # format of the csv is: id, x, y, z\n
    named_plane_data: Mapping[str, List[Tuple[float]]] = {}

    # load csv file
    with open(plane_file, "r") as f:

        for line in f.readlines():

            if line.startswith("#"):
                continue

            parts = [s.strip() for s in line.split(",")]

            # CHECK NUMBER OF COLUMNS FIRST
            if len(parts) != 4:
                logger.debug("Unexpected parts count %d: %s", len(parts), parts)
                raise ValueError(
                    f"Invalid plane line (expected 4 columns: id,x,y,z): {line}"
                )

            plane_id, plane_x, plane_y, plane_z = parts

            # check empty values (for instance x, y or z is missing)
            if not all([plane_x, plane_y, plane_z]):
                raise ValueError(
                    f"(missing x, y, or z value): {line.strip()}"
                )

            # check for non-numeric values
            try:
                values = [
                    float(plane_x),
                    float(plane_y),
                    float(plane_z),
                ]
            except ValueError:
                raise ValueError(
                    f"Non-numeric value in plane definition: {line.strip()}"
                )

            if plane_id not in named_plane_data:
                named_plane_data[plane_id] = []

            named_plane_data[plane_id].append(values)


    # ensure that we have at least 4 points per plane
    if any(len(plane) < 4 for plane in named_plane_data.values()):
        incorrect_planes = [
            plane_id
            for plane_id, plane_data in named_plane_data.items()
            if len(plane_data) < 4
        ]
        raise ValueError(
            f"Some plane(s) {incorrect_planes} are missing required number of points (minimum 4)"
        )

    # format right now is {key: [(x,y,z)]}
    # -> map it to [x1, y1, z1, ..., xi, yi, zi] for each plane
    return [
        tuple(coord for plane_group in plane_data for coord in plane_group)
        for plane_data in named_plane_data.values()
    ]



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
def load_triangulations_planes_from_csv(csv_file: TriangulationsPlanesData,) -> TriangulationData:
    """
    Load plane coordinates from one or multiple CSV files for triangulations.

    Each CSV must have columns: x, y, z. Lines starting with '#' are ignored.

    If multiple files are provided, an additional column 'id' is added:
    - first file -> id = 1
    - second file -> id = 2
    etc.

    Returns:
        np.ndarray: Array of shape (N, 4) -> [x, y, z, id]
    """

    if isinstance(csv_file, str):
        csv_files = [csv_file]
    else:
        csv_files = csv_file

    all_points = []

    for file_id, file_path in enumerate(csv_files, start=1):

        with open(file_path, "r") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]

                if len(parts) != 3:
                    raise ValueError(
                        f"{file_path} line {line_number}: "
                        f"expected 3 columns (x,y,z), got {len(parts)}"
                    )

                try:
                    x, y, z = map(float, parts)

                except ValueError:
                    raise ValueError(
                        f"{file_path} line {line_number}: "
                        f"non-numeric value found: {line}"
                    )

                all_points.append([x, y, z, file_id])

    if not all_points:
        raise ValueError("No points loaded from the CSV file(s).")

    return np.array(all_points, dtype=np.float64)


def classify_boundary_nodes(nodes: NDArray[np.float64], extent: List[float], tol_ratio: float = 1e-4) -> Dict[str, List[int]]:
    nodes = np.asarray(nodes)

    xmin, xmax, ymin, ymax, zmin, zmax = extent

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    tolx = dx * tol_ratio
    toly = dy * tol_ratio
    tolz = dz * tol_ratio

    boundary_groups = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
        "bottom": [],
        "top": [],
    }

    for i, (x, y, z) in enumerate(nodes):

        if abs(x - xmin) <= tolx:
            boundary_groups["left"].append(i)

        if abs(x - xmax) <= tolx:
            boundary_groups["right"].append(i)

        if abs(y - ymin) <= toly:
            boundary_groups["front"].append(i)

        if abs(y - ymax) <= toly:
            boundary_groups["back"].append(i)

        if abs(z - zmin) <= tolz:
            boundary_groups["bottom"].append(i)

        if abs(z - zmax) <= tolz:
            boundary_groups["top"].append(i)

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
    smooth: float = 1e-5,
    gmsh_flag: bool = False,
    mapping_litho: LithoMappingModeType = LithoMappingMode.AUTO.value,
    merge_file: Optional[str] = None,
    refinement: Optional[Refinement] = None,
) -> MeshResults:
    """
    Generates an unstructured geological mesh using a geomodel and additional structures
    such as wells, sources, shafts, and extra planes. It performs surface cleaning,
    fragmentation, and meshing using GMSH and returns the final MeshData object.

    Args:
        geomodel_result (StructuralModelResults): Result of a structural geological model, containing a StructuralFrame with the interpolated lithology block, grid, and surface meshes.
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
        gmsh_flag (bool):  If True, enables writing the GMSH original mesh to the disk (default=False)
        mapping_litho (str): Controls lithological processing mode.
            Options:
                - "auto"     : apply lithological mapping from structural models to the mesh (default)
                - "none"  : skip lithological mapping
                - "manual"   : merge lithology blocks using information provided by user
                - "automatic_dev" : experimental -- same as "auto" but votes on each
                  raw block's cell centroids instead of its node coordinates, which
                  are more reliable near a fault plane (nodes sit on block
                  boundaries, exactly where a fault tends to be; cell centroids are
                  guaranteed-interior points)
        merge_file (str, optional): Directory path containing merge definition file.
            Used only when mapping_litho="manual".
        refinement (Refinement, optional): Adaptive mesh refinement settings. Supports local refinement around wells, sources, faults, ellipses,
            and triangulated surfaces through GMSH background mesh fields.

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

    elif mapping_litho == LithoMappingMode.AUTO:
        if merge_file is not None:
            raise ValueError(
                "auto does not require merge_file"
            )

    elif mapping_litho == LithoMappingMode.AUTOMATIC_DEV:
        if merge_file is not None:
            raise ValueError(
                "automatic_dev does not require merge_file"
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
    extent_arr: NDArray[np.float64]

    if extent is None or len(extent) == 0:
        extent_arr = np.asarray(
            geomodel_result.structural_frame.grid.extent,
            dtype=float,
        )
    else:
        extent_arr = np.asarray(extent, dtype=float)

    #################################################
    # check if engineering objects are inside extent
    #################################################
    validate_wells(wells, extent_arr)
    validate_sources(sources, extent_arr)
    validate_shafts(mine_shafts, extent_arr)
    validate_planes(extra_planes, extent_arr)
    validate_ellipses(ellipses, extent_arr)

    if len(triangulations) > 0:
        triangulations = np.asarray(triangulations, dtype=np.float64)
        validate_triangulation(triangulations, extent_arr)

    # Initialize GMSH
    try:
        gmsh.initialize()

        ############################################
        # Get data from structural model and clean it
        ############################################
        cleaned_surfaces: List[NDArray[np.float64]]
        ref_surface_indices: Dict[int, int]
        grid_litho: pd.DataFrame

        cleaned_surfaces, ref_surface_indices, grid_litho = data_preparation(geomodel_result, DISTANCE_THRESHOLD=DISTANCE_THRESHOLD,
            PROJECTION_THRESHOLD=PROJECTION_THRESHOLD, EXTRUSION_FACTOR=EXTRUSION_FACTOR, z_threshold=z_threshold,
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

        ov, well_tags, shaft_tags, tri_group_tags, tri_surface_tags, source_tag, boundary_tags = fragment_surfaces(surfaces,
            bounds, ref_surface_indices, wells, extra_planes, sources, mine_shafts, ellipses=ellipses,
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

    finally:
        gmsh.finalize()


    ############
    # BC nodes
    ############
    x_min, y_min, z_min = nodes.min(axis=0)
    x_max, y_max, z_max = nodes.max(axis=0)

    extent_mesh = np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    boundary_groups_n = classify_boundary_nodes(nodes, extent_mesh)
    point_sets = build_point_sets(boundary_groups_n, len(nodes))


    return MeshResults(
        elements=cells,
        nodes=nodes,
        point_sets=point_sets,
        cell_data=cell_data,
        mesh_type=MeshType.UNSTRUCTURED,
)
