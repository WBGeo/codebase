"""
Geological and fault surfaces for unstructured meshing, taken directly from the
``extended_masked`` meshes of a structural model.

The structural model stores, per element and fault-domain component, one surface
piece from that component's own scalar field. Each piece is cut at its faults and at
the base of all younger groups, overshoots both by about one grid cell, and reaches
beyond the model box. The pieces therefore need no cleaning: every piece is turned
into one GMSH OCC surface (a plane if it is flat, otherwise one B-spline surface
fitted in the piece's own plane), trimmed to the piece's outline (its convex hull if
it is convex up to the cell-sized steps of the outline), and fragmented
with the model box like any other surface. The parts beyond the faults, above the
unconformities and outside the box end up as free surfaces and are not meshed.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gmsh
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from core.object_components import StructuralModelResults

logger = logging.getLogger(__name__)

EXTENDED_MASKED_MESH_TYPE = "extended_masked"

# A piece whose vertices deviate less than this (in grid cells) from their best-fit
# plane is imported as a plane surface instead of a B-spline surface.
_PLANAR_TOLERANCE_CELLS: float = 0.25

# Spacing of the B-spline control points, in grid cells.
_BSPLINE_STEP_CELLS: float = 2.0

# Pieces smaller than this area (in grid cell faces) are skipped, e.g. tiny marching
# cubes fragments at the edge of the padded grid.
_MIN_PIECE_AREA_CELLS: float = 4.0

# Tolerance (in grid cells) for simplifying a piece outline before trimming. Outlines
# only lie in the overshoot beyond faults, unconformities and the model box, so this
# must stay well below the one-cell overshoot.
_OUTLINE_SIMPLIFY_CELLS: float = 0.25

# A piece without holes whose outline deviates at most this much (in grid cells) from
# its convex hull is trimmed to the hull: the deviation is then only the cell-sized
# steps of the outline. Trimming to the stepped outline imprints many short segments
# onto the neighbouring surfaces near the faults and lowers the mesh quality there.
_HULL_TOLERANCE_CELLS: float = 1.5


@dataclass
class SurfacePiece:
    """
    One connected piece of an element's or fault's extended masked mesh.

    Attributes:
        name: Name of the structural element or fault the piece belongs to.
        is_fault: True for fault pieces.
        vertices: Vertex coordinates, shape (N, 3).
        faces: Triangle vertex indices, shape (M, 3).
    """
    name: str
    is_fault: bool
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


def build_grid_litho(geomodel_result: StructuralModelResults) -> pd.DataFrame:
    """
    Lithology of every structural grid cell, used to assign lithologies to the mesh volumes.

    Args:
        geomodel_result: Structural model result.

    Returns:
        DataFrame with columns ``x``, ``y``, ``z`` (cell centres) and ``lithology``.
    """
    frame = geomodel_result.structural_frame
    # grid_coordinates is (N, 3); lith_block is (nx, ny, nz) -- flatten to (N,).
    return pd.concat([
        pd.DataFrame(frame.grid.grid_coordinates, columns=["x", "y", "z"]),
        pd.DataFrame(frame.lith_block.flatten(), columns=["lithology"]),
    ], axis=1)


def split_connected_pieces(vertices: NDArray[np.float64], faces: NDArray[np.int64]
                           ) -> List[Tuple[NDArray[np.float64], NDArray[np.int64]]]:
    """
    Split a triangle mesh into its connected components (triangles sharing vertices).

    Args:
        vertices: Vertex coordinates, shape (N, 3).
        faces: Triangle vertex indices, shape (M, 3).

    Returns:
        List of (vertices, faces) per component, with re-indexed faces.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
    _, labels = connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]
    pieces = []
    for label in np.unique(face_labels):
        piece_faces = faces[face_labels == label]
        used, new_faces = np.unique(piece_faces, return_inverse=True)
        pieces.append((vertices[used], new_faces.reshape(-1, 3).astype(np.int64)))
    return pieces


def collect_surface_pieces(geomodel_result: StructuralModelResults) -> List[SurfacePiece]:
    """
    Connected pieces of the extended masked meshes of all elements and faults.

    Args:
        geomodel_result: Structural model result computed with extended masked meshes.

    Returns:
        Geological pieces first (youngest group first), then fault pieces.

    Raises:
        ValueError: If an element or fault has no extended masked mesh, i.e. the
            structural model was computed with an older version.
    """
    frame = geomodel_result.structural_frame
    spacing = np.asarray(frame.grid.spacing, dtype=float)
    min_area = _MIN_PIECE_AREA_CELLS * float(np.sort(spacing)[:2].prod())

    sources: List[Tuple[str, bool, object]] = [
        (elem.name, False, elem) for group in frame.structural_groups for elem in group.structural_elements
    ]
    if frame.fault_frame is not None:
        sources += [(fault.name, True, fault) for fault in frame.fault_frame.fault_elements]

    pieces: List[SurfacePiece] = []
    for name, is_fault, obj in sources:
        if EXTENDED_MASKED_MESH_TYPE not in obj.vertices:
            raise ValueError(
                f"'{name}' has no '{EXTENDED_MASKED_MESH_TYPE}' surface mesh. Recompute the structural model "
                f"with the current version, or use surface_source='reconstructed'."
            )
        for vertices, faces in split_connected_pieces(obj.vertices[EXTENDED_MASKED_MESH_TYPE],
                                                      obj.edges[EXTENDED_MASKED_MESH_TYPE]):
            tri = vertices[faces]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
            if area < min_area:
                logger.debug("Skipping a %.1f m2 fragment of '%s'", area, name)
                continue
            pieces.append(SurfacePiece(name=name, is_fault=is_fault, vertices=vertices, faces=faces))
    logger.info("Collected %d surface pieces (%d fault pieces) from the extended masked meshes",
                len(pieces), sum(p.is_fault for p in pieces))
    return pieces


def _principal_frame(points: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centre and orthonormal axes (rows: in-plane u, in-plane v, normal) of a point set."""
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    return centre, axes


def _boundary_loops(faces: NDArray[np.int64], uv: NDArray[np.float64]) -> List[NDArray[np.int64]]:
    """
    Closed boundary loops of a triangle mesh, as vertex index arrays.

    Boundary edges (used by one triangle only) are walked in the triangles'
    orientation. Where a vertex has several outgoing boundary edges (two loops
    touching in one vertex), the walk takes the sharpest turn to keep the loops simple.
    """
    # Walk with the interior on the left: mirror v if the (u, v) frame is flipped
    # relative to the triangle orientation.
    tri = uv[faces]
    if np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum() < 0:
        uv = uv * np.array([1.0, -1.0])

    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.ravel()] == 1]

    outgoing: Dict[int, List[int]] = {}
    for a, b in boundary:
        outgoing.setdefault(int(a), []).append(int(b))

    loops: List[NDArray[np.int64]] = []
    while outgoing:
        start = next(iter(outgoing))
        loop, prev, current = [start], None, start
        while True:
            options = outgoing.get(current)
            if not options:
                break
            if len(options) == 1 or prev is None:
                nxt = options[0]
            else:
                incoming = uv[current] - uv[prev]
                angles = [np.arctan2(np.cross(incoming, uv[o] - uv[current]), np.dot(incoming, uv[o] - uv[current]))
                          for o in options]
                nxt = options[int(np.argmax(angles))]
            options.remove(nxt)
            if not options:
                del outgoing[current]
            if nxt == start:
                break
            loop.append(nxt)
            prev, current = current, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _simplify_closed(points: NDArray[np.float64], tolerance: float) -> NDArray[np.float64]:
    """Douglas-Peucker simplification of a closed 2D polygon (first point not repeated)."""
    if len(points) <= 4:
        return points

    def simplify(chain: NDArray[np.float64]) -> List[int]:
        keep = np.zeros(len(chain), dtype=bool)
        keep[[0, -1]] = True
        stack = [(0, len(chain) - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            seg = chain[j] - chain[i]
            rel = chain[i + 1:j] - chain[i]
            length = np.linalg.norm(seg)
            dist = (np.abs(np.cross(seg, rel)) / length) if length > 0 else np.linalg.norm(rel, axis=1)
            k = int(np.argmax(dist))
            if dist[k] > tolerance:
                keep[i + 1 + k] = True
                stack += [(i, i + 1 + k), (i + 1 + k, j)]
        return list(np.flatnonzero(keep))

    # split the loop at its point farthest from the first one, simplify both chains
    far = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    first = simplify(points[:far + 1])
    second = simplify(np.vstack([points[far:], points[:1]]))
    idx = first + [far + k for k in second[1:-1]]
    return points[idx]


def _signed_area(polygon: NDArray[np.float64]) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _distance_to_polygon(points: NDArray[np.float64], polygon: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance of 2D points to the boundary of a closed polygon."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("pkj,kj->pk", points[:, None, :] - a[None], ab) / np.maximum((ab ** 2).sum(axis=1), 1e-300),
                0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _piece_outline(piece: SurfacePiece, centre: NDArray[np.float64], axes: NDArray[np.float64],
                   tolerance: float, hull_tolerance: float) -> List[NDArray[np.float64]]:
    """
    Trimming loops of a piece in its (u, v) frame, the outer loop first: its convex hull
    if the piece is convex up to ``hull_tolerance`` and has no holes, otherwise its
    outline simplified by ``tolerance``.
    """
    uv = (piece.vertices - centre) @ axes[:2].T
    loops = [_simplify_closed(uv[loop], tolerance) for loop in _boundary_loops(piece.faces, uv)]
    loops = [loop for loop in loops if len(loop) >= 3 and abs(_signed_area(loop)) > 0]
    loops.sort(key=lambda loop: -abs(_signed_area(loop)))
    if len(loops) == 1:
        hull = uv[ConvexHull(uv).vertices]
        if _distance_to_polygon(loops[0], hull).max() <= hull_tolerance:
            return [hull]
    return loops


def _add_wire(points_3d: NDArray[np.float64]) -> int:
    occ = gmsh.model.occ
    tags = [occ.addPoint(*p) for p in points_3d]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    return occ.addCurveLoop(lines)


def _add_plane_piece(loops_uv: List[NDArray[np.float64]], centre: NDArray[np.float64],
                     axes: NDArray[np.float64]) -> List[int]:
    """Plane surface bounded by the piece outline (outer loop + holes) in its best-fit plane."""
    wires = [_add_wire(centre + loop @ axes[:2]) for loop in loops_uv]
    return [gmsh.model.occ.addPlaneSurface(wires)]


def _add_bspline_piece(piece: SurfacePiece, loops_uv: List[NDArray[np.float64]], centre: NDArray[np.float64],
                       axes: NDArray[np.float64], step: float) -> List[int]:
    """
    One B-spline surface fitted to the piece as a height field over its own plane,
    trimmed to the piece outline (outer loop + holes).
    """
    occ = gmsh.model.occ
    uv = (piece.vertices - centre) @ axes[:2].T
    w = (piece.vertices - centre) @ axes[2]

    lo, hi = uv.min(axis=0) - step, uv.max(axis=0) + step
    nu, nv = np.maximum(np.ceil((hi - lo) / step).astype(int) + 1, 4)
    gu, gv = np.meshgrid(np.linspace(lo[0], hi[0], nu), np.linspace(lo[1], hi[1], nv), indexing="ij")
    grid_uv = np.column_stack([gu.ravel(), gv.ravel()])

    # Fit on about one vertex per control cell plus the outline, with a thin-plate spline
    fit_idx = np.unique(cKDTree(uv).query(grid_uv)[1])
    height = RBFInterpolator(uv[fit_idx], w[fit_idx], kernel="thin_plate_spline", smoothing=1e-3 * len(fit_idx),
                             degree=1)
    gw = height(grid_uv)
    residual = float(np.abs(height(uv) - w).max())
    if residual > step:
        logger.warning("Surface piece of '%s' deviates up to %.1f m from its fitted B-spline (folds back over "
                       "its own plane?)", piece.name, residual)

    control = centre + grid_uv @ axes[:2] + gw[:, None] * axes[2]
    point_tags = [occ.addPoint(*p) for p in control]
    # GMSH counts numPointsU along the fastest-varying index (here v).
    surface = occ.addBSplineSurface(point_tags, numPointsU=nv)

    # Trim: intersect with a prism over the outline, along the piece normal
    base = float(gw.min()) - 2 * step
    prism_height = float(np.ptp(gw)) + 4 * step
    wires = [_add_wire(centre + loop @ axes[:2] + base * axes[2]) for loop in loops_uv]
    footprint = occ.addPlaneSurface(wires)
    prism = [e for e in occ.extrude([(2, footprint)], *(prism_height * axes[2])) if e[0] == 3]
    trimmed, _ = occ.intersect([(2, surface)], prism, removeObject=True, removeTool=True)
    return [tag for dim, tag in trimmed if dim == 2]


def import_structural_surfaces(geomodel_result: StructuralModelResults) -> Tuple[List[int], Dict[str, int]]:
    """
    Create one GMSH OCC surface per extended masked surface piece of a structural model.

    Flat pieces become plane surfaces; all others one B-spline surface each, fitted in
    the piece's own plane. Each surface is trimmed to its piece outline, or to the
    piece's convex hull if the outline only deviates from it by cell-sized steps.

    Args:
        geomodel_result: Structural model result computed with extended masked meshes.

    Returns:
        surfaces: GMSH surface tags, geological pieces first, fault pieces last.
        reference_surface_indices: Fault pieces as {label: position in ``surfaces``};
            used by ``fragment_surfaces`` for fault physical groups and fault refinement.
    """
    spacing = np.asarray(geomodel_result.structural_frame.grid.spacing, dtype=float)
    cell = float(spacing.min())
    step = _BSPLINE_STEP_CELLS * float(spacing.max())

    surfaces: List[int] = []
    reference_surface_indices: Dict[str, int] = {}
    n_planar = 0
    for k, piece in enumerate(collect_surface_pieces(geomodel_result)):
        centre, axes = _principal_frame(piece.vertices)
        loops_uv = _piece_outline(piece, centre, axes, _OUTLINE_SIMPLIFY_CELLS * cell, _HULL_TOLERANCE_CELLS * cell)
        if not loops_uv:
            logger.warning("Skipping a surface piece of '%s' without a closed outline", piece.name)
            continue
        flatness = float(np.abs((piece.vertices - centre) @ axes[2]).max())
        if flatness < _PLANAR_TOLERANCE_CELLS * cell:
            tags = _add_plane_piece(loops_uv, centre, axes)
            n_planar += 1
        else:
            tags = _add_bspline_piece(piece, loops_uv, centre, axes, step)
        for tag in tags:
            if piece.is_fault:
                reference_surface_indices[f"{piece.name}#{k}.{tag}"] = len(surfaces)
            surfaces.append(tag)

    gmsh.model.occ.synchronize()
    logger.info("Imported %d surfaces from the extended masked meshes (%d plane pieces, %d fault surfaces)",
                len(surfaces), n_planar, len(reference_surface_indices))
    return surfaces, reference_surface_indices
