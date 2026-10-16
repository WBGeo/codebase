"""
Dual contouring (DC) of fitted scalar fields on the max-level cells of an octree.

The octree counterpart of the marching cubes helpers in
``surface_mesh_extraction.py``. Only cells at ``max_level`` take part: a surface
reaches max level wherever refinement saw it, and
:func:`~core.structural_modeling_components.structural_modeling_utility.octree_refinement.balance_mesh_depth_adjacency`
makes sure the cells around it are at max level as well.

For one isovalue:

  1) Field values at the corners of the cells, each lattice point evaluated once
     per cell group (cells of different groups, e.g. fault domains, never share
     corner values or faces).
  2) One crossing per distinct cell edge whose end values lie on opposite sides
     of the isovalue, placed by linear interpolation along the edge, with the
     field gradient there as its normal (central differences of the field).
     Which side a value equal to the isovalue is on must match the labels the
     octree was refined with (``>=`` for horizons, like the lithology; ``>`` for
     faults, like their masks); otherwise a crossing at a lattice point can fall
     into cells that were never refined.
  3) One vertex per cell with a crossing, from GemPy's QEF: a row per crossing
     (its normal through its point) plus three axis-aligned rows through the
     crossings' centroid, all at unit weight; the vertex is clamped to the cell.
  4) One quad per crossing edge from the 4 cells around it, split into two
     triangles whose normals point toward increasing field values. An edge with
     fewer than 4 such cells gets no quad: at the extent walls and next to cells
     of another group the surface ends there, next to a coarser cell this is a
     gap.

Ported from the flat dual-contouring path of the octree research code (itself a
port of GemPy's), adapted to explicit isovalues, integer cell addresses and cell
groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import numpy.typing as npt

from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    balance_mesh_depth_adjacency,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    CORNER_OFFSETS,
    GapMitigation,
    OctreeCells,
    OctreeGrid,
    linear_cell_index,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Field values at points of known cell groups: (K, 3) coordinates and (K,) group ids -> (K,) values.
GroupFieldValues = Callable[[FloatArray, IntArray], FloatArray]

# The 12 cell edges as (corner a, corner b), a at the lower end along the edge's
# axis. Corner n of a cell is at CORNER_OFFSETS[n] (bit 0 = x, bit 1 = y, bit 2 = z).
_EDGES: npt.NDArray[np.int64] = np.array([
    (0, 1), (2, 3), (4, 5), (6, 7),  # along x
    (0, 2), (1, 3), (4, 6), (5, 7),  # along y
    (0, 4), (1, 5), (2, 6), (3, 7),  # along z
], dtype=np.int64)
_EDGE_AXIS: npt.NDArray[np.int64] = np.repeat(np.arange(3), 4)

# Steps to the 26 cells that share a face, an edge or a corner with a cell.
_NEIGHBOUR_STEPS: npt.NDArray[np.int64] = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
    dtype=np.int64,
)

# Fraction of the finest cell size used as the central-difference step for normals.
_GRADIENT_STEP_CELLS: float = 0.25


@dataclass
class DualContourMesh:
    """
    A triangle mesh from :func:`dual_contour`.

    Attributes:
        vertices: Vertex coordinates, shape ``(V, 3)``.
        faces: Vertex indices of the triangles, shape ``(F, 3)``; normals point
            toward increasing field values.
        vertex_cells: Max-level cell index ``(i, j, k)`` each vertex belongs to,
            shape ``(V, 3)``.
        vertex_at_group_boundary: Whether a vertex's cell touches a cell of
            another group (shares a face, an edge or a corner), shape ``(V,)``.
        n_missing_quads: Crossing edges inside the extent that got no quad
            because a cell around them is not at max level: gaps in the surface.
    """

    vertices: FloatArray
    faces: IntArray
    vertex_cells: IntArray
    vertex_at_group_boundary: npt.NDArray[np.bool_]
    n_missing_quads: int


def contour_cells(grid: OctreeGrid, cells: OctreeCells) -> OctreeCells:
    """
    The max-level cells dual contouring runs on, after the grid's gap mitigation.

    Args:
        grid: The octree grid; its ``gap_mitigation`` decides whether the cells are
            balanced first (:func:`balance_mesh_depth_adjacency`).
        cells: Leaf cells that tile the extent, refined toward the surfaces.

    Returns:
        Only the max-level cells, from a balanced copy if mitigation is on.
        ``cells`` itself is not changed.
    """
    if grid.gap_mitigation == GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY:
        cells = balance_mesh_depth_adjacency(grid, cells)
    fine = cells.level == grid.max_level
    return OctreeCells(level=cells.level[fine], ijk=cells.ijk[fine])


def dual_contour(
        grid: OctreeGrid,
        cell_ijk: npt.NDArray[np.integer],
        cell_groups: npt.NDArray[np.integer],
        values: GroupFieldValues,
        isovalues: Sequence[float],
        strictly_above: bool = False,
) -> List[DualContourMesh]:
    """
    Dual-contour one field at several isovalues on max-level cells.

    Args:
        grid: The octree grid.
        cell_ijk: Max-level cell indices, shape ``(M, 3)``, without duplicates.
        cell_groups: Group of each cell, shape ``(M,)``. Corner values are
            evaluated per group and quads only join cells of one group.
        values: The field; see :data:`GroupFieldValues`.
        isovalues: The isovalues to extract.
        strictly_above: Count a value as above the isovalue only if it is
            greater (``>``) instead of greater or equal (``>=``). Must match the
            labels the cells were refined with.

    Returns:
        One :class:`DualContourMesh` per isovalue.
    """
    ijk = np.asarray(cell_ijk, dtype=np.int64).reshape(-1, 3)
    groups = np.asarray(cell_groups, dtype=np.int64).reshape(-1)
    if ijk.shape[0] == 0:
        return [_empty_mesh() for _ in isovalues]

    spacing = np.asarray(grid.finest_spacing, dtype=np.float64)
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)

    # Corner values, each (lattice point, group) once.
    corners = ijk[:, None, :] + CORNER_OFFSETS[None, :, :]                       # (M, 8, 3)
    corner_keys = np.column_stack([corners.reshape(-1, 3), np.repeat(groups, 8)])
    unique_corners, corner_inverse = np.unique(corner_keys, axis=0, return_inverse=True)
    corner_values = np.asarray(
        values(grid.origin + unique_corners[:, :3] * spacing, unique_corners[:, 3]), dtype=np.float64)
    corner_values = corner_values[corner_inverse.ravel()].reshape(-1, 8)        # (M, 8)

    # Edges: key = (lower-end lattice point, axis, group); each cell has 12.
    lower = corners[:, _EDGES[:, 0], :]                                          # (M, 12, 3)
    edge_keys = np.column_stack([
        lower.reshape(-1, 3), np.tile(_EDGE_AXIS, ijk.shape[0]), np.repeat(groups, 12)])
    unique_edges, edge_inverse = np.unique(edge_keys, axis=0, return_inverse=True)
    edge_inverse = edge_inverse.ravel().reshape(-1, 12)                          # (M, 12)
    first = np.empty(unique_edges.shape[0], dtype=np.int64)
    first[edge_inverse.ravel()] = np.arange(edge_inverse.size)                   # any incidence of each edge
    va = corner_values[:, _EDGES[:, 0]].ravel()[first]
    vb = corner_values[:, _EDGES[:, 1]].ravel()[first]

    # The 4 cells around each edge, in cyclic order. A cell sees an edge at its
    # corner offsets (o1, o2) along the two other axes, taken in right-handed
    # order (y, z), (z, x), (x, y); the slot follows from them.
    other_axes = np.array([[1, 2], [2, 0], [0, 1]], dtype=np.int64)
    offsets = CORNER_OFFSETS[_EDGES[:, 0]]                                       # (12, 3)
    o = offsets[np.arange(12)[:, None], other_axes[_EDGE_AXIS]]                  # (12, 2)
    slot_of = np.array([[0, 3], [1, 2]], dtype=np.int64)                         # (o1, o2) -> slot
    edge_slot = slot_of[o[:, 0], o[:, 1]]                                        # (12,)
    ring = np.full((unique_edges.shape[0], 4), -1, dtype=np.int64)
    ring[edge_inverse.ravel(), np.tile(edge_slot, ijk.shape[0])] = np.repeat(np.arange(ijk.shape[0]), 12)

    # An edge on the extent's outer faces can't have 4 cells around it.
    lo_pt = unique_edges[:, :3]
    axis = unique_edges[:, 3]
    on_wall = np.zeros(unique_edges.shape[0], dtype=bool)
    for a in range(3):
        across = axis != a
        on_wall |= across & ((lo_pt[:, a] == 0) | (lo_pt[:, a] == n_finest[a]))

    # Look-up of all cells by position, to tell another group's cell from no cell.
    keys = linear_cell_index(ijk, n_finest)
    order = np.argsort(keys)
    sorted_keys = keys[order]

    def find(points_ijk: IntArray) -> IntArray:
        """Row of the cell at each position, -1 where there is none."""
        inside = np.all((points_ijk >= 0) & (points_ijk < n_finest), axis=1)
        k = linear_cell_index(np.where(inside[:, None], points_ijk, 0), n_finest)
        pos = np.minimum(np.searchsorted(sorted_keys, k), sorted_keys.size - 1)
        return np.where(inside & (sorted_keys[pos] == k), order[pos], -1)

    at_group_boundary = np.zeros(ijk.shape[0], dtype=bool)
    for step in _NEIGHBOUR_STEPS:
        neighbour = find(ijk + step)
        at_group_boundary |= (neighbour >= 0) & (groups[np.maximum(neighbour, 0)] != groups)
    slot_offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)  # (o1, o2) per slot

    h = _GRADIENT_STEP_CELLS * spacing
    cell_min = grid.origin + ijk * spacing
    meshes = []
    for isovalue in isovalues:
        if strictly_above:
            crossing = (va > isovalue) != (vb > isovalue)
        else:
            crossing = (va >= isovalue) != (vb >= isovalue)
        idx = np.flatnonzero(crossing)
        if idx.size == 0:
            meshes.append(_empty_mesh())
            continue

        # Crossing points and normals.
        t = (isovalue - va[idx]) / (vb[idx] - va[idx])
        direction = np.eye(3)[axis[idx]]
        points = grid.origin + (lo_pt[idx] + t[:, None] * direction) * spacing
        edge_groups = unique_edges[idx, 4]
        normals = np.column_stack([
            (values(points + h[a] * np.eye(3)[a], edge_groups)
             - values(points - h[a] * np.eye(3)[a], edge_groups)) / (2.0 * h[a])
            for a in range(3)
        ])
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norm > 1e-12, norm, 1.0)

        # QEF per cell over its crossing edges.
        crossing_number = np.full(unique_edges.shape[0], -1, dtype=np.int64)
        crossing_number[idx] = np.arange(idx.size)
        cell_cross = crossing_number[edge_inverse]                               # (M, 12)
        valid = cell_cross >= 0
        has_vertex = valid.any(axis=1)
        cv = cell_cross[has_vertex]
        valid_v = valid[has_vertex]
        p = np.where(valid_v[..., None], points[np.maximum(cv, 0)], 0.0)        # (V, 12, 3)
        n = np.where(valid_v[..., None], normals[np.maximum(cv, 0)], 0.0)
        mass = p.sum(axis=1) / valid_v.sum(axis=1, keepdims=True)
        a_rows = np.concatenate([n, np.broadcast_to(np.eye(3), (cv.shape[0], 3, 3))], axis=1)
        b_rows = np.concatenate([np.einsum("vri,vri->vr", n, p), mass], axis=1)
        ata = np.einsum("vri,vrj->vij", a_rows, a_rows)
        atb = np.einsum("vri,vr->vi", a_rows, b_rows)
        vertices = np.linalg.solve(ata, atb[..., None])[..., 0]
        lo = cell_min[has_vertex]
        vertices = np.clip(vertices, lo, lo + spacing)

        # Quads around crossing edges with all 4 cells present.
        vertex_of_cell = np.full(ijk.shape[0], -1, dtype=np.int64)
        vertex_of_cell[has_vertex] = np.arange(int(has_vertex.sum()))
        quads = ring[idx]
        complete = np.all(quads >= 0, axis=1)
        gap = np.zeros(quads.shape[0], dtype=bool)
        for slot in range(4):
            empty = np.flatnonzero((quads[:, slot] < 0) & ~on_wall[idx])
            if empty.size == 0:
                continue
            # The cell that would fill the slot: is it another group's or absent?
            e = idx[empty]
            cell = lo_pt[e].copy()
            axes = other_axes[axis[e]]
            rows = np.arange(e.size)
            cell[rows, axes[:, 0]] -= slot_offsets[slot, 0]
            cell[rows, axes[:, 1]] -= slot_offsets[slot, 1]
            gap[empty[find(cell) < 0]] = True
        n_missing = int(gap.sum())
        quads = vertex_of_cell[quads[complete]]
        # Slots run counter-clockwise around the edge's axis seen from its +
        # end, so the quad normal points along +axis; flip where the field
        # decreases along the edge.
        increasing = (vb[idx] > va[idx])[complete]
        quads = np.where(increasing[:, None], quads, quads[:, ::-1])
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        meshes.append(DualContourMesh(
            vertices=vertices, faces=faces.astype(np.int64), vertex_cells=ijk[has_vertex],
            vertex_at_group_boundary=at_group_boundary[has_vertex], n_missing_quads=n_missing,
        ))
    return meshes


def interior_open_edges(mesh: DualContourMesh, grid: OctreeGrid) -> int:
    """
    Number of open mesh edges (used by one triangle) that border a gap.

    A surface legitimately ends half a cell inside a wall of the extent and half a
    cell before a cell of another group: an open edge whose two vertices both come
    from cells next to the same wall, or both from cells next to another group, is
    such an end. Every other open edge borders a gap in the surface.
    """
    if mesh.faces.shape[0] == 0:
        return 0
    edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique[counts == 1]
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)
    near_wall = np.concatenate([mesh.vertex_cells == 0, mesh.vertex_cells == n_finest - 1], axis=1)  # (V, 6)
    on_wall = np.any(near_wall[open_edges[:, 0]] & near_wall[open_edges[:, 1]], axis=1)
    at_boundary = mesh.vertex_at_group_boundary[open_edges[:, 0]] & mesh.vertex_at_group_boundary[open_edges[:, 1]]
    return int(np.sum(~on_wall & ~at_boundary))


def _empty_mesh() -> DualContourMesh:
    return DualContourMesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64),
        vertex_cells=np.zeros((0, 3), dtype=np.int64), vertex_at_group_boundary=np.zeros(0, dtype=bool),
        n_missing_quads=0,
    )
