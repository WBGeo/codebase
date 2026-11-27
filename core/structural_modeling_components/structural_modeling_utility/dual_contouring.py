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
     gap. With a cell mask (MASKED meshes), a quad is kept only if all 4 of its
     cells are kept.

EXTENDED meshes run on the cells plus a padding shell outside the extent
(:func:`padding_shell`), on a lattice padded by the same number of cells.

Ported from the flat dual-contouring path of the octree research code (itself a
port of GemPy's), adapted to explicit isovalues, integer cell addresses and cell
groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

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

# Largest lattice (in max-level cells) that gets a dense cell look-up array (int32,
# 4 bytes per entry, so up to 256 MB); larger lattices use a sorted search instead.
_DENSE_LOOKUP_MAX_ENTRIES: int = 1 << 26

# Dense key marking in _unique_keys when the key range is at most this many times
# the number of keys and at most _DENSE_UNIQUE_MAX_ENTRIES (a bool and an int32
# array over the range, 5 bytes per entry, so up to 2.5 GB); else a sort.
_DENSE_UNIQUE_RANGE_FACTOR: int = 8
_DENSE_UNIQUE_MAX_ENTRIES: int = 1 << 29


@dataclass
class DualContourMesh:
    """
    A triangle mesh from :func:`dual_contour`.

    Attributes:
        vertices: Vertex coordinates, shape ``(V, 3)``.
        faces: Vertex indices of the triangles, shape ``(F, 3)``; normals point
            toward increasing field values. Each quad is split into two
            triangles: with ``Q = F // 2`` quads, quad ``q`` is triangles ``q``
            and ``q + Q`` (see :func:`restrict_to_kept_cells`).
        vertex_cells: Max-level cell index ``(i, j, k)`` each vertex belongs to,
            shape ``(V, 3)``.
        vertex_rows: Row of each vertex's cell in the ``cell_ijk`` the mesh was
            contoured on, shape ``(V,)``.
        vertex_at_group_boundary: Whether a vertex's cell touches a cell of
            another group (shares a face, an edge or a corner), shape ``(V,)``.
        n_missing_quads: Crossing edges inside the extent that got no quad
            because a cell around them is not at max level: gaps in the surface.
    """

    vertices: FloatArray
    faces: IntArray
    vertex_cells: IntArray
    vertex_rows: IntArray
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


def padding_shell(grid: OctreeGrid, padding: int) -> IntArray:
    """
    All max-level cells of a shell ``padding`` finest cells thick around the extent.

    The octree counterpart of the regular grid's padded margin: the extent
    padded by ``padding`` cells on every side, minus the extent itself.

    Args:
        grid: The octree grid.
        padding: Shell thickness in finest cells.

    Returns:
        Cell indices of shape ``(S, 3)``, each outside ``[0, finest_resolution)``
        along at least one axis and inside ``[-padding, finest_resolution + padding)``.
    """
    n = np.array(grid.finest_resolution, dtype=np.int64)
    outer = [np.arange(-padding, n[a] + padding) for a in range(3)]
    inner = [np.arange(n[a]) for a in range(3)]
    sides = [np.r_[-padding:0, n[a]:n[a] + padding] for a in range(3)]

    def block(xs, ys, zs):
        return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    # Slabs outside x (full y, z), then outside y (x inside, full z), then outside z (x, y inside).
    return np.concatenate([
        block(sides[0], outer[1], outer[2]),
        block(inner[0], sides[1], outer[2]),
        block(inner[0], inner[1], sides[2]),
    ]).astype(np.int64)


def dual_contour(
        grid: OctreeGrid,
        cell_ijk: npt.NDArray[np.integer],
        cell_groups: npt.NDArray[np.integer],
        values: GroupFieldValues,
        isovalues: Sequence[float],
        strictly_above: bool = False,
        cell_kept: Optional[npt.NDArray[np.bool_]] = None,
        padding: int = 0,
        field_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
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
        cell_kept: Optional mask of the cells, shape ``(M,)``. A quad is kept
            only if all 4 of its cells are kept; vertices no kept quad uses are
            dropped. Quads dropped this way are no gaps. ``None`` keeps all.
        padding: Cells may lie up to this many finest cells outside the extent
            (see :func:`padding_shell`); the walls are those of the padded box.
        field_bounds: Optional ``(xmin, xmax, ymin, ymax, zmin, zmax)`` the field
            is defined in; the points of the normals' difference stencils are
            clamped into it.

    Returns:
        One :class:`DualContourMesh` per isovalue.
    """
    ijk = np.asarray(cell_ijk, dtype=np.int64).reshape(-1, 3)
    groups = np.asarray(cell_groups, dtype=np.int64).reshape(-1)
    kept = np.ones(ijk.shape[0], dtype=bool) if cell_kept is None else np.asarray(cell_kept, dtype=bool).reshape(-1)
    if ijk.shape[0] == 0:
        return [_empty_mesh() for _ in isovalues]

    spacing = np.asarray(grid.finest_spacing, dtype=np.float64)
    # The lattice: [-padding, finest_resolution + padding) along each axis.
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)
    n_lattice = n_finest + 2 * padding

    # Keys, one int64 each, ordered like the rows (x, y, z, [axis,] group): lattice
    # points (corners) are numbered over the padded lattice, groups by rank.
    group_ids, group_rank = np.unique(groups, return_inverse=True)
    n_groups = group_ids.size
    group_rank = group_rank.reshape(-1).astype(np.int64)
    n_points = n_lattice + 1                                                     # lattice points per axis
    shifted = ijk + padding
    cell_point = (shifted[:, 0] * n_points[1] + shifted[:, 1]) * n_points[2] + shifted[:, 2]  # lower corner
    corner_step = (CORNER_OFFSETS[:, 0] * n_points[1] + CORNER_OFFSETS[:, 1]) * n_points[2] + CORNER_OFFSETS[:, 2]

    # Corner values, each (lattice point, group) once.
    corner_keys = (cell_point[:, None] + corner_step[None, :]) * n_groups + group_rank[:, None]  # (M, 8)
    corner_unique, corner_inverse = _unique_keys(corner_keys.reshape(-1), int(np.prod(n_points)) * n_groups)
    del corner_keys
    corner_points, corner_group = np.divmod(corner_unique, n_groups)
    corner_values = np.asarray(
        values(grid.origin + (_lattice_coordinates(corner_points, n_points) - padding) * spacing,
               group_ids[corner_group]), dtype=np.float64)
    corner_values = corner_values[corner_inverse].reshape(-1, 8)                # (M, 8)
    del corner_unique, corner_inverse, corner_points, corner_group

    # From here on only the cells with a crossing: the 4 cells of a quad all contain
    # its crossing edge, so cells whose corners lie on one side of every isovalue
    # (e.g. cells refined for another group's surfaces) give neither vertices nor
    # quads. The look-up of neighbours below still sees all cells.
    all_ijk, all_groups = ijk, groups
    crossed = np.zeros(ijk.shape[0], dtype=bool)
    for isovalue in isovalues:
        above = corner_values > isovalue if strictly_above else corner_values >= isovalue
        crossed |= above.any(axis=1) & ~above.all(axis=1)
    cell_rows = np.flatnonzero(crossed)
    if cell_rows.size == 0:
        return [_empty_mesh() for _ in isovalues]
    ijk, groups, kept = ijk[cell_rows], groups[cell_rows], kept[cell_rows]
    cell_point, group_rank, corner_values = cell_point[cell_rows], group_rank[cell_rows], corner_values[cell_rows]
    del crossed

    # Edges: key = (lower-end lattice point, axis, group); each cell has 12.
    edge_keys = ((cell_point[:, None] + corner_step[_EDGES[:, 0]][None, :]) * 3
                 + _EDGE_AXIS[None, :]) * n_groups + group_rank[:, None]         # (M, 12)
    edge_unique, edge_inverse = _unique_keys(edge_keys.reshape(-1), int(np.prod(n_points)) * 3 * n_groups)
    del edge_keys
    edge_inverse = edge_inverse.reshape(-1, 12)                                  # (M, 12)
    n_edges = edge_unique.size
    edge_rest, edge_group_rank = np.divmod(edge_unique, n_groups)
    edge_points, axis = np.divmod(edge_rest, 3)
    lo_pt = _lattice_coordinates(edge_points, n_points) - padding                # (E, 3)
    edge_group = group_ids[edge_group_rank]
    del edge_unique, edge_rest, edge_points, edge_group_rank
    first = np.empty(n_edges, dtype=np.int64)
    first[edge_inverse.reshape(-1)] = np.arange(edge_inverse.size, dtype=np.int64)  # any incidence of each edge
    first_cell, first_edge = np.divmod(first, 12)
    va = corner_values[first_cell, _EDGES[first_edge, 0]]
    vb = corner_values[first_cell, _EDGES[first_edge, 1]]
    del first, first_cell, first_edge

    # The 4 cells around each edge, in cyclic order. A cell sees an edge at its
    # corner offsets (o1, o2) along the two other axes, taken in right-handed
    # order (y, z), (z, x), (x, y); the slot follows from them.
    other_axes = np.array([[1, 2], [2, 0], [0, 1]], dtype=np.int64)
    offsets = CORNER_OFFSETS[_EDGES[:, 0]]                                       # (12, 3)
    o = offsets[np.arange(12)[:, None], other_axes[_EDGE_AXIS]]                  # (12, 2)
    slot_of = np.array([[0, 3], [1, 2]], dtype=np.int64)                         # (o1, o2) -> slot
    edge_slot = slot_of[o[:, 0], o[:, 1]]                                        # (12,)
    ring = np.full((n_edges, 4), -1, dtype=np.int32)
    ring[edge_inverse, edge_slot[None, :]] = np.arange(ijk.shape[0], dtype=np.int32)[:, None]

    # An edge on the extent's outer faces can't have 4 cells around it.
    on_wall = np.zeros(n_edges, dtype=bool)
    for a in range(3):
        across = axis != a
        on_wall |= across & ((lo_pt[:, a] == -padding) | (lo_pt[:, a] == n_finest[a] + padding))

    # Look-up of all cells by position (rows of all_ijk), to tell another group's cell
    # from no cell: a dense array of cell rows over the lattice where it fits, else a
    # sorted search.
    keys = linear_cell_index(all_ijk + padding, n_lattice)
    if int(np.prod(n_lattice)) <= _DENSE_LOOKUP_MAX_ENTRIES:
        row_at = np.full(int(np.prod(n_lattice)), -1, dtype=np.int32)
        row_at[keys] = np.arange(all_ijk.shape[0], dtype=np.int32)

        def rows_at(k: IntArray) -> IntArray:
            return row_at[k]
    else:
        order = np.argsort(keys)
        sorted_keys = keys[order]

        def rows_at(k: IntArray) -> IntArray:
            pos = np.minimum(np.searchsorted(sorted_keys, k), sorted_keys.size - 1)
            return np.where(sorted_keys[pos] == k, order[pos], -1)

    def find(points_ijk: IntArray) -> IntArray:
        """Row (in all_ijk) of the cell at each position, -1 where there is none."""
        shifted = points_ijk + padding
        inside = np.all((shifted >= 0) & (shifted < n_lattice), axis=1)
        k = linear_cell_index(np.where(inside[:, None], shifted, 0), n_lattice)
        return np.where(inside, rows_at(k), -1)

    at_group_boundary = np.zeros(ijk.shape[0], dtype=bool)
    for step in _NEIGHBOUR_STEPS:
        neighbour = find(ijk + step)
        at_group_boundary |= (neighbour >= 0) & (all_groups[np.maximum(neighbour, 0)] != groups)
    slot_offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)  # (o1, o2) per slot

    h = _GRADIENT_STEP_CELLS * spacing
    if field_bounds is None:
        bounds_lo, bounds_hi = np.full(3, -np.inf), np.full(3, np.inf)
    else:
        bounds_lo, bounds_hi = np.array(field_bounds[0::2]), np.array(field_bounds[1::2])
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
        edge_groups = edge_group[idx]
        # All 6 stencil points in one call: every call into GemPy recompiles its
        # backend, which leaks Windows DLL-directory handles over many calls.
        steps = np.concatenate([np.diag(h), -np.diag(h)])                        # +x, +y, +z, -x, -y, -z
        stencil = np.clip(points[None, :, :] + steps[:, None, :], bounds_lo, bounds_hi)
        stencil_values = np.asarray(
            values(stencil.reshape(-1, 3), np.tile(edge_groups, 6)), dtype=np.float64).reshape(6, -1)
        distance = stencil[:3, :, :] - stencil[3:, :, :]                          # (3, K, 3)
        normals = ((stencil_values[:3] - stencil_values[3:])
                   / distance[np.arange(3), :, np.arange(3)]).T
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norm > 1e-12, norm, 1.0)

        # QEF per cell over its crossing edges.
        crossing_number = np.full(n_edges, -1, dtype=np.int32)
        crossing_number[idx] = np.arange(idx.size, dtype=np.int32)
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
        # Only gaps among kept cells count; quads with a masked cell are dropped anyway.
        all_kept = np.all(np.where(quads >= 0, kept[np.maximum(quads, 0)], True), axis=1)
        n_missing = int(np.sum(gap & all_kept))
        complete &= all_kept
        quads = vertex_of_cell[quads[complete]]
        # Slots run counter-clockwise around the edge's axis seen from its +
        # end, so the quad normal points along +axis; flip where the field
        # decreases along the edge.
        increasing = (vb[idx] > va[idx])[complete]
        quads = np.where(increasing[:, None], quads, quads[:, ::-1])
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        vertex_rows = cell_rows[has_vertex]                                     # rows of cell_ijk
        boundary = at_group_boundary[has_vertex]
        if cell_kept is not None:
            vertices, faces, used = select_faces(vertices, faces, np.ones(faces.shape[0], dtype=bool))
            vertex_rows, boundary = vertex_rows[used], boundary[used]
        meshes.append(DualContourMesh(
            vertices=vertices, faces=faces.astype(np.int64), vertex_cells=all_ijk[vertex_rows],
            vertex_rows=vertex_rows, vertex_at_group_boundary=boundary, n_missing_quads=n_missing,
        ))
    return meshes


def restrict_to_kept_cells(mesh: DualContourMesh, cell_kept: npt.NDArray[np.bool_]) -> DualContourMesh:
    """
    The mesh with only the quads all 4 of whose cells are kept.

    The same mesh as :func:`dual_contour` with ``cell_kept`` gives, without a
    second contouring pass: a quad's 4 vertices are its 4 cells' vertices, so
    a quad is kept if all of them are. Only for meshes as :func:`dual_contour`
    returns them (or restricted by this function), whose triangles ``q`` and
    ``q + F // 2`` form quad ``q``.

    Args:
        mesh: A mesh from :func:`dual_contour`.
        cell_kept: Mask of the cells the mesh was contoured on, shape ``(M,)``.

    Returns:
        The kept quads with only the vertices they use, in their original
        order. ``n_missing_quads`` is the unrestricted mesh's.
    """
    kept_vertex = np.asarray(cell_kept, dtype=bool)[mesh.vertex_rows]
    n_quads = mesh.faces.shape[0] // 2
    kept_quad = kept_vertex[mesh.faces[:n_quads]].all(axis=1) & kept_vertex[mesh.faces[n_quads:]].all(axis=1)
    vertices, faces, used = select_faces(mesh.vertices, mesh.faces, np.concatenate([kept_quad, kept_quad]))
    return DualContourMesh(
        vertices=vertices, faces=faces, vertex_cells=mesh.vertex_cells[used], vertex_rows=mesh.vertex_rows[used],
        vertex_at_group_boundary=mesh.vertex_at_group_boundary[used], n_missing_quads=mesh.n_missing_quads,
    )


def _unique_keys(keys: IntArray, key_range: int) -> Tuple[IntArray, npt.NDArray[np.int32]]:
    """
    ``np.unique(keys, return_inverse=True)`` for keys in ``[0, key_range)``, lean on memory.

    Corner and edge keys pack a whole row (lattice point, axis, group) into one
    int64, so they sort like the rows; ``np.unique`` on rows (``axis=0``) took
    most of the time of dual contouring, and the rows most of its memory. When
    the range is not much larger than the number of keys (cells packed around
    surfaces), the keys are marked in an array over the range and numbered in
    order, without sorting; otherwise they are sorted.

    Args:
        keys: Non-negative int64 keys, shape ``(K,)``.
        key_range: An upper bound of the keys (exclusive).

    Returns:
        ``(unique, inverse)``: the distinct keys in increasing order (int64) and,
        for each key, the index of its distinct key (int32), shape ``(K,)``.
    """
    if keys.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    if key_range <= min(_DENSE_UNIQUE_RANGE_FACTOR * keys.size, _DENSE_UNIQUE_MAX_ENTRIES):
        present = np.zeros(key_range, dtype=bool)
        present[keys] = True
        number = np.cumsum(present, dtype=np.int32)
        number -= 1
        return np.flatnonzero(present), number[keys]
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, inverse.reshape(-1).astype(np.int32)


def _lattice_coordinates(points: IntArray, n_points: IntArray) -> IntArray:
    """Lattice indices ``(i, j, k)`` of points numbered ``(i * n_points[1] + j) * n_points[2] + k``, shape ``(K, 3)``."""
    rest, k = np.divmod(points, n_points[2])
    i, j = np.divmod(rest, n_points[1])
    return np.column_stack([i, j, k])


def select_faces(
        vertices: FloatArray, faces: npt.NDArray[np.integer], keep: npt.NDArray[np.bool_]
) -> Tuple[FloatArray, IntArray, IntArray]:
    """
    The triangles where ``keep`` is True, with only the vertices they use.

    Args:
        vertices: Vertex coordinates, shape ``(V, 3)``.
        faces: Vertex indices of the triangles, shape ``(F, 3)``.
        keep: Which triangles to keep, shape ``(F,)``.

    Returns:
        ``(vertices, faces, used)``: the kept vertices in their original order,
        the kept triangles renumbered to them (int64), and the original index of
        each kept vertex.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)[np.asarray(keep, dtype=bool)]
    used = np.unique(faces)
    new_index = np.full(np.asarray(vertices).shape[0], -1, dtype=np.int64)
    new_index[used] = np.arange(used.size)
    return np.asarray(vertices)[used], new_index[faces], used


def interior_open_edges(mesh: DualContourMesh, grid: OctreeGrid, padding: int = 0) -> int:
    """
    Number of open mesh edges (used by one triangle) that border a gap.

    A surface legitimately ends half a cell inside a wall of the extent and half a
    cell before a cell of another group: an open edge whose two vertices both come
    from cells next to the same wall, or both from cells next to another group, is
    such an end. Every other open edge borders a gap in the surface. With
    ``padding`` (EXTENDED meshes), the walls are those of the padded box.
    """
    if mesh.faces.shape[0] == 0:
        return 0
    edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique[counts == 1]
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)
    near_wall = np.concatenate([mesh.vertex_cells == -padding, mesh.vertex_cells == n_finest - 1 + padding],
                               axis=1)  # (V, 6)
    on_wall = np.any(near_wall[open_edges[:, 0]] & near_wall[open_edges[:, 1]], axis=1)
    at_boundary = mesh.vertex_at_group_boundary[open_edges[:, 0]] & mesh.vertex_at_group_boundary[open_edges[:, 1]]
    return int(np.sum(~on_wall & ~at_boundary))


def _empty_mesh() -> DualContourMesh:
    return DualContourMesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64),
        vertex_cells=np.zeros((0, 3), dtype=np.int64), vertex_rows=np.zeros(0, dtype=np.int64),
        vertex_at_group_boundary=np.zeros(0, dtype=bool), n_missing_quads=0,
    )
