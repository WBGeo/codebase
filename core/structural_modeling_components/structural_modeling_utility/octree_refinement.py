"""
Adaptive refinement and coarsening of an :class:`OctreeGrid`.

:func:`refine_cells`: starting from the base blocks (or from given cells), every
cell whose 8 corners get different labels is split into its 8 children, level by
level, until ``max_level``. Cells whose corners all agree become leaves.

:func:`coarsen_cells`: the reverse, bottom-up: 8 sibling leaves are merged back
into their parent where the labels are identical across the whole parent. This
lets a model refine for one purpose (e.g. surfaces needed for meshes) and then
store only the cells another purpose needs (e.g. the lithology).

The result of both is a set of cells (:class:`OctreeCells`) that tiles the extent.

Both can work per fault domain: given the domain of every cell
(:data:`CellDomains`), each point is labelled with the domain of the cell it
belongs to, a lattice point shared by cells of different domains is labelled
once per domain, and only siblings of one domain are merged (or, with separate
merge labels, only siblings with the same merge label).

:func:`balance_2to1`: refine until face-adjacent cells differ by at most one
level.

Ported from the adaptive classification loop of the octree research code
(corner-disagreement refinement), adapted to integer cell addresses and a grid
of base blocks.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    CORNER_OFFSETS,
    OctreeCells,
    OctreeGrid,
    linear_cell_index,
)

# Labels points: (M, 3) coordinates -> (M,) or (M, k) integer labels. A cell is
# split when any label column differs between its corners.
PointLabeler = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.integer]]

# Labels points of known domains: (M, 3) coordinates and (M,) domain ids -> labels.
DomainPointLabeler = Callable[[npt.NDArray[np.float64], npt.NDArray[np.int64]], npt.NDArray[np.integer]]

# Domain of cells: a level and (N, 3) cell indices at that level -> (N,) domain ids.
CellDomains = Callable[[int, npt.NDArray[np.int64]], npt.NDArray[np.int64]]


def base_cells(grid: OctreeGrid) -> OctreeCells:
    """All base blocks of ``grid`` as level-0 cells."""
    axes = (np.arange(n) for n in grid.base_resolution)
    ii, jj, kk = np.meshgrid(*axes, indexing="ij")
    ijk = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    return OctreeCells(level=np.zeros(len(ijk), dtype=np.uint8), ijk=ijk.astype(np.int32))


def refine_cells(
        grid: OctreeGrid,
        label_points: Union[PointLabeler, DomainPointLabeler],
        start: Optional[OctreeCells] = None,
        cell_domains: Optional[CellDomains] = None,
) -> OctreeCells:
    """
    Refine cells wherever their corner labels disagree.

    Args:
        grid: The octree grid.
        label_points: Labels for points; see :data:`PointLabeler`. Only corner
            points are labelled. With ``cell_domains`` a :data:`DomainPointLabeler`.
        start: Cells to start from, at any levels. Defaults to the base blocks.
            Cells already at ``max_level`` are kept as they are.
        cell_domains: Domain of each cell; see :data:`CellDomains`. When given,
            the corners of a cell are labelled with the cell's domain. Children
            lie inside their parent, so they keep its domain as long as no start
            cell spans two domains.

    Returns:
        The leaf cells. They tile the extent; cells of the same level keep
        the order in which they were created.
    """
    start = start if start is not None else base_cells(grid)
    pending: Dict[int, List[npt.NDArray[np.int64]]] = {}
    for level in np.unique(start.level):
        pending[int(level)] = [start.ijk[start.level == level].astype(np.int64)]

    leaf_levels: List[npt.NDArray[np.uint8]] = []
    leaf_ijk: List[npt.NDArray[np.int64]] = []

    for level in range(grid.max_level + 1):
        if level not in pending:
            continue
        ijk = np.concatenate(pending.pop(level))
        if level == grid.max_level or ijk.shape[0] == 0:
            mixed = np.zeros(ijk.shape[0], dtype=bool)
        else:
            mixed = _corners_disagree(grid, level, ijk, label_points, cell_domains)

        leaf_levels.append(np.full(int((~mixed).sum()), level, dtype=np.uint8))
        leaf_ijk.append(ijk[~mixed])

        if mixed.any():
            children = 2 * ijk[mixed][:, None, :] + CORNER_OFFSETS[None, :, :]
            pending.setdefault(level + 1, []).append(children.reshape(-1, 3))

    return OctreeCells(level=np.concatenate(leaf_levels), ijk=np.concatenate(leaf_ijk).astype(np.int32))


def coarsen_cells(
        grid: OctreeGrid,
        cells: OctreeCells,
        label_points: Union[PointLabeler, DomainPointLabeler],
        cell_domains: Optional[CellDomains] = None,
        merge_labels: Optional[CellDomains] = None,
) -> OctreeCells:
    """
    Merge sibling leaves back into their parent wherever the labels don't change.

    Bottom-up from ``max_level``: when all 8 children of a parent are leaves and
    every label column is identical at all 27 lattice points of the parent (its
    corners, edge and face midpoints and center, which are the corners of the 8
    children), the children are replaced by the parent. Merged parents can merge
    again one level up.

    Labels never change inside a merged cell at the lattice points checked, so a
    per-cell quantity derived from the labels (e.g. lithology classified at cell
    centers) is the same before and after coarsening.

    Args:
        grid: The octree grid.
        cells: Leaf cells that tile the extent, e.g. from :func:`refine_cells`.
        label_points: Labels for points; see :data:`PointLabeler`. With
            ``cell_domains`` a :data:`DomainPointLabeler`.
        cell_domains: Domain of each cell; see :data:`CellDomains`. When given,
            each child's corners are labelled with the child's domain, and
            (without ``merge_labels``) only children of one domain are merged.
        merge_labels: Optional label of each cell, in the form of
            :data:`CellDomains`: only children with the same merge label are
            merged. Lets children of different domains merge where that doesn't
            matter, while their corners are still labelled with their own
            domains. Defaults to ``cell_domains``.

    Returns:
        The coarsened leaf cells.
    """
    merge_labels = merge_labels if merge_labels is not None else cell_domains
    level = cells.level.astype(np.int64)
    ijk = cells.ijk.astype(np.int64)

    for child_level in range(grid.max_level, 0, -1):
        at_level = np.flatnonzero(level == child_level)
        if at_level.size == 0:
            continue
        parent_level = child_level - 1
        parent_ijk_of_child = ijk[at_level] // 2
        keys = linear_cell_index(parent_ijk_of_child, grid.cells_per_axis(parent_level))
        unique_keys, first, inverse, counts = np.unique(
            keys, return_index=True, return_inverse=True, return_counts=True)
        complete = counts == 8
        if not complete.any():
            continue

        candidates = parent_ijk_of_child[first[complete]]
        one_label = np.ones(candidates.shape[0], dtype=bool)
        if merge_labels is not None:
            child_labels = np.asarray(merge_labels(child_level, ijk[at_level]), dtype=np.int64)
            lowest = np.full(unique_keys.size, np.iinfo(np.int64).max)
            highest = np.full(unique_keys.size, np.iinfo(np.int64).min)
            np.minimum.at(lowest, inverse.ravel(), child_labels)
            np.maximum.at(highest, inverse.ravel(), child_labels)
            one_label = (lowest == highest)[complete]

        # The 8 children of every candidate, and their corners (the parent's 27
        # lattice points), each labelled with its child's domain.
        parent_of = inverse.ravel()
        is_candidate = complete[parent_of]
        order = np.argsort(parent_of[is_candidate], kind="stable")
        children = at_level[is_candidate][order].reshape(-1, 8)
        lattice = (ijk[children][:, :, None, :] + CORNER_OFFSETS[None, None, :, :]).reshape(-1, 64, 3)
        domains = None
        if cell_domains is not None:
            child_domains = np.asarray(cell_domains(child_level, ijk[children.ravel()]), dtype=np.int64)
            domains = np.repeat(child_domains.reshape(-1, 8), 8, axis=1)
        labels = _labels_at_lattice(grid, child_level, lattice, label_points, domains)
        uniform = one_label & np.all(labels == labels[:, :1, :], axis=(1, 2))
        if not uniform.any():
            continue

        merged_parent = np.zeros(unique_keys.size, dtype=bool)
        merged_parent[np.flatnonzero(complete)[uniform]] = True
        drop = at_level[merged_parent[inverse.ravel()]]
        keep = np.ones(level.size, dtype=bool)
        keep[drop] = False
        level = np.concatenate([level[keep], np.full(int(uniform.sum()), parent_level, dtype=np.int64)])
        ijk = np.concatenate([ijk[keep], candidates[uniform]])

    return OctreeCells(level=level.astype(np.uint8), ijk=ijk.astype(np.int32))


def balance_2to1(grid: OctreeGrid, cells: OctreeCells) -> OctreeCells:
    """
    Refine until cells that share a face differ by at most one level (2:1 balance).

    Repeatedly splits every leaf that has a face neighbour more than one level
    finer into its 8 children, until none is left. Only splits, never merges:
    every cell at max level stays, and the result refines ``cells``.

    Args:
        grid: The octree grid.
        cells: Leaf cells that tile the extent.

    Returns:
        The balanced leaves. ``cells`` itself if it is balanced already.
    """
    level = cells.level.astype(np.int64)
    ijk = cells.ijk.astype(np.int64)
    changed = False
    while True:
        # Level of every finest voxel, and the finest level among its face neighbours.
        level_map = np.empty(grid.finest_resolution, dtype=np.int8)
        for lvl in np.unique(level):
            members = np.flatnonzero(level == lvl)
            n = 2 ** (grid.max_level - int(lvl))
            view = level_map.reshape(level_map.shape[0] // n, n, level_map.shape[1] // n, n,
                                     level_map.shape[2] // n, n)
            view[ijk[members, 0], :, ijk[members, 1], :, ijk[members, 2], :] = lvl
        finer_neighbour = np.full(level_map.shape, -1, dtype=np.int8)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(0, -1), slice(1, None)
            np.maximum(finer_neighbour[tuple(lower)], level_map[tuple(upper)], out=finer_neighbour[tuple(lower)])
            np.maximum(finer_neighbour[tuple(upper)], level_map[tuple(lower)], out=finer_neighbour[tuple(upper)])

        split = np.zeros(level.size, dtype=bool)
        for lvl in np.unique(level):
            if lvl >= grid.max_level - 1:
                continue  # no neighbour can be two levels finer
            members = np.flatnonzero(level == lvl)
            n = 2 ** (grid.max_level - int(lvl))
            nx, ny, nz = grid.cells_per_axis(int(lvl))
            finest_near = finer_neighbour.reshape(nx, n, ny, n, nz, n).max(axis=(1, 3, 5))
            split[members] = finest_near[ijk[members, 0], ijk[members, 1], ijk[members, 2]] > lvl + 1
        if not split.any():
            break
        changed = True
        children = (2 * ijk[split])[:, None, :] + CORNER_OFFSETS[None, :, :]
        level = np.concatenate([level[~split], np.repeat(level[split] + 1, 8)])
        ijk = np.concatenate([ijk[~split], children.reshape(-1, 3)])
    if not changed:
        return cells
    return OctreeCells(level=level.astype(np.uint8), ijk=ijk.astype(np.int32))


def balance_mesh_depth_adjacency(grid: OctreeGrid, cells: OctreeCells) -> OctreeCells:
    """
    Split every leaf that shares a face with a max-level leaf down to max level (one pass).

    Dual contouring joins max-level cells only, so a surface crossing next to a
    coarser leaf would leave a gap; so would a surface that grazes a coarse leaf
    without changing the label at any of its corners. Splitting the coarse
    neighbours of all max-level leaves closes both kinds of gap in the octree
    research code's test models. Only one pass: repeated, it floods the whole
    extent.

    Args:
        grid: The octree grid.
        cells: Leaf cells that tile the extent.

    Returns:
        The leaves after the split; split leaves are replaced by their max-level
        descendants. Cells keep their positions within a level otherwise.
    """
    from scipy import ndimage

    level = cells.level.astype(np.int64)
    ijk = cells.ijk.astype(np.int64)
    at_max = level == grid.max_level
    if not at_max.any() or at_max.all():
        return cells

    finest = np.zeros(grid.finest_resolution, dtype=bool)
    finest[ijk[at_max, 0], ijk[at_max, 1], ijk[at_max, 2]] = True
    touched = ndimage.binary_dilation(finest, structure=ndimage.generate_binary_structure(3, 1))

    split = np.zeros(level.size, dtype=bool)
    for lvl in np.unique(level[~at_max]):
        members = np.flatnonzero(level == lvl)
        n = 2 ** (grid.max_level - int(lvl))
        nx, ny, nz = grid.cells_per_axis(int(lvl))
        blocks = touched.reshape(nx, n, ny, n, nz, n).any(axis=(1, 3, 5))
        split[members] = blocks[ijk[members, 0], ijk[members, 1], ijk[members, 2]]
    if not split.any():
        return cells

    new_ijk = [ijk[~split]]
    new_level = [level[~split]]
    for lvl in np.unique(level[split]):
        members = np.flatnonzero(split & (level == lvl))
        n = 2 ** (grid.max_level - int(lvl))
        a = np.arange(n)
        offsets = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)
        children = (ijk[members] * n)[:, None, :] + offsets[None, :, :]
        new_ijk.append(children.reshape(-1, 3))
        new_level.append(np.full(children.shape[0] * children.shape[1], grid.max_level, dtype=np.int64))
    return OctreeCells(level=np.concatenate(new_level).astype(np.uint8),
                       ijk=np.concatenate(new_ijk).astype(np.int32))


# The 27 lattice points of a cell, in units of its children's size.
_PARENT_LATTICE_OFFSETS: npt.NDArray[np.int64] = np.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=np.int64
)


def _corners_disagree(
        grid: OctreeGrid,
        level: int,
        ijk: npt.NDArray[np.int64],
        label_points: Union[PointLabeler, DomainPointLabeler],
        cell_domains: Optional[CellDomains] = None,
) -> npt.NDArray[np.bool_]:
    """For each cell, whether its 8 corners don't all get the same labels."""
    corners = ijk[:, None, :] + CORNER_OFFSETS[None, :, :]
    domains = None if cell_domains is None else np.asarray(cell_domains(level, ijk), dtype=np.int64)
    per_corner = _labels_at_lattice(grid, level, corners, label_points, domains)
    return np.any(per_corner != per_corner[:, :1, :], axis=(1, 2))


def _labels_at_lattice(
        grid: OctreeGrid,
        level: int,
        lattice: npt.NDArray[np.int64],
        label_points: Union[PointLabeler, DomainPointLabeler],
        domains: Optional[npt.NDArray[np.int64]] = None,
) -> npt.NDArray[np.integer]:
    """
    Labels at integer lattice points of a level, shape ``lattice.shape[:2] + (k,)``.

    Neighbouring cells share lattice points, so each distinct point is labelled
    once. With ``domains`` (one per row, shape ``(n_cells,)``, or one per point,
    shape ``lattice.shape[:2]``) each point is labelled with its domain, and a
    point is shared only between rows of the same domain.
    """
    n_cells, n_points = lattice.shape[:2]
    keys = lattice.reshape(-1, 3)
    if domains is not None:
        domains = np.asarray(domains)
        per_point = domains.ravel() if domains.ndim == 2 else np.repeat(domains, n_points)
        keys = np.column_stack([keys, per_point])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    points = grid.origin + unique_keys[:, :3] * grid.cell_size(level)
    if domains is None:
        labels = np.asarray(label_points(points))
    else:
        labels = np.asarray(label_points(points, unique_keys[:, 3]))
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels[inverse.ravel()].reshape(n_cells, n_points, labels.shape[1])
