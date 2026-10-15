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

Ported from the adaptive classification loop of the octree research code
(corner-disagreement refinement), adapted to integer cell addresses and a grid
of base blocks.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

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


def base_cells(grid: OctreeGrid) -> OctreeCells:
    """All base blocks of ``grid`` as level-0 cells."""
    axes = (np.arange(n) for n in grid.base_resolution)
    ii, jj, kk = np.meshgrid(*axes, indexing="ij")
    ijk = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    return OctreeCells(level=np.zeros(len(ijk), dtype=np.uint8), ijk=ijk.astype(np.int32))


def refine_cells(
        grid: OctreeGrid,
        label_points: PointLabeler,
        start: Optional[OctreeCells] = None,
) -> OctreeCells:
    """
    Refine cells wherever their corner labels disagree.

    Args:
        grid: The octree grid.
        label_points: Labels for points; see :data:`PointLabeler`. Only corner
            points are labelled.
        start: Cells to start from, at any levels. Defaults to the base blocks.
            Cells already at ``max_level`` are kept as they are.

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
            mixed = _corners_disagree(grid, level, ijk, label_points)

        leaf_levels.append(np.full(int((~mixed).sum()), level, dtype=np.uint8))
        leaf_ijk.append(ijk[~mixed])

        if mixed.any():
            children = 2 * ijk[mixed][:, None, :] + CORNER_OFFSETS[None, :, :]
            pending.setdefault(level + 1, []).append(children.reshape(-1, 3))

    return OctreeCells(level=np.concatenate(leaf_levels), ijk=np.concatenate(leaf_ijk).astype(np.int32))


def coarsen_cells(
        grid: OctreeGrid,
        cells: OctreeCells,
        label_points: PointLabeler,
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
        label_points: Labels for points; see :data:`PointLabeler`.

    Returns:
        The coarsened leaf cells.
    """
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
        lattice = (2 * candidates[:, None, :] + _PARENT_LATTICE_OFFSETS[None, :, :])
        labels = _labels_at_lattice(grid, child_level, lattice, label_points)
        uniform = np.all(labels == labels[:, :1, :], axis=(1, 2))
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


# The 27 lattice points of a cell, in units of its children's size.
_PARENT_LATTICE_OFFSETS: npt.NDArray[np.int64] = np.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=np.int64
)


def _corners_disagree(
        grid: OctreeGrid,
        level: int,
        ijk: npt.NDArray[np.int64],
        label_points: PointLabeler,
) -> npt.NDArray[np.bool_]:
    """For each cell, whether its 8 corners don't all get the same labels."""
    corners = ijk[:, None, :] + CORNER_OFFSETS[None, :, :]
    per_corner = _labels_at_lattice(grid, level, corners, label_points)
    return np.any(per_corner != per_corner[:, :1, :], axis=(1, 2))


def _labels_at_lattice(
        grid: OctreeGrid,
        level: int,
        lattice: npt.NDArray[np.int64],
        label_points: PointLabeler,
) -> npt.NDArray[np.integer]:
    """
    Labels at integer lattice points of a level, shape ``lattice.shape[:2] + (k,)``.

    Neighbouring cells share lattice points, so each distinct point is labelled once.
    """
    n_cells, n_points = lattice.shape[:2]
    unique_points, inverse = np.unique(lattice.reshape(-1, 3), axis=0, return_inverse=True)
    labels = np.asarray(label_points(grid.origin + unique_points * grid.cell_size(level)))
    if labels.ndim == 1:
        labels = labels[:, None]
    return labels[inverse.ravel()].reshape(n_cells, n_points, labels.shape[1])
