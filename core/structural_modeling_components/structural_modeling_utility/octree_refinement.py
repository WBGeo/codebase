"""
Adaptive refinement of an :class:`OctreeGrid`.

Starting from the base blocks (or from given cells), every cell whose 8 corners
get different labels is split into its 8 children, level by level, until
``max_level``. Cells whose corners all agree become leaves. The result is the
set of realized cells (:class:`OctreeCells`) that per-cell results are stored on.

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


def _corners_disagree(
        grid: OctreeGrid,
        level: int,
        ijk: npt.NDArray[np.int64],
        label_points: PointLabeler,
) -> npt.NDArray[np.bool_]:
    """For each cell, whether its 8 corners don't all get the same labels."""
    # Corners as integer lattice points of this level; neighbouring cells share
    # them, so each distinct corner is labelled once.
    corners = (ijk[:, None, :] + CORNER_OFFSETS[None, :, :]).reshape(-1, 3)
    unique_corners, inverse = np.unique(corners, axis=0, return_inverse=True)
    points = grid.origin + unique_corners * grid.cell_size(level)

    labels = np.asarray(label_points(points))
    if labels.ndim == 1:
        labels = labels[:, None]
    per_corner = labels[inverse.ravel()].reshape(ijk.shape[0], 8, labels.shape[1])
    return np.any(per_corner != per_corner[:, :1, :], axis=(1, 2))
