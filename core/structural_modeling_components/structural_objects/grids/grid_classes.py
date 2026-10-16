"""
Grid utilities for discretizing a structural geological model.

This module defines :class:`RegularGrid`, a (pydantic) dataclass that computes
cell-centered coordinate vectors and a flattened list of grid point coordinates,
:class:`OctreeGrid`, the specification of an adaptive octree grid together
with the geometry of its cell addresses, and :class:`OctreeCells`, the cells an
octree ends up with after refinement.

Per-cell results follow one layout rule for every grid type: on a
:class:`RegularGrid` they are arrays of shape ``resolution``; on an
:class:`OctreeGrid` they are 1D arrays aligned with the :class:`OctreeCells`
stored next to them. :func:`cell_centers` and :func:`to_regular` work for both.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, model_validator
from pydantic.dataclasses import dataclass
from pydantic_numpy import NpNDArrayInt32, NpNDArrayUint8
from py_api_wbgeo.nodesapi import wbgeo_type

logger = logging.getLogger(__name__)

# Type aliases to make intent explicit and keep annotations readable.
Extent6 = Tuple[float, float, float, float, float, float]
Resolution3 = Tuple[int, int, int]


@wbgeo_type(
    name="Grid for discretization of a structural geological model",
    color="#b0dfa9",
    identifier="RegularGrid",
)
@dataclass
class RegularGrid:
    """
    Regular (axis-aligned) grid defined by an extent and a 3D resolution.

    The grid is *cell-centered*: the coordinate vectors (x/y/z) are generated so
    that points lie at the centers of the voxels.

    Attributes:
        extent: Axis-aligned bounding box as
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        resolution: Number of cells in each direction as ``(nx, ny, nz)``.

    Computed (created in ``__post_init__``):
        spacing: Cell size as ``(dx, dy, dz)``.
        gridx: 1D array of x cell-center coordinates, shape ``(nx,)``.
        gridy: 1D array of y cell-center coordinates, shape ``(ny,)``.
        gridz: 1D array of z cell-center coordinates, shape ``(nz,)``.
        g: 3D meshgrid tuple as returned by ``np.meshgrid(..., indexing="ij")``.
        grid_coordinates: Flattened coordinates of all grid points,
            shape ``(nx*ny*nz, 3)`` with dtype ``float64``.
    """

    extent: Extent6
    resolution: Resolution3

    def __post_init__(self) -> None:
        """
        Compute spacing, coordinate vectors, meshgrid, and flattened coordinates.

        Notes:
            - Uses cell centers (half-cell offset at each boundary).
            - The flattened coordinate array is ordered according to ``indexing="ij"``.
        """
        if any(self.extent[2 * i] >= self.extent[2 * i + 1] for i in range(3)):
            raise ValueError(
                f"Invalid grid extent: each min must be strictly less than max. Got {self.extent}"
            )
        if any(r <= 0 for r in self.resolution):
            raise ValueError(
                f"Invalid grid resolution: all dimensions must be positive integers. Got {self.resolution}"
            )

        dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
        dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
        dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        self.spacing: Tuple[float, float, float] = (dx, dy, dz)

        # Cell-centered coordinates along each axis.
        self.gridx: npt.NDArray[np.float64] = np.linspace(
            self.extent[0] + dx / 2,
            self.extent[1] - dx / 2,
            self.resolution[0],
        )
        self.gridy: npt.NDArray[np.float64] = np.linspace(
            self.extent[2] + dy / 2,
            self.extent[3] - dy / 2,
            self.resolution[1],
        )
        self.gridz: npt.NDArray[np.float64] = np.linspace(
            self.extent[4] + dz / 2,
            self.extent[5] - dz / 2,
            self.resolution[2],
        )

        coords = self.gridx, self.gridy, self.gridz
        self.g: Tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ] = np.meshgrid(*coords, indexing="ij")

        # Flatten meshgrid into an (N, 3) coordinate array with float64 dtype.
        self.grid_coordinates: npt.NDArray[np.float64] = np.vstack(
            tuple(map(np.ravel, self.g))
        ).T.astype("float64")

    def xyz_to_indices(self, coords: npt.NDArray[np.floating]) -> npt.NDArray[np.int_]:
        """
        Convert (X, Y, Z) world coordinates to voxel indices (i, j, k).

        Args:
            coords: Array of shape ``(N, 3)`` containing ``[X, Y, Z]`` coordinates.

        Returns:
            Array of shape ``(N, 3)`` containing ``[i, j, k]`` voxel indices.

        Raises:
            AttributeError: If spacing has not been computed (i.e., ``__post_init__``
                has not run / the instance is not fully initialized).
        """
        # spacing is computed during __post_init__; guard for partially-initialized objects.
        if not hasattr(self, "spacing"):
            raise AttributeError(
                "Grid must have 'origin' and 'spacing' attributes defined."
            )

        relative = coords - np.array((self.extent[0], self.extent[2], self.extent[4]))
        indices = np.floor_divide(relative, self.spacing).astype(int)
        return indices


# Highest refinement level an OctreeGrid accepts. At level 10 every base block
# holds 1024^3 finest cells, far beyond what the structural modeling step can
# evaluate, so larger values are almost certainly input mistakes.
MAX_OCTREE_LEVEL = 10

# Base cells whose longest side exceeds this multiple of their shortest side
# trigger a warning: dual contouring and the refinement tests work best on
# near-cubic cells.
MAX_CELL_ASPECT_RATIO = 2.0

# Offsets of the 8 corners of a cell in units of its size, shape (8, 3).
# Corner ``c`` has bit 0 for x, bit 1 for y and bit 2 for z:
#   idx  x    y    z        idx  x    y    z
#    0   min  min  min       4   min  min  max
#    1   max  min  min       5   max  min  max
#    2   min  max  min       6   min  max  max
#    3   max  max  min       7   max  max  max
CORNER_OFFSETS: npt.NDArray[np.int64] = np.array(
    [[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], dtype=np.int64
)

Level = Union[int, npt.NDArray[np.integer]]


class GapMitigation(StrEnum):
    """
    Strategy against gaps in dual-contouring surfaces on an adaptive octree.

    Gaps appear where a surface crosses cells at different refinement depths.

    A ``StrEnum`` because the Workbench runner deserializes dropdown values of
    plain ``Enum`` types by member name, but those of ``StrEnum`` types by value.

    Members:
        BALANCE_MESH_DEPTH_ADJACENCY: After refinement, split every leaf that
            touches a max-level leaf face-to-face down to max level as well
            (one pass). Closes the gaps at the cost of extra cells.
        NONE: No mitigation; surfaces may contain gaps.
    """

    BALANCE_MESH_DEPTH_ADJACENCY = "balance_mesh_depth_adjacency"
    NONE = "none"


class CellBalance(StrEnum):
    """
    Grading of the stored octree cells: how much neighbouring cells may differ in level.

    A ``StrEnum`` for the same reason as :class:`GapMitigation`.

    Members:
        NONE: No constraint; a max-level cell can sit next to a much coarser one.
        FACE_2TO1: Cells that share a face differ by at most one level (2:1
            balance). Only adds cells; every feature stays at max level.
    """

    NONE = "none"
    FACE_2TO1 = "face_2to1"


@wbgeo_type(
    name="Octree grid for adaptive discretization of a structural geological model",
    color="#7fc7a3",
    identifier="OctreeGrid",
)
@dataclass
class OctreeGrid:
    """
    Specification of an adaptive octree grid over an axis-aligned extent.

    The extent is divided into a regular grid of base blocks (level 0), each of
    which is the root of an octree. A cell at level ``L`` is addressed by
    ``(L, ix, iy, iz)`` with ``0 <= ix < base_resolution[0] * 2**L`` (same for y
    and z). Refinement stops at ``max_level``, so the finest cells match those of
    a :class:`RegularGrid` with resolution ``base_resolution * 2**max_level``,
    see :meth:`equivalent_regular_grid`.

    This object only describes the grid. Which cells end up as leaves is decided
    during modeling and stored in the results, not here.

    Attributes:
        extent: Axis-aligned bounding box as
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        base_resolution: Number of base blocks along each axis as
            ``(bx, by, bz)``. These are the coarsest cells the grid can have;
            choose them so that the blocks are close to cubic.
        max_level: Number of times a base block can be halved along every axis.
            ``0`` makes the grid a plain regular grid of base blocks.
        gap_mitigation: How gaps in dual-contouring surfaces are avoided.
        cell_balance: Grading of the stored cells (fault and structural
            results); see :class:`CellBalance`. Off by default.

    Computed (created in ``__post_init__``):
        origin: Lower corner of the extent as ``(xmin, ymin, zmin)``.
        base_spacing: Size of a base block as ``(dx, dy, dz)``.
        finest_resolution: Number of max-level cells along each axis,
            ``base_resolution * 2**max_level``.
        finest_spacing: Size of a max-level cell as ``(dx, dy, dz)``.
    """

    extent: Extent6
    base_resolution: Resolution3
    max_level: int = 4
    gap_mitigation: GapMitigation = GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY
    cell_balance: CellBalance = CellBalance.NONE

    def __post_init__(self) -> None:
        """
        Validate the inputs and compute the derived geometry.

        Raises:
            ValueError: If the extent is empty along an axis, a base resolution
                entry is not positive, or ``max_level`` is outside
                ``[0, MAX_OCTREE_LEVEL]``.
        """
        if any(self.extent[2 * i] >= self.extent[2 * i + 1] for i in range(3)):
            raise ValueError(
                f"Invalid grid extent: each min must be strictly less than max. Got {self.extent}"
            )
        if any(b <= 0 for b in self.base_resolution):
            raise ValueError(
                f"Invalid base resolution: all dimensions must be positive integers. Got {self.base_resolution}"
            )
        if not 0 <= self.max_level <= MAX_OCTREE_LEVEL:
            raise ValueError(
                f"Invalid max_level: must be between 0 and {MAX_OCTREE_LEVEL}. Got {self.max_level}"
            )

        self.origin: npt.NDArray[np.float64] = np.array(self.extent[0::2], dtype=np.float64)
        lengths = np.array(self.extent[1::2], dtype=np.float64) - self.origin
        base = np.array(self.base_resolution, dtype=np.int64)
        self.base_spacing: npt.NDArray[np.float64] = lengths / base
        self.finest_resolution: Resolution3 = tuple(int(n) for n in base * 2 ** self.max_level)
        self.finest_spacing: npt.NDArray[np.float64] = self.base_spacing / 2 ** self.max_level

        aspect_ratio = self.base_spacing.max() / self.base_spacing.min()
        if aspect_ratio > MAX_CELL_ASPECT_RATIO:
            # Cubes with the shortest side keep the resolution the user chose on
            # the finest axis.
            suggestion = tuple(int(n) for n in np.maximum(1, np.rint(lengths / self.base_spacing.min())))
            logger.warning(
                "OctreeGrid base blocks are stretched: their sides are %s, a ratio of %.1f. "
                "Cells keep this shape at every level, which makes surfaces less accurate. "
                "base_resolution=%s would give near-cubic blocks.",
                tuple(round(float(s), 6) for s in self.base_spacing), aspect_ratio, suggestion,
            )

    def cells_per_axis(self, level: int) -> npt.NDArray[np.int64]:
        """
        Number of cells along each axis at a refinement level.

        Args:
            level: Refinement level in ``[0, max_level]``.

        Returns:
            Array of shape ``(3,)``: ``base_resolution * 2**level``.
        """
        self._check_level(level)
        return np.array(self.base_resolution, dtype=np.int64) * 2 ** int(level)

    def cell_size(self, level: Level) -> npt.NDArray[np.float64]:
        """
        Size of the cells at one or several refinement levels.

        Args:
            level: A level, or an array of shape ``(N,)`` with one level per cell.

        Returns:
            Shape ``(3,)`` for a single level, ``(N, 3)`` for an array of levels.
        """
        levels = self._check_level(level)
        return self.base_spacing / (2.0 ** levels)[..., None]

    def cell_bounds(
        self, level: Level, ijk: npt.NDArray[np.integer]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Lower and upper corners of cells.

        Args:
            level: A level shared by all cells, or an array of shape ``(N,)``.
            ijk: Integer cell indices at that level, shape ``(N, 3)``.

        Returns:
            Tuple ``(mins, maxs)``, each of shape ``(N, 3)``.
        """
        size = self.cell_size(level)
        mins = self.origin + np.asarray(ijk) * size
        return mins, mins + size

    def cell_centers(self, level: Level, ijk: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
        """
        Centers of cells.

        Args:
            level: A level shared by all cells, or an array of shape ``(N,)``.
            ijk: Integer cell indices at that level, shape ``(N, 3)``.

        Returns:
            Array of shape ``(N, 3)``.
        """
        size = self.cell_size(level)
        return self.origin + (np.asarray(ijk) + 0.5) * size

    def cell_corners(self, level: Level, ijk: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
        """
        The 8 corners of cells, ordered as in :data:`CORNER_OFFSETS`.

        Args:
            level: A level shared by all cells, or an array of shape ``(N,)``.
            ijk: Integer cell indices at that level, shape ``(N, 3)``.

        Returns:
            Array of shape ``(N, 8, 3)``.
        """
        size = self.cell_size(level)
        mins = self.origin + np.asarray(ijk) * size
        return mins[:, None, :] + CORNER_OFFSETS[None, :, :] * size[..., None, :]

    def locate(self, xyz: npt.NDArray[np.floating], level: int) -> npt.NDArray[np.int64]:
        """
        Indices of the cells at a refinement level that contain the given points.

        Points on a shared face go to the upper cell. Points outside the extent
        are clamped to the nearest boundary cell.

        Args:
            xyz: Point coordinates, shape ``(N, 3)``.
            level: Refinement level in ``[0, max_level]``.

        Returns:
            Integer indices of shape ``(N, 3)``.
        """
        n_cells = self.cells_per_axis(level)
        indices = np.floor((np.asarray(xyz) - self.origin) / self.cell_size(level)).astype(np.int64)
        return np.clip(indices, 0, n_cells - 1)

    def equivalent_regular_grid(self) -> RegularGrid:
        """
        The regular grid made of all max-level cells.

        Its cell centers coincide with the centers of the max-level octree cells.

        Returns:
            A :class:`RegularGrid` over the same extent with
            ``resolution = finest_resolution``.
        """
        return RegularGrid(extent=self.extent, resolution=self.finest_resolution)

    def _check_level(self, level: Level) -> npt.NDArray[np.int64]:
        """Return ``level`` as an integer array, raising if it is outside ``[0, max_level]``."""
        levels = np.asarray(level, dtype=np.int64)
        if np.any(levels < 0) or np.any(levels > self.max_level):
            raise ValueError(
                f"Octree level must be between 0 and max_level={self.max_level}. Got {level}"
            )
        return levels


ModelGrid = Union[RegularGrid, OctreeGrid]


class OctreeCells(BaseModel):
    """
    The cells an :class:`OctreeGrid` ended up with after adaptive refinement.

    Cell ``n`` is the octree cell ``(level[n], *ijk[n])`` of the grid it was
    built on. Together the cells tile the grid's extent without overlap. Every
    per-cell result array on the octree path has shape ``(len(cells),)`` and is
    aligned with this order.

    Attributes:
        level: Refinement level of each cell, shape ``(N,)``.
        ijk: Integer cell indices at that level, shape ``(N, 3)``.
    """

    level: NpNDArrayUint8
    ijk: NpNDArrayInt32

    @model_validator(mode="after")
    def _check_shapes(self) -> "OctreeCells":
        if self.level.ndim != 1 or self.ijk.shape != (self.level.shape[0], 3):
            raise ValueError(
                f"OctreeCells needs level of shape (N,) and ijk of shape (N, 3). "
                f"Got {self.level.shape} and {self.ijk.shape}"
            )
        return self

    def __len__(self) -> int:
        return int(self.level.shape[0])

    def centers(self, grid: OctreeGrid) -> npt.NDArray[np.float64]:
        """Cell centers, shape ``(N, 3)``."""
        return grid.cell_centers(self.level, self.ijk)

    def locate(self, grid: OctreeGrid, xyz: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
        """
        Index of the cell that contains each point.

        Points outside the extent are clamped to the nearest boundary cell, as in
        :meth:`OctreeGrid.locate`.

        Args:
            grid: The grid the cells were built on.
            xyz: Point coordinates, shape ``(M, 3)``.

        Returns:
            Cell indices into this object, shape ``(M,)``; ``-1`` where no cell
            contains the point, which only happens if the cells don't tile the
            extent.
        """
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        found = np.full(points.shape[0], -1, dtype=np.int64)
        levels = self.level.astype(np.int64)
        for level in np.unique(levels):
            members = np.flatnonzero(levels == level)
            n_cells = grid.cells_per_axis(int(level))
            keys = linear_cell_index(self.ijk[members], n_cells)
            order = np.argsort(keys)
            sorted_keys = keys[order]
            point_keys = linear_cell_index(grid.locate(points, int(level)), n_cells)
            pos = np.minimum(np.searchsorted(sorted_keys, point_keys), sorted_keys.size - 1)
            hit = sorted_keys[pos] == point_keys
            found[hit] = members[order[pos[hit]]]
        return found


def linear_cell_index(ijk: npt.NDArray[np.integer], n_cells: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Row-major linear index of cell indices within a ``n_cells`` block."""
    ijk = np.asarray(ijk, dtype=np.int64)
    return (ijk[:, 0] * n_cells[1] + ijk[:, 1]) * n_cells[2] + ijk[:, 2]


def cell_array_shape(grid: ModelGrid, cells: Optional[OctreeCells]) -> Tuple[int, ...]:
    """
    Shape of a per-cell result array for a grid.

    Args:
        grid: The model grid.
        cells: The realized cells; required for an :class:`OctreeGrid`, must be
            ``None`` for a :class:`RegularGrid`.

    Returns:
        ``resolution`` for a regular grid, ``(len(cells),)`` for an octree.
    """
    if isinstance(grid, OctreeGrid):
        if cells is None:
            raise ValueError("An OctreeGrid result needs its cells to interpret per-cell arrays.")
        return (len(cells),)
    if cells is not None:
        raise ValueError("A RegularGrid result must not carry octree cells.")
    return tuple(int(n) for n in grid.resolution)


def cell_centers(grid: ModelGrid, cells: Optional[OctreeCells]) -> npt.NDArray[np.float64]:
    """
    Center of every cell, in the order of a per-cell array's ``ravel()``.

    Args:
        grid: The model grid.
        cells: The realized cells for an :class:`OctreeGrid`, else ``None``.

    Returns:
        Array of shape ``(n_cells, 3)``: ``grid_coordinates`` for a regular grid,
        the cell centers for an octree.
    """
    cell_array_shape(grid, cells)
    if isinstance(grid, OctreeGrid):
        return cells.centers(grid)
    return grid.grid_coordinates


def to_regular(
        grid: ModelGrid,
        cells: Optional[OctreeCells],
        values: npt.NDArray[np.generic],
        resolution: Optional[Resolution3] = None,
) -> npt.NDArray[np.generic]:
    """
    Resample a per-cell array onto a regular grid over the same extent.

    Every cell of the target grid takes the value of the cell that contains its
    center. For an octree this is exact wherever the octree cells are at least as
    fine as the target cells, i.e. everywhere at the default resolution, since
    each octree cell then covers whole target cells.

    Args:
        grid: The grid ``values`` belongs to.
        cells: The realized cells for an :class:`OctreeGrid`, else ``None``.
        values: Per-cell array with shape ``cell_array_shape(grid, cells)``.
        resolution: Target resolution. Defaults to the grid's own resolution for
            a regular grid and to ``finest_resolution`` for an octree.

    Returns:
        Array of shape ``resolution``. For a regular grid at its own
        resolution, ``values`` itself is returned.
    """
    values = np.asarray(values)
    expected = cell_array_shape(grid, cells)
    if values.shape != expected:
        raise ValueError(f"Per-cell array has shape {values.shape}, expected {expected}.")

    if isinstance(grid, OctreeGrid):
        target_resolution = tuple(resolution) if resolution is not None else grid.finest_resolution
        target = RegularGrid(extent=grid.extent, resolution=target_resolution)
        index = cells.locate(grid, target.grid_coordinates)
        if np.any(index < 0):
            raise ValueError("The octree cells do not cover the whole extent.")
        return values[index].reshape(target_resolution)

    if resolution is None or tuple(resolution) == tuple(grid.resolution):
        return values
    target = RegularGrid(extent=grid.extent, resolution=tuple(resolution))
    ijk = np.clip(grid.xyz_to_indices(target.grid_coordinates), 0, np.array(grid.resolution) - 1)
    return values[ijk[:, 0], ijk[:, 1], ijk[:, 2]].reshape(tuple(resolution))
