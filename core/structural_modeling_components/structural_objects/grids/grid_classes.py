"""
Regular grid utilities for discretizing a structural geological model.

This module defines :class:`RegularGrid`, a (pydantic) dataclass that computes
cell-centered coordinate vectors and a flattened list of grid point coordinates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from pydantic.dataclasses import dataclass
from py_api_wbgeo.nodesapi import wbgeo_type

# Type aliases to make intent explicit and keep annotations readable.
Extent6 = Tuple[float, float, float, float, float, float]
Resolution3 = Tuple[int, int, int]


@wbgeo_type(
    name="Grid for discretization of a structural geological model",
    color="orange",
    identifier="RegularGrid",
)
@dataclass(config={"arbitrary_types_allowed": True})
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
