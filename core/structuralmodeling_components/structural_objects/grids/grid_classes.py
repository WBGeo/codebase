import numpy as np
from py_api_wbgeo.nodesapi import wbgeo_type
from pydantic.dataclasses import dataclass
from typing import Tuple


# Create a Grid class to handle grid creation and attributes
@wbgeo_type(name='Grid for discretization of a structural geological model',
            color='orange',
            identifier='RegularGrid')
@dataclass(config={"arbitrary_types_allowed": True})
class RegularGrid:
    extent: Tuple[float, float, float, float, float, float]
    resolution: Tuple[int, int, int]

    def __post_init__(self):
        dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
        dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
        dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        self.spacing = (dx, dy, dz)

        self.gridx = np.linspace(
            self.extent[0] + dx / 2,
            self.extent[1] - dx / 2,
            self.resolution[0],
        )
        self.gridy = np.linspace(
            self.extent[2] + dy / 2,
            self.extent[3] - dy / 2,
            self.resolution[1],
        )
        self.gridz = np.linspace(
            self.extent[4] + dz / 2,
            self.extent[5] - dz / 2,
            self.resolution[2],
        )

        coords = self.gridx, self.gridy, self.gridz
        self.g = np.meshgrid(*coords, indexing="ij")
        self.grid_coordinates = np.vstack(
            tuple(map(np.ravel, self.g))
        ).T.astype("float64")

    def xyz_to_indices(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert (X, Y, Z) world coordinates to voxel indices (i, j, k).

        Args:
            coords (np.ndarray): Array of shape (N, 3) with [X, Y, Z] coordinates.

        Returns:
            np.ndarray: Array of shape (N, 3) with [i, j, k] voxel indices.
        """
        if not hasattr(self, "spacing"):
            raise AttributeError("Grid must have 'origin' and 'spacing' attributes defined.")

        relative = coords - np.array((self.extent[0], self.extent[2], self.extent[4]))
        indices = np.floor_divide(relative, self.spacing).astype(int)
        return indices
