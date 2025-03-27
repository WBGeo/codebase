import numpy as np


# Create a Grid class to handle grid creation and attributes
class RegularGrid:
    def __init__(self, extent, resolution):
        self.extent = extent
        self.resolution = resolution
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]
        self.spacing = (dx, dy, dz)
        self.gridx = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0])
        self.gridy = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1])
        self.gridz = np.linspace(extent[4] + dz / 2, extent[5] - dz, resolution[2])
        coords = self.gridx, self.gridy, self.gridz
        self.g = np.meshgrid(*coords, indexing="ij")
        self.grid_coordinates = np.vstack(tuple(map(np.ravel, self.g))).T.astype("float64")
