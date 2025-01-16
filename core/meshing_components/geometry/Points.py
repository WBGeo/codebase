import numpy as np
class Points:
    def __init__(self, node_array):
        """
        Initializes the Points class.

        Args:
            node_array (np.ndarray): Array of coordinates with columns [x, y, z].
        """
        self.coordinates = node_array[:, 1:-1].astype(float)
        self.point_id = node_array[:, 0].astype(int)
        self.surface_id = node_array[:, -1].astype(int)

    def get_coordinates(self):
        """Returns the x, y, z coordinates."""
        return self.coordinates

    def get_point_id(self):
        """Returns the point ids."""
        return self.point_id

    def get_surface_id(self):
        """Returns the surface ids."""
        return self.surface_id
