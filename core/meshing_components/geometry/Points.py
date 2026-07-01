import numpy as np
from numpy.typing import NDArray

class Points:

    def __init__(self, node_array):
        """
        Initializes the Points class.

        Args:
            node_array (np.ndarray): Array of coordinates with columns [x, y, z].
        """

        self.coordinates = node_array[:, 1:].astype(float)
        self.point_id = node_array[:, 0].astype(int)

    def get_coordinates(self) -> NDArray[np.float64]:
        """Returns the x, y, z coordinates."""
        return self.coordinates

    def get_point_id(self) -> NDArray[np.int64]:
        """Returns the point ids."""
        return self.point_id


