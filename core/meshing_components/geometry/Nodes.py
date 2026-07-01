import numpy as np
from numpy.typing import NDArray
from typing import Dict
from core.meshing_components.geometry.Points import Points

class Nodes(Points):
    def __init__(self, node_array):
        """
        Initializes the Nodes class.

        Args:
            node_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
        """
        # Call the parent class (Points) to initialize common attributes

        super().__init__(node_array)

        # Node-specific attributes (getting from Point class)
        self.node_ids = self.point_id.astype(int)
        self.x_coords = self.coordinates[:, 0]
        self.y_coords = self.coordinates[:, 1]
        self.z_coords = self.coordinates[:, 2]

    def get_node_ids(self) -> NDArray[np.int64]:
        """Returns the node IDs."""
        return self.node_ids

    def total_nodes(self) -> int:
        """Returns the total number of nodes."""
        return len(self.node_ids)

    def nodes_on_boundaries(self) -> Dict[str, NDArray[np.int64]]:
        """Returns a dictionary categorizing nodes on each boundary."""

        boundaries = {}

        min_y, max_y = np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1])
        min_z, max_z = np.min(self.coordinates[:, 2]), np.max(self.coordinates[:, 2])
        min_x, max_x = np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0])

        boundaries['front'] = self.node_ids[self.coordinates[:, 1] == min_y]
        boundaries['back'] = self.node_ids[self.coordinates[:, 1] == max_y]
        boundaries['bottom'] = self.node_ids[self.coordinates[:, 2] == min_z]
        boundaries['top'] = self.node_ids[self.coordinates[:, 2] == max_z]
        boundaries['left'] = self.node_ids[self.coordinates[:, 0] == min_x]
        boundaries['right'] = self.node_ids[self.coordinates[:, 0] == max_x]

        return boundaries

