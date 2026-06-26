import unittest
import numpy as np
from core.meshing_components.explicit.structured.mesh_data import (
    create_hexahedral_elements_with_nodes
)

class TestCreateHexahedralElements(unittest.TestCase):

    def test_simple_2x2x2_grid(self):
        """
        Smallest meaningful structured grid:
        2x2 nodes per layer, 2 layers → 1 hex element
        """

        n_gx, n_gy = 2, 2

        # 2 layers × (3*n_gx*n_gy + 1)
        # simplify: just fill required structure

        layer0 = np.array([
            0, 1, 2, 3,   # x
            0, 0, 1, 1,   # y
            10, 10, 10, 10,  # z
            0, 0, 0, 0    # surface id
        ])

        layer1 = np.array([
            0, 1, 2, 3,
            0, 0, 1, 1,
            20, 20, 20, 20,
            1, 1, 1, 1
        ])

        adjusted = np.vstack([layer0, layer1])

        elements, nodes = create_hexahedral_elements_with_nodes(
            adjusted, n_gx, n_gy
        )

        # basic shape checks
        self.assertEqual(elements.shape[1], 10)  # 8 nodes + elem id + surf id
        self.assertEqual(nodes.shape[1], 4)      # node_id removed surface column

        # only 1 element expected
        self.assertEqual(len(elements), 1)

        # element should contain 8 nodes
        self.assertEqual(len(elements[0]) - 2, 8)  # exclude element_id + surface_id

        # node uniqueness (no duplicates expected)
        node_ids = elements[0][1:-1]
        self.assertEqual(len(node_ids), len(set(node_ids)))

        # nodes should be renumbered starting from 0
        self.assertEqual(nodes[:, 0].min(), 0)
        self.assertEqual(nodes[:, 0].max(), len(nodes) - 1)

    def test_multiple_layers(self):
        """
        Check that multiple layers generate multiple elements
        """

        n_gx, n_gy = 2, 2

        layer0 = np.zeros(4 * 4)
        layer1 = np.ones(4 * 4)
        layer2 = np.ones(4 * 4) * 2

        adjusted = np.vstack([layer0, layer1, layer2])

        elements, nodes = create_hexahedral_elements_with_nodes(
            adjusted, n_gx, n_gy
        )

        # should create 2 vertical elements
        self.assertEqual(len(elements), 2)

        # nodes must exist
        self.assertGreater(len(nodes), 0)

#############################################
if __name__ == "__main__":
    unittest.main()
