import unittest
import numpy as np

from core.meshing_components.geometry.Nodes import Nodes


class TestNodes(unittest.TestCase):

    def setUp(self):
        """
        node format:
        [node_id, x, y, z, surface_id]
        """
        self.node_array = np.array([
            [1, 10.0, 20.0, 30.0, 100],
            [2, 11.0, 21.0, 31.0, 100],
            [3, 12.0, 22.0, 32.0, 101],
            [4, 13.0, 23.0, 33.0, 101],
            [5, 14.0, 24.0, 34.0, 102],
            [6, 15.0, 25.0, 35.0, 102],
            [7, 16.0, 26.0, 36.0, 103],
            [8, 17.0, 27.0, 37.0, 103],
        ])

        self.nodes = Nodes(self.node_array)

    # ---------------------------------------------------
    def test_coordinate_shape(self):
        coords = self.nodes.get_coordinates()

        # IMPORTANT: current implementation keeps 4 columns (x,y,z,surface_id)
        self.assertEqual(coords.shape, (8, 4))

    # ---------------------------------------------------
    def test_coordinate_values_are_correct(self):
        coords = self.nodes.get_coordinates()

        expected = self.node_array[:, 1:].astype(float)
        np.testing.assert_allclose(coords, expected)

    # ---------------------------------------------------
    def test_node_ids(self):
        ids = self.nodes.get_node_ids()

        self.assertEqual(ids.shape, (8,))
        self.assertTrue(np.array_equal(ids, np.arange(1, 9)))

    # ---------------------------------------------------
    def test_total_nodes(self):
        self.assertEqual(self.nodes.total_nodes(), 8)

    # ---------------------------------------------------
    def test_surface_column_exists_in_coordinates(self):
        coords = self.nodes.get_coordinates()

        # last column is surface_id in current design
        self.assertTrue(np.all(coords[:, -1] >= 100))

    # ---------------------------------------------------
    def test_boundary_classification_logic_exists(self):
        boundaries = self.nodes.nodes_on_boundaries()

        self.assertIn("front", boundaries)
        self.assertIn("back", boundaries)
        self.assertIn("left", boundaries)
        self.assertIn("right", boundaries)
        self.assertIn("top", boundaries)
        self.assertIn("bottom", boundaries)

        # sanity checks
        self.assertTrue(len(boundaries["left"]) > 0)
        self.assertTrue(len(boundaries["right"]) > 0)


if __name__ == "__main__":
    unittest.main()
