import unittest
import numpy as np
from core.meshing_components.geometry.Points import Points

class TestPoints(unittest.TestCase):

    def setUp(self):

        # format: [id, x, y, z]
        self.node_array = np.array([
            [1, 10.0, 20.0, 30.0],
            [2, 11.0, 21.0, 31.0],
            [3, 12.0, 22.0, 32.0],
        ], dtype=float)

        self.points = Points(self.node_array)

    def test_coordinates_shape(self):

        coords = self.points.get_coordinates()

        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(coords.shape, (3, 3))

    def test_point_ids(self):

        ids = self.points.get_point_id()

        self.assertIsInstance(ids, np.ndarray)
        self.assertEqual(ids.shape, (3,))
        self.assertTrue(np.array_equal(ids, np.array([1, 2, 3])))

    def test_coordinate_values(self):

        coords = self.points.get_coordinates()

        expected = np.array([
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
        ])

        np.testing.assert_allclose(coords, expected)

    def test_single_point(self):

        node_array = np.array([[7, 1.0, 2.0, 3.0]], dtype=float)

        p = Points(node_array)

        self.assertEqual(p.get_coordinates().shape, (1, 3))
        self.assertEqual(p.get_point_id().shape, (1,))
        self.assertEqual(p.get_point_id()[0], 7)

    def test_dtype_conversion(self):

        node_array = np.array([
            [1, 1, 2, 3],
            [2, 4, 5, 6],
        ], dtype=float)

        p = Points(node_array)

        self.assertEqual(p.get_coordinates().dtype, float)
        self.assertTrue(np.issubdtype(p.get_point_id().dtype, np.integer))

########################################
if __name__ == "__main__":
    unittest.main()
