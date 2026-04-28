import unittest
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data  import (
    validate_triangulation
)


class TestValidateTriangulation(unittest.TestCase):

    def setUp(self):
        self.extent = (0, 10, 0, 10, 0, 10)

    # -------------------------
    # VALID CASE (returns None)
    # -------------------------
    def test_valid_points(self):
        pts = np.array([
            [1, 1, 1],
            [5, 5, 5],
            [9, 9, 9]
        ])

        result = validate_triangulation(pts, self.extent)
        self.assertIsNone(result)   # ✅ FIX

    # -------------------------
    # INVALID TYPE
    # -------------------------
    def test_invalid_type(self):
        pts = [[1, 1, 1], [2, 2, 2]]

        with self.assertRaises(TypeError):
            validate_triangulation(pts, self.extent)

    # -------------------------
    # INVALID SHAPE
    # -------------------------
    def test_invalid_shape(self):
        pts = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            validate_triangulation(pts, self.extent)

    def test_invalid_shape_2(self):
        pts = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            validate_triangulation(pts, self.extent)

    # -------------------------
    # POINTS OUTSIDE (RAISE)
    # -------------------------
    def test_points_outside_raise(self):
        pts = np.array([
            [1, 1, 1],
            [20, 5, 5],
            [5, 5, 5]
        ])

        with self.assertRaises(ValueError):
            validate_triangulation(pts, self.extent, raise_error=True)

    # -------------------------
    # POINTS OUTSIDE (FILTER)
    # -------------------------
    def test_points_outside_filter(self):
        pts = np.array([
            [1, 1, 1],
            [20, 5, 5],
            [5, 5, 5]
        ])

        filtered = validate_triangulation(pts, self.extent, raise_error=False)

        expected = np.array([
            [1, 1, 1],
            [5, 5, 5]
        ])

        self.assertTrue(np.array_equal(filtered, expected))

    # -------------------------
    # ALL POINTS OUTSIDE
    # -------------------------
    def test_all_points_outside(self):
        pts = np.array([
            [20, 20, 20],
            [30, 30, 30]
        ])

        filtered = validate_triangulation(pts, self.extent, raise_error=False)

        self.assertEqual(len(filtered), 0)

    # -------------------------
    # POINTS ON BOUNDARY (VALID)
    # -------------------------
    def test_points_on_boundary(self):
        pts = np.array([
            [0, 0, 0],
            [10, 10, 10],
            [0, 10, 5]
        ])

        result = validate_triangulation(pts, self.extent)
        self.assertIsNone(result)   # ✅ FIX


if __name__ == "__main__":
    unittest.main()
