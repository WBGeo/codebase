import unittest
import numpy as np
from scipy.spatial import cKDTree
from core.meshing_components.explicit.structured.mesh_data import (
    resample_preserve_z_nearest
)

class TestResamplePreserveZNearest(unittest.TestCase):

    # Basic shape test
    def test_output_shape(self):
        points = np.array([
            [0, 0, 1],
            [10, 0, 2],
            [0, 10, 3],
            [10, 10, 4],
        ])

        extent = [0, 10, 0, 10, 0, 10]
        nx, ny = 5, 4

        result = resample_preserve_z_nearest(points, nx, ny, extent)

        self.assertEqual(result.shape, (nx * ny, 3))

    # Empty input should fail
    def test_empty_input_raises(self):
        points = np.empty((0, 3))
        extent = [0, 10, 0, 10, 0, 10]

        with self.assertRaises(ValueError):
            resample_preserve_z_nearest(points, 3, 3, extent)

    # Exact nearest-neighbor mapping (core test)
    def test_exact_nearest_mapping(self):
        """
        Grid points match input exactly → deterministic result
        """

        points = np.array([
            [0, 0, 10],
            [10, 0, 20],
            [0, 10, 30],
            [10, 10, 40],
        ])

        extent = [0, 10, 0, 10, 0, 50]
        nx, ny = 2, 2

        result = resample_preserve_z_nearest(points, nx, ny, extent)

        expected_xy = np.array([
            [0, 0],
            [10, 0],
            [0, 10],
            [10, 10],
        ])

        expected_z = np.array([10, 20, 30, 40])

        # XY must match grid exactly
        np.testing.assert_allclose(result[:, :2], expected_xy)

        # Z must come from nearest points
        np.testing.assert_allclose(result[:, 2], expected_z)

    # Nearest-neighbor correctness using brute-force check
    def test_nearest_neighbor_correctness(self):
        """
        Compare KDTree result against manual brute-force check
        """

        np.random.seed(0)

        points = np.random.rand(20, 3)
        points[:, 2] *= 100  # make Z distinct

        extent = [0, 1, 0, 1, 0, 100]

        nx, ny = 6, 5

        result = resample_preserve_z_nearest(points, nx, ny, extent)

        grid_xy = result[:, :2]

        # manual nearest neighbor
        tree = cKDTree(points[:, :2])
        _, idx_manual = tree.query(grid_xy)

        expected_z = points[idx_manual, 2]

        np.testing.assert_allclose(result[:, 2], expected_z)

    # Z values must come from original dataset
    def test_z_values_are_subset(self):
        points = np.array([
            [0, 0, 5],
            [1, 0, 15],
            [0, 1, 25],
            [1, 1, 35],
        ])

        extent = [0, 1, 0, 1, 0, 40]
        nx, ny = 4, 4

        result = resample_preserve_z_nearest(points, nx, ny, extent)

        self.assertTrue(np.all(np.isin(result[:, 2], points[:, 2])))

    #  Deterministic output (important for regression)
    def test_deterministic_output(self):
        points = np.random.rand(10, 3)
        extent = [0, 1, 0, 1, 0, 1]

        r1 = resample_preserve_z_nearest(points, 5, 5, extent)
        r2 = resample_preserve_z_nearest(points, 5, 5, extent)

        np.testing.assert_allclose(r1, r2)

#####################################
if __name__ == "__main__":
    unittest.main()
