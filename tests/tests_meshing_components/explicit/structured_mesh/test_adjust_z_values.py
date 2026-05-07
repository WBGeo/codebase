import unittest
import numpy as np

from core.meshing_components.explicit.structured.mesh_data import (
    adjust_z_values
)


class TestAdjustZValues(unittest.TestCase):

    def test_no_adjustment_needed(self):
        """
        Case: all z values are far apart → no modification expected
        """

        n_gx, n_gy = 2, 2

        # 2 layers, 4 points each
        # z values clearly separated (> tolerance)
        layer1_z = np.array([10, 10, 10, 10])
        layer2_z = np.array([20, 20, 20, 20])

        x = np.zeros(8)
        y = np.zeros(8)

        all_points = np.zeros((2, 3 * n_gx * n_gy))

        all_points[0, :4] = x[:4]
        all_points[0, 4:8] = y[:4]
        all_points[0, 8:12] = layer1_z

        all_points[1, :4] = x[:4]
        all_points[1, 4:8] = y[:4]
        all_points[1, 8:12] = layer2_z

        original = all_points.copy()

        result = adjust_z_values(all_points, n_gx, n_gy, z_threshold=1.0, tolerance=0.5)

        # nothing should change
        np.testing.assert_array_equal(result, original)

    # ---------------------------------------------------------
    def test_adjustment_happens(self):
        """
        Case: z values are too close → adjustment should occur
        """

        n_gx, n_gy = 2, 2

        # layer 1 and 2 are very close → triggers adjustment
        layer1_z = np.array([10, 10, 10, 10])
        layer2_z = np.array([10.3, 10.3, 10.3, 10.3])  # within tolerance

        x = np.zeros(8)
        y = np.zeros(8)

        all_points = np.zeros((2, 3 * n_gx * n_gy))

        all_points[0, :4] = x[:4]
        all_points[0, 4:8] = y[:4]
        all_points[0, 8:12] = layer1_z

        all_points[1, :4] = x[:4]
        all_points[1, 4:8] = y[:4]
        all_points[1, 8:12] = layer2_z

        result = adjust_z_values(
            all_points.copy(),
            n_gx,
            n_gy,
            z_threshold=1.0,
            tolerance=0.5
        )

        # layer1 must be reduced somewhere (because adjustment happened)
        self.assertFalse(np.allclose(result[0, 8:12], layer1_z))

        # layer2 should remain unchanged
        np.testing.assert_array_almost_equal(result[1, 8:12], layer2_z)

    # ---------------------------------------------------------
    def test_multiple_adjustments_possible(self):
        """
        Case: repeated adjustment loop should still converge
        """

        n_gx, n_gy = 2, 2

        layer1_z = np.array([10, 10, 10, 10])
        layer2_z = np.array([10.2, 10.2, 10.2, 10.2])
        layer3_z = np.array([10.4, 10.4, 10.4, 10.4])

        x = np.zeros(12)
        y = np.zeros(12)

        all_points = np.zeros((3, 3 * n_gx * n_gy))

        all_points[0, 8:12] = layer1_z
        all_points[1, 8:12] = layer2_z
        all_points[2, 8:12] = layer3_z

        result = adjust_z_values(
            all_points.copy(),
            n_gx,
            n_gy,
            z_threshold=0.5,
            tolerance=0.5
        )

        # function should not crash and should modify values
        self.assertEqual(result.shape, all_points.shape)
        self.assertFalse(np.allclose(result, all_points))


if __name__ == "__main__":
    unittest.main()
