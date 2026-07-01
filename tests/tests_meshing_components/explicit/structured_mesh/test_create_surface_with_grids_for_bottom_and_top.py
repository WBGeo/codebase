import unittest
import numpy as np
from core.meshing_components.explicit.structured.mesh_data import (
    create_surfaces_with_grids_for_bottom_and_top
)

class TestBottomTopSurfaces(unittest.TestCase):

    def test_shape_and_structure(self):
        """
        Check correct output shape and basic structure
        """

        min_x, max_x = 0, 10
        min_y, max_y = 0, 20
        min_z, max_z = -5, 5
        n_gx, n_gy = 3, 4

        result = create_surfaces_with_grids_for_bottom_and_top(
            min_x, max_x, min_y, max_y,
            min_z, max_z,
            n_gx, n_gy
        )

        # 2 surfaces × (3 * n_gx * n_gy)
        expected_shape = (2, 3 * n_gx * n_gy)
        self.assertEqual(result.shape, expected_shape)

    def test_bottom_surface_constant_z(self):
        """
        Bottom surface must have all z = min_z
        """

        min_x, max_x = 0, 4
        min_y, max_y = 0, 4
        min_z, max_z = -10, 10
        n_gx, n_gy = 2, 2

        result = create_surfaces_with_grids_for_bottom_and_top(
            min_x, max_x, min_y, max_y,
            min_z, max_z,
            n_gx, n_gy
        )

        n = n_gx * n_gy

        bottom_z = result[0, 2 * n:]

        self.assertTrue(np.all(bottom_z == min_z))

    def test_top_surface_constant_z(self):
        """
        Top surface must have all z = max_z
        """

        min_x, max_x = 0, 4
        min_y, max_y = 0, 4
        min_z, max_z = -10, 10
        n_gx, n_gy = 2, 2

        result = create_surfaces_with_grids_for_bottom_and_top(
            min_x, max_x, min_y, max_y,
            min_z, max_z,
            n_gx, n_gy
        )

        n = n_gx * n_gy

        top_z = result[1, 2 * n:]

        self.assertTrue(np.all(top_z == max_z))

    def test_xy_grid_consistency(self):
        """
        Ensure both surfaces share identical X/Y structure
        """

        min_x, max_x = 0, 2
        min_y, max_y = 0, 2
        min_z, max_z = 0, 10
        n_gx, n_gy = 2, 2

        result = create_surfaces_with_grids_for_bottom_and_top(
            min_x, max_x, min_y, max_y,
            min_z, max_z,
            n_gx, n_gy
        )

        n = n_gx * n_gy

        bottom_x = result[0, :n]
        top_x = result[1, :n]

        bottom_y = result[0, n:2*n]
        top_y = result[1, n:2*n]

        np.testing.assert_array_equal(bottom_x, top_x)
        np.testing.assert_array_equal(bottom_y, top_y)

##########################################
if __name__ == "__main__":
    unittest.main()
