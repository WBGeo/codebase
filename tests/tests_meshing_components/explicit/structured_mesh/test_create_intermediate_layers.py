import unittest
import numpy as np

from core.meshing_components.explicit.structured.store_grid_data import (
    create_intermediate_layers
)


class TestCreateIntermediateLayers(unittest.TestCase):

    def setUp(self):
        self.n_gx = 2
        self.n_gy = 2
        self.n = self.n_gx * self.n_gy

        # bottom surface (x, y, z)
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z_bottom = np.array([0, 0, 0, 0])
        z_top = np.array([10, 10, 10, 10])

        # bottom/top container: (2, 3n)
        self.bottom_top = np.zeros((2, 3 * self.n))
        self.bottom_top[0, :self.n] = x
        self.bottom_top[0, self.n:2*self.n] = y
        self.bottom_top[0, 2*self.n:] = z_bottom

        self.bottom_top[1, 2*self.n:] = z_top

        # one intermediate surface (same grid)
        surf = np.zeros((1, 3 * self.n))
        surf[0, 2*self.n:] = np.array([5, 5, 5, 5])

        self.output_array = surf

    def test_output_shape_is_correct(self):
        refinement = [2, 2]  # must be len(output_array)+1

        result = create_intermediate_layers(
            self.bottom_top,
            self.output_array,
            refinement,
            self.n_gx,
            self.n_gy
        )

        # number of generated layers = bottom + intermediates + surface layers + top
        self.assertTrue(result.shape[0] > 0)

        # each row must be 4*n (x,y,z,id)
        self.assertEqual(result.shape[1], 4 * self.n)

    def test_bottom_and_top_exist(self):
        refinement = [2, 2]

        result = create_intermediate_layers(
            self.bottom_top,
            self.output_array,
            refinement,
            self.n_gx,
            self.n_gy
        )

        # bottom check (first row)
        np.testing.assert_array_equal(
            result[0, :self.n], self.bottom_top[0, :self.n]
        )

        # top check (last row z-values should match top)
        self.assertTrue(
            np.allclose(result[-1, 2*self.n:3*self.n], self.bottom_top[1, 2*self.n:])
        )

    def test_surface_ids_exist(self):
        refinement = [2, 2]

        result = create_intermediate_layers(
            self.bottom_top,
            self.output_array,
            refinement,
            self.n_gx,
            self.n_gy
        )

        # last column = surface ids
        ids = result[:, -self.n:]

        self.assertTrue(np.all(ids >= 0))

    def test_z_interpolation_behavior(self):
        refinement = [2, 2]

        result = create_intermediate_layers(
            self.bottom_top,
            self.output_array,
            refinement,
            self.n_gx,
            self.n_gy
        )

        z_values = result[:, 2*self.n:3*self.n]

        # sanity: z should stay within range
        self.assertTrue(np.all(z_values >= 0))
        self.assertTrue(np.all(z_values <= 10))


if __name__ == "__main__":
    unittest.main()
