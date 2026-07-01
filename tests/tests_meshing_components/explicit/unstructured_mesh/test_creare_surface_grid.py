import unittest
import numpy as np
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import create_surface_grid

class TestCreateSurfaceGrid(unittest.TestCase):

    def make_planar_surface(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        z = x + y
        points = np.column_stack([x, y, z])
        return [(0, points)]

    def make_vertical_surface(self):
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 0, 1])
        x = np.zeros_like(y)
        points = np.column_stack([x, y, z])
        return [(0, points)]

    # Output structure
    def test_output_structure(self):
        surfaces = self.make_planar_surface()

        result = create_surface_grid(surfaces)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape[1], 3)

    # Planar surface sanity
    def test_planar_surface_grid_properties(self):
        surfaces = self.make_planar_surface()

        result = create_surface_grid(surfaces)
        grid = result[0]

        self.assertFalse(np.isnan(grid).any())
        self.assertFalse(np.isinf(grid).any())

        self.assertAlmostEqual(grid[:, 0].min(), 0, places=2)
        self.assertAlmostEqual(grid[:, 0].max(), 1, places=2)
        self.assertAlmostEqual(grid[:, 1].min(), 0, places=2)
        self.assertAlmostEqual(grid[:, 1].max(), 1, places=2)

    # Vertical surface
    def test_vertical_surface(self):
        surfaces = self.make_vertical_surface()

        result = create_surface_grid(surfaces)
        grid = result[0]

        self.assertTrue(np.allclose(grid[:, 0], 0, atol=1e-6))
        self.assertFalse(np.isnan(grid).any())

    # Multiple surfaces
    def test_multiple_surfaces(self):
        s1 = self.make_planar_surface()[0]
        s2 = self.make_planar_surface()[0]

        result = create_surface_grid([s1, s2])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape[1], 3)
        self.assertEqual(result[1].shape[1], 3)

###############################
if __name__ == "__main__":
    unittest.main()
