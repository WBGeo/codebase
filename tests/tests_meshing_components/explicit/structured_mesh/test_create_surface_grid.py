import unittest
import numpy as np
from types import SimpleNamespace
from core.meshing_components.explicit.structured.mesh_data import (
    create_surface_grid
)

class TestCreateSurfaceGrid(unittest.TestCase):

    def build_mock_results(self):
        """
        Create a fake StructuralModelResults-like object
        with surface_meshes_vertices[2]
        """

        # Two simple surfaces:
        # surface 0 = plane z = 0
        # surface 1 = plane z = 10
        surf1 = np.array([
            [0, 0, 0],
            [0, 10, 0],
            [10, 0, 0],
            [10, 10, 0],
        ])

        surf2 = np.array([
            [0, 0, 10],
            [0, 10, 10],
            [10, 0, 10],
            [10, 10, 10],
        ])

        surface_meshes_vertices = [[], [], [surf1, surf2]]

        return SimpleNamespace(surface_meshes_vertices=surface_meshes_vertices)

    def test_output_structure(self):
        results = self.build_mock_results()

        extent = [0, 10, 0, 10, 0, 10]

        surfaces, n_gx, n_gy = create_surface_grid(results, extent)

        # dictionary output check
        self.assertIsInstance(surfaces, dict)
        self.assertEqual(len(surfaces), 2)

        # grid size must be consistent
        self.assertEqual(n_gx, 2)
        self.assertEqual(n_gy, 2)

    def test_surface_keys_exist(self):
        results = self.build_mock_results()
        extent = [0, 10, 0, 10, 0, 10]

        surfaces, _, _ = create_surface_grid(results, extent)

        self.assertIn("surface_0", surfaces)
        self.assertIn("surface_1", surfaces)

    def test_grid_shape_correct(self):
        results = self.build_mock_results()
        extent = [0, 10, 0, 10, 0, 10]

        surfaces, n_gx, n_gy = create_surface_grid(results, extent)

        expected_size = n_gx * n_gy

        for key, grid in surfaces.items():
            # each grid row = (x,y,z)
            self.assertEqual(grid.shape, (expected_size, 3))

    def test_xy_grid_is_consistent(self):
        results = self.build_mock_results()
        extent = [0, 10, 0, 10, 0, 10]

        surfaces, _, _ = create_surface_grid(results, extent)

        base_xy = surfaces["surface_0"][:, :2]

        for key in surfaces:
            np.testing.assert_allclose(
                surfaces[key][:, :2],
                base_xy
            )

########################################
if __name__ == "__main__":
    unittest.main()
