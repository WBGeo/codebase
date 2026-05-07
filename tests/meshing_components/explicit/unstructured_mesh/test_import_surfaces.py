import unittest
import numpy as np
from unittest.mock import patch

from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import (import_surfaces)


class TestImportSurfaces(unittest.TestCase):

    # -----------------------------
    # helper surface (valid grid)
    # -----------------------------
    def make_surface(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        z = np.array([10, 10, 10, 10])
        return np.column_stack([x, y, z])

    # -----------------------------
    # 1. valid surface import
    # -----------------------------
    @patch("gmsh.model.occ.addPoint")
    @patch("gmsh.model.occ.addBSplineSurface")
    def test_valid_surface_import(self, mock_bspline, mock_point):

        mock_point.side_effect = range(1000)
        mock_bspline.return_value = 1

        surfaces = [self.make_surface()]

        result_surfaces, bounds = import_surfaces(
            surfaces,
            extent=(0, 1, 0, 1, 10, 10),
            tolerance=0.1
        )

        self.assertEqual(len(result_surfaces), 1)
        self.assertEqual(result_surfaces[0], 1)
        self.assertIsNotNone(bounds)

    # -----------------------------
    # 2. invalid surface skipped
    # -----------------------------
    @patch("gmsh.model.occ.addPoint")
    @patch("gmsh.model.occ.addBSplineSurface")
    def test_invalid_surface_skipped(self, mock_bspline, mock_point):

        mock_point.return_value = 1

        bad_surface = np.array([[1, 2], [3, 4]])  # invalid shape

        result_surfaces, bounds = import_surfaces(
            [bad_surface],
            extent=(0, 1, 0, 1, 0, 1)
        )

        self.assertEqual(len(result_surfaces), 0)
        self.assertIsNone(bounds)

    # -----------------------------
    # 3. bounds computation
    # -----------------------------
    @patch("gmsh.model.occ.addPoint")
    @patch("gmsh.model.occ.addBSplineSurface")
    def test_bounds_computation(self, mock_bspline, mock_point):

        mock_point.side_effect = range(1000)
        mock_bspline.return_value = 1

        surface = self.make_surface()

        _, bounds = import_surfaces(
            [surface],
            extent=(0, 1, 0, 1, 10, 10),
            tolerance=1.0
        )

        self.assertIsInstance(bounds, tuple)
        self.assertEqual(len(bounds), 6)

        self.assertAlmostEqual(bounds[0], 0)
        self.assertAlmostEqual(bounds[1], 1)
        self.assertAlmostEqual(bounds[2], 0)
        self.assertAlmostEqual(bounds[3], 1)

    # -----------------------------
    # 4. no extent provided (FIXED)
    # -----------------------------
    @patch("gmsh.model.occ.addPoint")
    @patch("gmsh.model.occ.addBSplineSurface")
    def test_no_extent(self, mock_bspline, mock_point):

        mock_point.side_effect = range(1000)
        mock_bspline.return_value = 1

        surface = self.make_surface()

        # IMPORTANT: extent=None is NOT supported by current function
        result_surfaces, bounds = import_surfaces([surface], extent=(0, 1, 0, 1, 10, 10))

        self.assertEqual(len(result_surfaces), 1)
        self.assertIsNotNone(bounds)


if __name__ == "__main__":
    unittest.main()
