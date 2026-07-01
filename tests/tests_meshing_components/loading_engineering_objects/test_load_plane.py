import re
import unittest
import os
from core.meshing_components.explicit.unstructured.mesh_data import (
    load_planes_from_csv,
    validate_planes
)

data_dir = os.path.dirname(__file__) + "/data/planes/"

class LoadPlanesTestCase(unittest.TestCase):
    """Tests loading and validating planes from CSV, including extent and planarity checks."""

    def setUp(self):
        # Example model extent: xmin, xmax, ymin, ymax, zmin, zmax
        self.extent = [0, 1000, 0, 1000, 0, 1000]

    def test_correct(self):
        planes = load_planes_from_csv(data_dir + "valid_plane.csv")
        self.assertEqual(2, len(planes), "Number of planes did not match")

        # Plane 1
        self.assertEqual(
            (
                100.0, 100.0, 100.0,
                900.0, 100.0, 100.0,
                900.0, 900.0, 100.0,
                100.0, 900.0, 100.0
            ),
            planes[0]
        )

        # Plane 2
        self.assertEqual(
            (
                500.0, 500.0, 400.0,
                900.0, 500.0, 400.0,
                900.0, 900.0, 400.0,
                500.0, 900.0, 400.0
            ),
            planes[1]
        )

        # Validate planes including extent
        validate_planes(planes, self.extent)

    def test_points_outside_extent(self):
        # Plane with a point outside the extent
        planes = [
            (100.0, 100.0, 100.0,
             900.0, 100.0, 100.0,
             900.0, 900.0, 100.0,
             1500.0, 900.0, 100.0)  # x=1500 > xmax=1000
        ]
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Plane #1, point (1500.0, 900.0, 100.0) is outside extent")
        ):
            validate_planes(planes, self.extent)

    def test_non_coplanar_plane(self):
        # 4 points not on the same plane
        planes = [
            (0.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             1.0, 1.0, 1.0)  # z=1 breaks planarity
        ]
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Plane #1 is NOT planar")
        ):
            validate_planes(planes, self.extent)

    # CSV loader tests
    def test_missing_value(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("(missing x, y, or z value)")
        ):
            load_planes_from_csv(
               data_dir + "invalid_planes_missing_value.csv"
            )

    def test_invalid_column_count(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Invalid plane line")
        ):
            load_planes_from_csv(
               data_dir + "invalid_planes_columns.csv"
            )

    def test_non_numeric_values(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Non-numeric value in plane definition")
        ):
            load_planes_from_csv(
                data_dir + "invalid_planes_non_numeric.csv"
            )

    def test_not_enough_points(self):
        # fewer than 4 points → cannot define a plane polygon
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Some plane(s) ['1'] are missing required number of points (minimum 4)")
        ):
            load_planes_from_csv(
                data_dir + "invalid_planes_not_enough_points.csv"
            )

##########################################
if __name__ == "__main__":
    unittest.main()
