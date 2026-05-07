import re
import unittest
import os
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data import (
    load_triangulations_planes_from_csv,
    validate_triangulation
)

data_dir = os.path.dirname(__file__) + "/data/triangulation/"


class LoadTriangulationsPlanesTestCase(unittest.TestCase):
    """Tests loading and validating triangulation planes, including extent checks."""

    def setUp(self):
        # Define a bounding box for extent validation
        self.extent = (0, 1000, 0, 1000, 0, 1000)

    def test_correct(self):
        ret = load_triangulations_planes_from_csv(
            data_dir + "valid_triangulation.csv"
        )

        # expected number of points
        self.assertEqual(175, len(ret), "Number of triangulation points did not match")

        # check first point
        np.testing.assert_array_equal(
            np.array([183.12500000,616.00000000,250.00000100]),
            ret[0]
        )

        # check last point
        np.testing.assert_array_equal(
            np.array([271.53125000,679.50000000,157.38330179]),
            ret[-1]
        )

        # check shape
        self.assertEqual(ret.shape[1], 3, "Triangulation points must have 3 coordinates")

        # Validate that points lie inside extent
        # This should pass without error for valid points
        validate_triangulation(ret, extent=self.extent)

    def test_missing_value(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected 3 columns (x,y,z)")
        ):
            load_triangulations_planes_from_csv(
                data_dir + "invalid_triangulations_missing_value.csv"
            )

    def test_invalid_column_count(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("expected 3 columns (x,y,z)")
        ):
            load_triangulations_planes_from_csv(
                data_dir + "invalid_triangulations_columns.csv"
            )

    def test_non_numeric_values(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("non-numeric value found")
        ):
            load_triangulations_planes_from_csv(
               data_dir + "invalid_triangulations_non_numeric.csv"
            )

    def test_empty_file(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("No points loaded from the CSV file")
        ):
            load_triangulations_planes_from_csv(
                data_dir + "invalid_triangulations_empty.csv"
            )

    def test_return_type(self):
        ret = load_triangulations_planes_from_csv(
            data_dir + "valid_triangulation.csv"
        )

        self.assertIsInstance(ret, np.ndarray, "Return type must be numpy.ndarray")
        self.assertEqual(ret.dtype, np.float64, "Data type must be float64")

    # ----------------- Extent validation test -----------------
    def test_points_outside_extent(self):
        # Create points some of which are outside the extent
        points_outside = np.array([
            [150.0, 650.0, 200.0],  # inside
            [3500.0, 650.0, 200.0],  # x > xmax → outside
            [200.0, 7500.0, 200.0],  # y > ymax → outside
            [200.0, 650.0, 3000.0]   # z > zmax → outside
        ])

        with self.assertRaisesRegex(
            ValueError,
            re.escape("triangulation points lie outside the extent")
        ):
            validate_triangulation(points_outside, extent=self.extent)


if __name__ == "__main__":
    unittest.main()
