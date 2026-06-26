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

    def setUp(self):
        self.extent = (0, 1000, 0, 1000, 0, 1000)

    # MAIN VALID TEST
    def test_correct(self):

        ret = load_triangulations_planes_from_csv(
            data_dir + "valid_triangulation.csv"
        )

        self.assertEqual(175, len(ret))

        # check first point (x,y,z,id)
        np.testing.assert_array_equal(
            np.array([183.12500000, 616.00000000, 250.00000100, 1.0]),
            ret[0]
        )

        # check last point (x,y,z,id)
        np.testing.assert_array_equal(
            np.array([271.53125000, 679.50000000, 157.38330179, 1.0]),
            ret[-1]
        )

        # shape must be (N, 4)
        self.assertEqual(
            ret.shape[1],
            4,
            "Triangulation points must have 4 columns (x,y,z,id)"
        )

        # validate only xyz
        validate_triangulation(ret, extent=self.extent)

    # INVALID COLUMN COUNT
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

    # NON NUMERIC
    def test_non_numeric_values(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("non-numeric value found")
        ):
            load_triangulations_planes_from_csv(
                data_dir + "invalid_triangulations_non_numeric.csv"
            )

    # EMPTY FILE
    def test_empty_file(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("No points loaded from the CSV file")
        ):
            load_triangulations_planes_from_csv(
                data_dir + "invalid_triangulations_empty.csv"
            )

    # RETURN TYPE
    def test_return_type(self):

        ret = load_triangulations_planes_from_csv(
            data_dir + "valid_triangulation.csv"
        )

        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.dtype, np.float64)

    # EXTENT VALIDATION
    def test_points_outside_extent(self):

        points_outside = np.array([
            [150.0, 650.0, 200.0],
            [3500.0, 650.0, 200.0],
            [200.0, 7500.0, 200.0],
            [200.0, 650.0, 3000.0]
        ])

        with self.assertRaisesRegex(
            ValueError,
            re.escape("triangulation points lie outside the extent")
        ):
            validate_triangulation(points_outside, extent=self.extent)

###################################
if __name__ == "__main__":
    unittest.main()
