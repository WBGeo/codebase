import re
import unittest
import os
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data import (
    load_shafts_from_csv,
    validate_shafts
)

data_dir = os.path.dirname(__file__) + "/data/shafts/"

class LoadShaftsTestCase(unittest.TestCase):
    """Tests loading and validating mine shafts from CSV, including extent checks."""

    def setUp(self):
        # Example model extent: xmin, xmax, ymin, ymax, zmin, zmax
        self.extent = [0, 1000, 0, 1000, 0, 1000]

    def test_correct(self):
        shafts = load_shafts_from_csv(data_dir + "valid_shaft.csv")
        self.assertEqual(2, len(shafts), "Number of shafts did not match")

        # Shaft 1
        self.assertEqual(
            (200.0, 500.0, 100.0, 1000.0, 0.0, 0.0, 20.0),
            shafts[0]
        )

        # Shaft 2
        self.assertEqual(
            (300.0, 700.0, 400.0, 1000.0, 0.0, 0.0, 20.0),
            shafts[1]
        )

        # Validate shafts against extent
        # Convert to dictionary format expected by validate_shafts
        shafts_dict = [
            {"center": s[:3], "axis": s[3:6], "radius": s[6]} for s in shafts
        ]
        validate_shafts(shafts_dict, self.extent)

    # ----------------- CSV loader error tests -----------------
    def test_missing_value(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Missing value(s)")
        ):
            load_shafts_from_csv(data_dir + "invalid_shafts_missing_value.csv"
            )

    def test_invalid_column_count(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Invalid shaft line")
        ):
            load_shafts_from_csv(
                data_dir + "invalid_shafts_columns.csv"
            )

    def test_duplicate_shaft_id(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Duplicate shaft definition for shaft ID '1'")
        ):
            load_shafts_from_csv(
                data_dir + "invalid_shafts_duplicate_id.csv"
            )

    def test_non_numeric_values(self):
        with self.assertRaises(ValueError):
            load_shafts_from_csv(
                data_dir + "invalid_shafts_non_numeric.csv"
            )

    # ----------------- Extent validation test -----------------
    def test_center_outside_extent(self):
        # Shaft with center outside the extent
        shafts_outside = [
            {"center": (1500.0, 500.0, 100.0), "axis": (0, 0, 1), "radius": 10.0}
        ]
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Shaft '1' center (1500.0, 500.0, 100.0) is outside extent")
        ):
            validate_shafts(shafts_outside, self.extent)


if __name__ == "__main__":
    unittest.main()
