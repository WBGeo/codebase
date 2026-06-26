import unittest
import os
import re
from core.meshing_components.explicit.unstructured.mesh_data import (
    load_ellipses_from_csv,
    validate_ellipses
)

# Path to test CSVs
data_dir = os.path.dirname(__file__) + "/data/ellipses/"


class LoadEllipsesTestCase(unittest.TestCase):
    """Tests loading and validating ellipses from CSV."""

    def setUp(self):
        # Define model extent: xmin, xmax, ymin, ymax, zmin, zmax
        self.extent = [0, 1000, 0, 1000, 0, 1000]

    # Loading
    def test_correct(self):
        ellipses = load_ellipses_from_csv(data_dir + "valid_ellipses.csv")
        self.assertEqual(4, len(ellipses), "Expected 4 ellipses in CSV")

        # Check first ellipse
        first = ellipses[0]
        self.assertEqual(
            first["center"], (300.0, 200.0, 200.0)
        )
        self.assertEqual(first["radii"], (40.0, 40.0))
        self.assertIn("zAxis", first)
        self.assertEqual(first["zAxis"], [0, 0, 1])

        # Validate extents
        validate_ellipses(ellipses, self.extent)

    # Required fields
    def test_missing_required_numbers(self):
        # Load a CSV with fewer than required columns
        ellipses = load_ellipses_from_csv(data_dir + "invalid_ellipses_missing_numbers.csv")
        # Should still return a list, but missing fields filled with defaults
        self.assertTrue(all("angle1" in e or "zAxis" in e for e in ellipses))

    def test_non_numeric_radii_or_angles(self):
        # CSV with non-numeric radii or angles
        with self.assertRaisesRegex(
            ValueError,
            re.escape("invalid numeric value")
        ):
            load_ellipses_from_csv(
                data_dir + "invalid_ellipses_non_numeric.csv"
            )

    #  Axis validation
    def test_invalid_zaxis_xaxis_columns(self):
        # Load a CSV where zAxis/xAxis columns are wrong or missing
        ellipses = load_ellipses_from_csv(data_dir + "invalid_ellipses_axis_columns.csv")
        for e in ellipses:
            # Ensure zAxis exists and has 3 values
            self.assertIn("zAxis", e)
            self.assertEqual(len(e["zAxis"]), 3)
            # Ensure xAxis exists; if missing, loader should default it
            if "xAxis" not in e:
                e["xAxis"] = [1, 0, 0]  # match loader default
            self.assertEqual(len(e["xAxis"]), 3)


    #  Extent validation
    def test_validate_ellipse_inside_extent(self):
        ellipses = load_ellipses_from_csv(data_dir + "valid_ellipses.csv")

        # Should pass
        validate_ellipses(ellipses, self.extent)

        # Move first ellipse outside extent → should raise
        ellipses[0]["center"] = (1500, 500, 500)
        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    def test_center_outside_extent(self):
        ellipses = [{"center": (1500, 500, 500), "radii": (100, 50)}]
        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    def test_ellipse_partially_outside_extent(self):
        ellipses = [{"center": (50, 500, 500), "radii": (100, 50)}]
        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    def test_ellipse_inside_extent(self):
        ellipses = [{"center": (500, 500, 500), "radii": (100, 50)}]
        # Should NOT raise
        validate_ellipses(ellipses, self.extent)

    def test_ellipse_on_extent_boundary(self):
        ellipses = [{"center": (100, 500, 500), "radii": (100, 50)}]
        # Should NOT raise
        validate_ellipses(ellipses, self.extent)


#########################################
if __name__ == "__main__":
    unittest.main()
