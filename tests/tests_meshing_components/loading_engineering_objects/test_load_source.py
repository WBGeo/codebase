import re
import unittest
import os
from core.meshing_components.explicit.unstructured.mesh_data import (
    load_sources_from_csv,
    validate_sources
)

data_dir = os.path.dirname(__file__) + "/data/sources/"

class LoadSourcesTestCase(unittest.TestCase):
    """Tests for loading and validating sources from CSV."""

    def setUp(self):
        # Define a model extent: xmin, xmax, ymin, ymax, zmin, zmax
        self.extent = [0, 1000, 0, 1000, 0, 1000]

    # Loading
    def test_correct(self):
        sources = load_sources_from_csv(data_dir + "valid_source.csv")

        # Expect two sources
        self.assertEqual(2, len(sources), "Number of sources did not match")

        # Source 1
        self.assertEqual((200.0, 500.0, 100.0), sources[0])

        # Source 2
        self.assertEqual((300.0, 700.0, 400.0), sources[1])

        # Validate sources are inside extent
        validate_sources(sources, self.extent)

    # Invalid CSV tests
    def test_missing_value(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("(missing x, y, or z value)")
        ):
            load_sources_from_csv(
                data_dir + "invalid_sources_missing_value.csv"
            )

    def test_invalid_column_count(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Invalid source line")
        ):
            load_sources_from_csv(
                data_dir + "invalid_sources_columns.csv"
            )

    def test_duplicate_source_id(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("has more than one coordinates")
        ):
            load_sources_from_csv(
                data_dir + "invalid_sources_duplicate_id.csv"
            )

    def test_non_numeric_values(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Non-numeric value in source definition")
        ):
            load_sources_from_csv(
                data_dir + "invalid_sources_non_numeric.csv"
            )

    #  Extent validation
    def test_source_outside_extent(self):
        # Sources with one point outside the extent
        sources_outside = [
            (500.0, 500.0, 500.0),
            (1200.0, 500.0, 100.0)  # x > xmax
        ]
        # Adjust regex to match actual error message from validate_sources
        with self.assertRaisesRegex(
            ValueError,
            r"Source '2' at \(1200\.0, 500\.0, 100\.0\) is outside extent"
        ):
            validate_sources(sources_outside, self.extent)


    def test_sources_on_boundary(self):
        # Sources exactly on the boundary → should pass
        boundary_sources = [
            (0.0, 0.0, 0.0),
            (1000.0, 1000.0, 1000.0)
        ]
        # Should not raise
        validate_sources(boundary_sources, self.extent)

###########################################
if __name__ == "__main__":
    unittest.main()
