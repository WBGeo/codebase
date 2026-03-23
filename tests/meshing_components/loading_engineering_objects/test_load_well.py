import re
import unittest
import os
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data import (
    load_wells_from_csv,
    validate_wells
)

data_dir = os.path.dirname(__file__) + "/data/wells/"


class LoadWellsTestCase(unittest.TestCase):
    """Tests loading and validating wells from CSV, including extent checks."""

    def setUp(self):
        # Example model extent: xmin, xmax, ymin, ymax, zmin, zmax
        self.extent = [0, 1000, 0, 1000, 0, 1000]

    def test_correct(self):
        wells = load_wells_from_csv(data_dir + 'valid_wells.csv')
        self.assertEqual(2, len(wells), "Length did not match")
        self.assertEqual(
            (100.0, 100.0, 100.0, 100.0,100.0, 500.0, 100.0, 200.0, 500.0, 100.0, 200.0, 200.0),
            wells[0]
        )
        self.assertEqual(
            (500.0, 500.0, 500.0, 500.0, 500.0, 900.0),
            wells[1]
        )

        # Validate wells are inside extent
        validate_wells(wells, self.extent)

    def test_not_a_vertice(self):
        # One well is missing a second point → no vertice could be created
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Some well(s) ['1'] are missing their second point")
        ):
            load_wells_from_csv(os.path.join(data_dir, "invalid_wells_not_a_vertice_wells.csv"))
    def test_vertex_missing_component(self):

        with self.assertRaisesRegex(
            ValueError,
            re.escape("Invalid well line (expected 4 columns: id,x,y,z)")
        ):
            load_wells_from_csv(
                os.path.join(data_dir, "invalid_wells_missing_vertex_component.csv")
            )
    def test_vertex_non_numeric(self):

        with self.assertRaisesRegex(
            ValueError,
            re.escape("Non-numeric value in well definition")
        ):
            load_wells_from_csv(
                os.path.join(data_dir, "invalid_wells_non_numeric_vertex.csv")
            )

    # ----------------- Extent validation test -----------------
    def test_point_outside_extent(self):
        # Well with a point outside the extent
        wells_outside = [
            (500.0, 500.0, 500.0, 1500.0, 500.0, 500.0)  # second point x=1500 > xmax
        ]
        with self.assertRaisesRegex(
            ValueError,
            r"Well '1', point #2 at \(1500\.0, 500\.0, 500\.0\) is outside extent .*"
        ):
            validate_wells(wells_outside, self.extent)



if __name__ == '__main__':
    unittest.main()
