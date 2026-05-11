import unittest
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data import  validate_shafts


class TestValidateShafts(unittest.TestCase):

    # -----------------------------------
    # VALID SHAFTS
    # -----------------------------------
    def test_valid_shafts(self):
        extent = [0, 10, 0, 10, 0, 10]

        shafts = [
            {
                "center": (5, 5, 5),
                "axis": (0, 0, 1),
                "radius": 1
            },
            {
                "center": (0, 0, 0),  # boundary case
                "axis": (1, 0, 0),
                "radius": 2
            }
        ]

        # should NOT raise
        validate_shafts(shafts, extent)


    # -----------------------------------
    # SHAFT OUTSIDE EXTENT
    # -----------------------------------
    def test_shaft_outside_raises(self):
        extent = [0, 10, 0, 10, 0, 10]

        shafts = [
            {
                "center": (5, 5, 5),
                "axis": (0, 0, 1),
                "radius": 1
            },
            {
                "center": (20, 5, 5),  # outside
                "axis": (0, 1, 0),
                "radius": 1
            }
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_shafts(shafts, extent)

        self.assertIn("Shaft", str(ctx.exception))
        self.assertIn("outside extent", str(ctx.exception))


    # -----------------------------------
    # EMPTY INPUT
    # -----------------------------------
    def test_empty_shafts(self):
        extent = [0, 10, 0, 10, 0, 10]

        # should NOT raise
        validate_shafts([], extent)


    # -----------------------------------
    # INVALID EXTENT
    # -----------------------------------
    def test_invalid_extent(self):
        shafts = [
            {
                "center": (1, 1, 1),
                "axis": (0, 0, 1),
                "radius": 1
            }
        ]

        with self.assertRaises(ValueError):
            validate_shafts(shafts, [0, 10])  # wrong format


    # -----------------------------------
    # MULTIPLE SHAFTS ONE INVALID
    # -----------------------------------
    def test_multiple_shafts_one_invalid(self):
        extent = [0, 10, 0, 10, 0, 10]

        shafts = [
            {
                "center": (2, 2, 2),
                "axis": (0, 0, 1),
                "radius": 1
            },
            {
                "center": (3, 3, 3),
                "axis": (1, 0, 0),
                "radius": 1
            },
            {
                "center": (-5, 0, 0),  # outside
                "axis": (0, 1, 0),
                "radius": 1
            }
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_shafts(shafts, extent)

        self.assertIn("Shaft", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
