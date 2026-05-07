import unittest
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data  import (
    validate_ellipses
)


class TestValidateEllipses(unittest.TestCase):

    def setUp(self):
        self.extent = [0, 10, 0, 10, 0, 10]

    # -------------------------
    # VALID CASE
    # -------------------------
    def test_valid_ellipse(self):
        ellipses = [{
            "center": (5, 5, 5),
            "radii": (1, 2)
        }]

        # Should NOT raise
        validate_ellipses(ellipses, self.extent)

    # -------------------------
    # MISSING KEYS
    # -------------------------
    def test_missing_center(self):
        ellipses = [{
            "radii": (1, 2)
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    def test_missing_radii(self):
        ellipses = [{
            "center": (5, 5, 5)
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    # -------------------------
    # NON-FINITE VALUES
    # -------------------------
    def test_non_finite_values(self):
        ellipses = [{
            "center": (np.nan, 5, 5),
            "radii": (1, 2)
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    # -------------------------
    # NEGATIVE / ZERO RADII
    # -------------------------
    def test_negative_radius(self):
        ellipses = [{
            "center": (5, 5, 5),
            "radii": (-1, 2)
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    def test_zero_radius(self):
        ellipses = [{
            "center": (5, 5, 5),
            "radii": (0, 2)
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    # -------------------------
    # OUTSIDE EXTENT
    # -------------------------
    def test_outside_extent(self):
        ellipses = [{
            "center": (9.5, 5, 5),
            "radii": (1, 1)  # extends beyond xmax=10
        }]

        with self.assertRaises(ValueError):
            validate_ellipses(ellipses, self.extent)

    # -------------------------
    # EXACTLY ON BOUNDARY (VALID)
    # -------------------------
    def test_on_boundary(self):
        ellipses = [{
            "center": (5, 5, 5),
            "radii": (5, 5)  # touches boundaries exactly
        }]

        validate_ellipses(ellipses, self.extent)

    # -------------------------
    # MULTIPLE ELLIPSES
    # -------------------------
    def test_multiple_ellipses(self):
        ellipses = [
            {"center": (3, 3, 3), "radii": (1, 1)},
            {"center": (7, 7, 7), "radii": (2, 2)}
        ]

        validate_ellipses(ellipses, self.extent)


if __name__ == "__main__":
    unittest.main()
