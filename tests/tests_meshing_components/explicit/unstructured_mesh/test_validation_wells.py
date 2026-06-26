import unittest
import numpy as np
from core.meshing_components.explicit.unstructured.mesh_data import point_on_line_segment, _check_extent_format, point_inside_extent, assert_point_inside_extent, validate_wells

class TestGeometryUtilities(unittest.TestCase):

    # point_on_line_segment
    def test_point_on_line_true(self):
        p1 = (0, 0, 0)
        p2 = (2, 2, 2)
        pt = (1, 1, 1)

        self.assertTrue(point_on_line_segment(pt, p1, p2))

    def test_point_on_line_false(self):
        p1 = (0, 0, 0)
        p2 = (2, 2, 2)
        pt = (1, 2, 1)

        self.assertFalse(point_on_line_segment(pt, p1, p2))

    def test_axis_aligned_line(self):
        p1 = (0, 0, 0)
        p2 = (0, 0, 5)
        pt = (0, 0, 3)

        self.assertTrue(point_on_line_segment(pt, p1, p2))

    def test_axis_aligned_off_line(self):
        p1 = (0, 0, 0)
        p2 = (0, 0, 5)
        pt = (1, 0, 3)

        self.assertFalse(point_on_line_segment(pt, p1, p2))


    # _check_extent_format
    def test_valid_extent(self):
        extent = [0, 1, 0, 1, 0, 1]
        _check_extent_format(extent)  # should not raise

    def test_extent_none(self):
        with self.assertRaises(ValueError):
            _check_extent_format(None)

    def test_extent_wrong_length(self):
        with self.assertRaises(ValueError):
            _check_extent_format([0, 1, 0])  # too short


    # point_inside_extent
    def test_point_inside(self):
        extent = [0, 10, 0, 10, 0, 10]
        pt = (5, 5, 5)

        self.assertTrue(point_inside_extent(pt, extent))

    def test_point_on_boundary(self):
        extent = [0, 10, 0, 10, 0, 10]
        pt = (0, 10, 5)

        self.assertTrue(point_inside_extent(pt, extent))

    def test_point_outside(self):
        extent = [0, 10, 0, 10, 0, 10]
        pt = (11, 5, 5)

        self.assertFalse(point_inside_extent(pt, extent))


    # assert_point_inside_extent
    def test_assert_inside(self):
        extent = [0, 10, 0, 10, 0, 10]
        pt = (5, 5, 5)

        # should not raise
        assert_point_inside_extent(pt, extent)

    def test_assert_outside_raises(self):
        extent = [0, 10, 0, 10, 0, 10]
        pt = (15, 5, 5)

        with self.assertRaises(ValueError) as ctx:
            assert_point_inside_extent(pt, extent, obj_type="Well", obj_id=1, point_idx=0)

        self.assertIn("outside extent", str(ctx.exception))


    # validate_wells
    def test_valid_wells(self):
        extent = [0, 10, 0, 10, 0, 10]

        wells = [
            (0, 0, 0, 1, 1, 1),
            (2, 2, 2, 3, 3, 3)
        ]

        # should not raise
        validate_wells(wells, extent)

    def test_well_outside_raises(self):
        extent = [0, 10, 0, 10, 0, 10]

        wells = [
            (0, 0, 0, 1, 1, 1),
            (2, 2, 2, 30, 3, 3)  # <-- outside
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_wells(wells, extent)

        self.assertIn("Well", str(ctx.exception))

    def test_empty_wells(self):
        extent = [0, 10, 0, 10, 0, 10]

        # should not raise
        validate_wells([], extent)

#############################################
if __name__ == "__main__":
    unittest.main()
