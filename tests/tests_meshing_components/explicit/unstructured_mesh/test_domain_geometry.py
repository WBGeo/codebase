"""Geometry contracts introduced by oriented-domain support."""
import unittest
import numpy as np
from core.meshing_components.explicit.unstructured.domain_geometry import (
    normalize_domain, domain_aabb, domain_local_coordinates,
    point_inside_domain, points_inside_domain, classify_local_face,
)


class DomainGeometryTests(unittest.TestCase):
    def setUp(self):
        # Parallelepiped with non-axis-aligned local x direction.
        o = np.array([10., 20., 30.]); u = np.array([2., 1., 0.])
        v = np.array([0., 3., 0.]); w = np.array([0., 0., 4.])
        self.corners = np.array([o, o+u, o+u+v, o+v,
                                 o+w, o+u+w, o+u+v+w, o+v+w])

    def test_legacy_extent_is_supported(self):
        kind, data = normalize_domain([0, 2, 1, 4, -1, 3])
        self.assertEqual(kind, 'extent')
        np.testing.assert_array_equal(data, [0, 2, 1, 4, -1, 3])

    def test_oriented_corners_and_flattened_input(self):
        for domain in (self.corners, self.corners.ravel()):
            kind, corners = normalize_domain(domain)
            self.assertEqual(kind, 'corners')
            np.testing.assert_allclose(corners, self.corners)

    def test_local_coordinates_and_inclusion(self):
        point = self.corners[0] + .25*(self.corners[1]-self.corners[0]) + .5*(self.corners[3]-self.corners[0]) + .75*(self.corners[4]-self.corners[0])
        np.testing.assert_allclose(domain_local_coordinates(point, self.corners), [.25, .5, .75])
        self.assertTrue(point_inside_domain(point, self.corners))
        self.assertFalse(point_inside_domain(self.corners[0] - [0, 0, .01], self.corners))
        np.testing.assert_array_equal(points_inside_domain([point, self.corners[0]-[0,0,.01]], self.corners), [True, False])

    def test_axis_aligned_bounding_box_is_not_the_oriented_domain(self):
        np.testing.assert_allclose(domain_aabb(self.corners), [10, 12, 20, 24, 30, 34])
        # Inside the AABB but outside the slanted box.
        self.assertFalse(point_inside_domain([11.9, 20.1, 32], self.corners))

    def test_local_face_names(self):
        self.assertCountEqual(classify_local_face(self.corners[0], self.corners), ['XMIN', 'YMIN', 'ZMIN'])
        self.assertCountEqual(classify_local_face(self.corners[6], self.corners), ['XMAX', 'YMAX', 'ZMAX'])

    def test_invalid_corner_order_and_degenerate_box(self):
        invalid = self.corners.copy()
        invalid[[1, 2]] = invalid[[2, 1]]
        with self.assertRaises(ValueError):
            normalize_domain(invalid)
        invalid = self.corners.copy()
        invalid[4:] = invalid[:4]
        with self.assertRaises(ValueError):
            normalize_domain(invalid)

    def test_oriented_domain_boundary_tolerance(self):
        """A point just beyond a rotated face is accepted only within tolerance."""
        origin = self.corners[0]
        u = self.corners[1] - origin
        v = self.corners[3] - origin
        w = self.corners[4] - origin
        just_outside = origin + 1.000001 * u + 0.5 * v + 0.5 * w
        self.assertFalse(point_inside_domain(just_outside, self.corners))
        self.assertTrue(point_inside_domain(just_outside, self.corners, tol=1e-5))
        np.testing.assert_allclose(
            domain_local_coordinates(just_outside, self.corners),
            [1.000001, 0.5, 0.5], atol=1e-10,
        )

    def test_invalid_extent(self):
        for extent in ([1, 1, 0, 2, 0, 3], [0, 1, 2, 1, 0, 3], [0, 1, 0]):
            with self.subTest(extent=extent), self.assertRaises(ValueError):
                normalize_domain(extent)
