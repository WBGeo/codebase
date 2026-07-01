import unittest
import numpy as np
from core.meshing_components.explicit.unstructured.mesh_data  import (
    fit_plane,
    max_point_plane_distance,
    validate_planes
)

class TestPlaneValidation(unittest.TestCase):

    def setUp(self):
        self.extent = [0, 10, 0, 10, 0, 10]

    # fit_plane
    def test_fit_plane_basic(self):
        # Points on plane z = 5
        pts = np.array([
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5],
            [1, 1, 5]
        ])

        normal, centroid = fit_plane(pts)

        # Normal should be along z (±)
        self.assertAlmostEqual(abs(normal[2]), 1.0, places=6)
        self.assertTrue(np.allclose(centroid, [0.5, 0.5, 5]))

    def test_fit_plane_invalid_shape(self):
        pts = np.array([1, 2, 3])  # invalid

        with self.assertRaises(ValueError):
            fit_plane(pts)

    # max_point_plane_distance
    def test_max_distance_zero(self):
        pts = np.array([
            [0, 0, 5],
            [1, 1, 5],
            [2, 2, 5]
        ])
        normal = np.array([0, 0, 1])
        p0 = np.array([0, 0, 5])

        dist = max_point_plane_distance(pts, normal, p0)
        self.assertAlmostEqual(dist, 0.0)

    def test_max_distance_nonzero(self):
        pts = np.array([
            [0, 0, 5],
            [1, 1, 6],  # off plane
            [2, 2, 5]
        ])
        normal = np.array([0, 0, 1])
        p0 = np.array([0, 0, 5])

        dist = max_point_plane_distance(pts, normal, p0)
        self.assertAlmostEqual(dist, 1.0)

    # validate_planes: valid case
    def test_validate_planes_valid(self):
        plane = [
            0, 0, 5,
            1, 0, 5,
            0, 1, 5,
            1, 1, 5
        ]

        # Should NOT raise
        validate_planes([plane], self.extent)

    # validate_planes: outside extent
    def test_validate_planes_outside_extent(self):
        plane = [
            0, 0, 5,
            1, 0, 5,
            0, 1, 5,
            100, 1, 5  # outside
        ]

        with self.assertRaises(ValueError):
            validate_planes([plane], self.extent)

    # validate_planes: non-planar
    def test_validate_planes_non_planar(self):
        plane = [
            0, 0, 5,
            1, 0, 5,
            0, 1, 5,
            1, 1, 6  # not coplanar
        ]

        with self.assertRaises(ValueError):
            validate_planes([plane], self.extent, tol=1e-10)

    # validate_planes: dict input
    def test_validate_planes_dict_input(self):
        plane = {
            "coords": [
                0, 0, 5,
                1, 0, 5,
                0, 1, 5,
                1, 1, 5
            ]
        }

        validate_planes([plane], self.extent)

    # multiple planes
    def test_multiple_planes(self):
        planes = [
            [0, 0, 5, 1, 0, 5, 0, 1, 5],
            [2, 2, 6, 3, 2, 6, 2, 3, 6]
        ]

        validate_planes(planes, self.extent)

####################################
if __name__ == "__main__":
    unittest.main()
