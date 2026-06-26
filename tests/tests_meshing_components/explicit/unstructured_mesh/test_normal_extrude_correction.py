import unittest
import numpy as np
from scipy.spatial import cKDTree
from core.meshing_components.explicit.unstructured.create_clean_surface import get_normals, calculate_normals, correct_extrusion_direction

class TestNormalsFunctions(unittest.TestCase):

    # get_normals
    def test_get_normals_basic(self):
        points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        near_points = np.array([
            [1, 1, 1],
            [2, 2, 2]
        ])

        normals = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

        normal_vec = (0, normals)

        result = get_normals(near_points, points, normal_vec)

        expected = np.array([
            [0, 1, 0],
            [1, 0, 0]
        ])

        np.testing.assert_array_equal(result, expected)

    # calculate_normals
    def test_calculate_normals_basic(self):
        surface_points = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])

        surface_normals = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

        cleaned_surfaces = [
            ("A", surface_points)
        ]

        cleaned_normals = [
            ("A", surface_normals)
        ]

        subset_points = np.array([
            [2, 2, 2],
            [0, 0, 0]
        ])

        result = calculate_normals(
            subset_points,
            cleaned_surfaces,
            cleaned_normals,
            file_index="A"
        )

        expected = np.array([
            [1, 0, 0],
            [0, 0, 1]
        ])

        np.testing.assert_array_equal(result, expected)

    # calculate_normals (missing point)
    def test_calculate_normals_partial_match(self):
        surface_points = np.array([
            [0, 0, 0],
            [1, 1, 1]
        ])

        surface_normals = np.array([
            [0, 0, 1],
            [0, 1, 0]
        ])

        cleaned_surfaces = [("A", surface_points)]
        cleaned_normals = [("A", surface_normals)]

        subset_points = np.array([
            [1, 1, 1],
            [9, 9, 9]  # not present
        ])

        result = calculate_normals(
            subset_points,
            cleaned_surfaces,
            cleaned_normals,
            file_index="A"
        )

        expected = np.array([
            [0, 1, 0]
        ])

        np.testing.assert_array_equal(result, expected)

    # correct_extrusion_direction
    def test_correct_extrusion_direction_shape(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0]
        ])

        normals = np.array([
            [0, 0, 1],
            [0, 0, 1]
        ])

        intersection_points = np.array([
            [100, 0, 0]
        ])

        nearest_points_dict = {
            "A": np.array([
                [0, 0, 0],
                [1, 0, 0]
            ])
        }

        result = correct_extrusion_direction(
            points,
            normals,
            file="A",
            intersection_points=intersection_points,
            nearest_points_dict=nearest_points_dict,
            EXTRUSION_FACTOR=10,
            num_steps=5
        )

        # shape: (n_points, num_steps+1, 3)
        self.assertEqual(result.shape, (2, 6, 3))

    # extrusion direction consistency
    def test_extrusion_direction_nonzero(self):
        points = np.array([[0, 0, 0]])
        normals = np.array([[0, 0, 1]])

        intersection_points = np.array([[10, 0, 0]])
        nearest_points_dict = {"A": np.array([[0, 0, 0]])}

        result = correct_extrusion_direction(
            points,
            normals,
            file="A",
            intersection_points=intersection_points,
            nearest_points_dict=nearest_points_dict,
            EXTRUSION_FACTOR=10,
            num_steps=5
        )

        # ensure movement occurs
        first = result[0, 0]
        last = result[0, -1]

        self.assertFalse(np.allclose(first, last))

#####################################
if __name__ == "__main__":
    unittest.main()
