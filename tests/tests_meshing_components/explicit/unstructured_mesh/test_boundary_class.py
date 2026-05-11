import unittest
import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data  import classify_boundary_nodes, build_point_sets


class TestBoundaryClassification(unittest.TestCase):

    # ---------------------------------------------------
    def setUp(self):
        # simple cube domain
        self.extent = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)

        # structured grid nodes (3x3x3)
        self.nodes = []
        for z in [0.0, 5.0, 10.0]:
            for y in [0.0, 5.0, 10.0]:
                for x in [0.0, 5.0, 10.0]:
                    self.nodes.append((x, y, z))

        self.nodes = np.array(self.nodes)

    # ---------------------------------------------------
    def test_classify_boundary_nodes_basic(self):

        groups = classify_boundary_nodes(self.nodes, self.extent, tol_ratio=1e-6)

        # corners should exist in multiple groups
        self.assertIn("left", groups)
        self.assertIn("right", groups)
        self.assertIn("bottom", groups)
        self.assertIn("top", groups)

        # check that left boundary is not empty
        self.assertGreater(len(groups["left"]), 0)

        # node at xmin=0 must be in left
        self.assertTrue(any(self.nodes[i][0] == 0.0 for i in groups["left"]))

    # ---------------------------------------------------
    def test_all_boundaries_present(self):

        groups = classify_boundary_nodes(self.nodes, self.extent)

        expected_keys = ["left", "right", "front", "back", "bottom", "top"]

        for k in expected_keys:
            self.assertIn(k, groups)

    # ---------------------------------------------------
    def test_tolerance_effect(self):

        # very strict tolerance → fewer boundary nodes
        groups_strict = classify_boundary_nodes(self.nodes, self.extent, tol_ratio=1e-12)

        # relaxed tolerance → more nodes included
        groups_loose = classify_boundary_nodes(self.nodes, self.extent, tol_ratio=1e-2)

        self.assertGreaterEqual(
            len(groups_loose["left"]),
            len(groups_strict["left"])
        )

    # ---------------------------------------------------
    def test_build_point_sets_valid(self):

        groups = classify_boundary_nodes(self.nodes, self.extent)

        point_sets = build_point_sets(groups, len(self.nodes))

        self.assertIsInstance(point_sets, dict)

        # ensure keys preserved
        self.assertIn("left", point_sets)
        self.assertIn("right", point_sets)

        # ensure numpy arrays
        for v in point_sets.values():
            self.assertIsInstance(v, np.ndarray)

    # ---------------------------------------------------
    def test_build_point_sets_invalid_index(self):

        groups = {
            "left": [0, 1, 2],
            "right": [999999]  # invalid
        }

        with self.assertRaises(ValueError):
            build_point_sets(groups, n_nodes=10)


if __name__ == "__main__":
    unittest.main()
