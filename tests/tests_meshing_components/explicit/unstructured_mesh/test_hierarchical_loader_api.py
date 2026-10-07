import inspect
import os
import tempfile
import unittest

import numpy as np

from core.meshing_components.explicit.unstructured.mesh_data import (
    load_wells_from_csv,
    load_sources_from_csv,
    load_shafts_from_csv,
    load_planes_from_csv,
    load_ellipses_from_csv,
    load_triangulations_planes_from_csv,
)


class TestHierarchicalLoaderAPI(unittest.TestCase):

    def setUp(self):
        self.loaders = (
            load_wells_from_csv,
            load_sources_from_csv,
            load_shafts_from_csv,
            load_planes_from_csv,
            load_ellipses_from_csv,
            load_triangulations_planes_from_csv,
        )
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _csv(self, name, rows):
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        return path

    def test_all_loaders_expose_key_hierarchical(self):
        for loader in self.loaders:
            with self.subTest(loader=loader.__name__):
                self.assertIn(
                    "key_hierarchical",
                    inspect.signature(loader).parameters,
                )

    def test_hierarchical_mode_is_opt_in(self):
        for loader in self.loaders:
            with self.subTest(loader=loader.__name__):
                parameter = inspect.signature(loader).parameters["key_hierarchical"]
                self.assertIs(parameter.default, False)

    def test_wells_flat_and_hierarchical_return_shapes(self):
        flat_file = self._csv("wells.csv", [
            "w1,0,0,0",
            "w1,1,1,1",
            "w2,2,2,2",
            "w2,3,3,3",
        ])
        hierarchical_file = self._csv("wells_h.csv", [
            "w1,0,0,0,1",
            "w1,1,1,1,1",
            "w2,2,2,2,2",
            "w2,3,3,3,2",
        ])

        flat = load_wells_from_csv(flat_file)
        hierarchical = load_wells_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, list)
        self.assertNotIsInstance(flat, dict)
        self.assertEqual(flat, [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                                (2.0, 2.0, 2.0, 3.0, 3.0, 3.0)])
        self.assertIsInstance(hierarchical, dict)
        self.assertEqual(set(hierarchical), {1, 2})
        self.assertEqual(hierarchical[1], [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)])
        self.assertEqual(hierarchical[2], [(2.0, 2.0, 2.0, 3.0, 3.0, 3.0)])

    def test_sources_flat_and_hierarchical_return_shapes(self):
        flat_file = self._csv("sources.csv", ["s1,1,2,3", "s2,4,5,6"])
        hierarchical_file = self._csv("sources_h.csv", ["s1,1,2,3,1", "s2,4,5,6,2"])

        flat = load_sources_from_csv(flat_file)
        hierarchical = load_sources_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, list)
        self.assertEqual(flat, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        self.assertEqual(hierarchical, {1: [(1.0, 2.0, 3.0)], 2: [(4.0, 5.0, 6.0)]})

    def test_shafts_flat_and_hierarchical_return_shapes(self):
        flat_file = self._csv("shafts.csv", [
            "s1,0,0,0,0,0,1,2",
            "s2,1,1,1,1,0,0,3",
        ])
        hierarchical_file = self._csv("shafts_h.csv", [
            "s1,0,0,0,0,0,1,2,1",
            "s2,1,1,1,1,0,0,3,2",
        ])

        flat = load_shafts_from_csv(flat_file)
        hierarchical = load_shafts_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, list)
        self.assertEqual(set(hierarchical), {1, 2})
        self.assertEqual(len(hierarchical[1]), 1)
        self.assertEqual(len(hierarchical[2]), 1)

    def test_planes_flat_and_hierarchical_return_shapes(self):
        flat_rows = [
            "p1,0,0,0", "p1,1,0,0", "p1,1,1,0", "p1,0,1,0",
            "p2,0,0,1", "p2,1,0,1", "p2,1,1,1", "p2,0,1,1",
        ]
        hierarchical_rows = [
            "p1,0,0,0,1", "p1,1,0,0,1", "p1,1,1,0,1", "p1,0,1,0,1",
            "p2,0,0,1,2", "p2,1,0,1,2", "p2,1,1,1,2", "p2,0,1,1,2",
        ]
        flat_file = self._csv("planes.csv", flat_rows)
        hierarchical_file = self._csv("planes_h.csv", hierarchical_rows)

        flat = load_planes_from_csv(flat_file)
        hierarchical = load_planes_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, list)
        self.assertEqual(len(flat), 2)
        self.assertEqual(set(hierarchical), {1, 2})
        self.assertEqual(len(hierarchical[1]), 1)
        self.assertEqual(len(hierarchical[2]), 1)

    def test_ellipses_flat_and_hierarchical_return_shapes(self):
        flat_file = self._csv("ellipses.csv", ["0,0,0,2,1", "1,1,1,3,2"])
        hierarchical_file = self._csv("ellipses_h.csv", ["0,0,0,2,1,1", "1,1,1,3,2,2"])

        flat = load_ellipses_from_csv(flat_file)
        hierarchical = load_ellipses_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, list)
        self.assertEqual(len(flat), 2)
        self.assertEqual(set(hierarchical), {1, 2})
        self.assertEqual(len(hierarchical[1]), 1)
        self.assertEqual(len(hierarchical[2]), 1)

    def test_triangulations_flat_and_hierarchical_return_shapes(self):
        flat_file = self._csv("triangulation.csv", ["0,0,0", "1,0,0", "0,1,0"])
        hierarchical_file = self._csv("triangulation_h.csv", [
            "0,0,0,1", "1,0,0,1", "0,1,0,1",
            "0,0,1,2", "1,0,1,2", "0,1,1,2",
        ])

        flat = load_triangulations_planes_from_csv(flat_file)
        hierarchical = load_triangulations_planes_from_csv(hierarchical_file, key_hierarchical=True)

        self.assertIsInstance(flat, np.ndarray)
        self.assertEqual(flat.shape, (3, 4))
        self.assertIsInstance(hierarchical, dict)
        self.assertEqual(set(hierarchical), {1, 2})
        self.assertEqual(hierarchical[1].shape, (3, 4))
        self.assertEqual(hierarchical[2].shape, (3, 4))
        self.assertTrue(np.all(hierarchical[1][:, 3] == 1.0))
        self.assertTrue(np.all(hierarchical[2][:, 3] == 1.0))


if __name__ == "__main__":
    unittest.main()
