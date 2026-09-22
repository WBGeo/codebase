import inspect
import unittest

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


if __name__ == "__main__":
    unittest.main()
