import unittest
import numpy as np
from core.meshing_components.explicit.unstructured.mesh_data import validate_sources

class TestValidateSources(unittest.TestCase):

    # VALID CASE
    def test_valid_sources(self):
        extent = [0, 10, 0, 10, 0, 10]

        sources = [
            (1, 1, 1),
            (5, 5, 5),
            (10, 10, 10)  # boundary is allowed
        ]

        # should NOT raise
        validate_sources(sources, extent)


    # OUTSIDE SOURCE
    def test_source_outside_raises(self):
        extent = [0, 10, 0, 10, 0, 10]

        sources = [
            (1, 1, 1),
            (20, 5, 5)  # outside
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_sources(sources, extent)

        self.assertIn("Source", str(ctx.exception))
        self.assertIn("outside extent", str(ctx.exception))


    # EMPTY INPUT
    def test_empty_sources(self):
        extent = [0, 10, 0, 10, 0, 10]

        # should NOT raise
        validate_sources([], extent)


    # INVALID EXTENT
    def test_invalid_extent(self):
        sources = [(1, 1, 1)]

        with self.assertRaises(ValueError):
            validate_sources(sources, [0, 10])  # invalid format


    # MULTIPLE SOURCES WITH ONE INVALID
    def test_multiple_sources_one_invalid(self):
        extent = [0, 10, 0, 10, 0, 10]

        sources = [
            (1, 1, 1),
            (2, 2, 2),
            (5, 5, 20)  # invalid
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_sources(sources, extent)

        # ensure correct object label
        self.assertIn("Source", str(ctx.exception))

#######################################
if __name__ == "__main__":
    unittest.main()
