"""Regression tests for refinement at embedded source points."""
import unittest
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearSourceRefinement, FunctionSourceRefinement,
)
from core.meshing_components.explicit.unstructured.refinement_fields import source_point_size


class SourcePointSizeTests(unittest.TestCase):
    def test_no_refinement(self):
        self.assertIsNone(source_point_size(None))
        self.assertIsNone(source_point_size(Refinement()))

    def test_linear_refinement_uses_size_min(self):
        r = Refinement(sources=LinearSourceRefinement(SizeMin=2, SizeMax=20, DistMin=3, DistMax=30))
        self.assertEqual(source_point_size(r), 2.0)

    def test_function_at_zero_distance(self):
        r = Refinement(sources=FunctionSourceRefinement(expression='3 + DIST^2'))
        self.assertEqual(source_point_size(r), 3.0)

    def test_function_supports_math_functions(self):
        r = Refinement(sources=FunctionSourceRefinement(expression='sqrt(16) + DIST'))
        self.assertEqual(source_point_size(r), 4.0)

    def test_invalid_function_returns_none(self):
        r = Refinement(sources=FunctionSourceRefinement(expression='not_a_function(DIST)'))
        with self.assertLogs('core.meshing_components.explicit.unstructured.refinement_fields', level='WARNING'):
            self.assertIsNone(source_point_size(r))
