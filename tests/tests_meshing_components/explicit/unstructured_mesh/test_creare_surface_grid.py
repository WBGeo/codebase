import unittest
import numpy as np
from scipy.interpolate import Rbf
import core.meshing_components.explicit.unstructured.create_grid_fragment_surface as create_grid_fragment_surface_module
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import (
    create_surface_grid, _evaluate_rbf_chunked,
)

class TestCreateSurfaceGrid(unittest.TestCase):

    def make_planar_surface(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        z = x + y
        points = np.column_stack([x, y, z])
        return [(0, points)]

    def make_vertical_surface(self):
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 0, 1])
        x = np.zeros_like(y)
        points = np.column_stack([x, y, z])
        return [(0, points)]

    # Output structure
    def test_output_structure(self):
        surfaces = self.make_planar_surface()

        result = create_surface_grid(surfaces)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape[1], 3)

    # Planar surface sanity
    def test_planar_surface_grid_properties(self):
        surfaces = self.make_planar_surface()

        result = create_surface_grid(surfaces)
        grid = result[0]

        self.assertFalse(np.isnan(grid).any())
        self.assertFalse(np.isinf(grid).any())

        self.assertAlmostEqual(grid[:, 0].min(), 0, places=2)
        self.assertAlmostEqual(grid[:, 0].max(), 1, places=2)
        self.assertAlmostEqual(grid[:, 1].min(), 0, places=2)
        self.assertAlmostEqual(grid[:, 1].max(), 1, places=2)

    # Vertical surface
    def test_vertical_surface(self):
        surfaces = self.make_vertical_surface()

        result = create_surface_grid(surfaces)
        grid = result[0]

        self.assertTrue(np.allclose(grid[:, 0], 0, atol=1e-6))
        self.assertFalse(np.isnan(grid).any())

    # Multiple surfaces
    def test_multiple_surfaces(self):
        s1 = self.make_planar_surface()[0]
        s2 = self.make_planar_surface()[0]

        result = create_surface_grid([s1, s2])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape[1], 3)
        self.assertEqual(result[1].shape[1], 3)

class TestEvaluateRbfChunked(unittest.TestCase):
    """_evaluate_rbf_chunked exists to avoid the dense (n_eval x n_source)
    matrix scipy.interpolate.Rbf.__call__ allocates in one shot -- confirmed
    to OOM on a real faulted model at high structural-grid resolution
    (25000 x 11180 eval/source points -> 2.08 GiB in one allocation).
    """

    def make_rbf(self, n_source=50, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 100, n_source)
        y = rng.uniform(0, 100, n_source)
        z = np.sin(x / 10) + np.cos(y / 10)
        rbf = Rbf(x, y, z, function='multiquadric', epsilon=2, smooth=1e-5)
        return rbf, n_source

    def test_matches_direct_call_when_chunking_not_needed(self):
        rbf, n_source = self.make_rbf()
        grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 8))

        direct = rbf(grid_x, grid_y)
        chunked = _evaluate_rbf_chunked(rbf, n_source, grid_x, grid_y)

        self.assertEqual(direct.shape, chunked.shape)
        np.testing.assert_array_equal(direct, chunked)

    def test_matches_direct_call_when_forced_to_chunk(self):
        rbf, n_source = self.make_rbf()
        grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 40), np.linspace(0, 100, 30))

        direct = rbf(grid_x, grid_y)

        original_budget = create_grid_fragment_surface_module._RBF_EVAL_CHUNK_BYTES
        create_grid_fragment_surface_module._RBF_EVAL_CHUNK_BYTES = 50  # forces chunk_size == 1
        try:
            chunked = _evaluate_rbf_chunked(rbf, n_source, grid_x, grid_y)
        finally:
            create_grid_fragment_surface_module._RBF_EVAL_CHUNK_BYTES = original_budget

        self.assertEqual(direct.shape, chunked.shape)
        # Chunked evaluation reduces the same per-point sum over source
        # points in smaller batches -- float64 summation isn't strictly
        # associative, so results match to machine precision, not bit-for-bit.
        np.testing.assert_allclose(direct, chunked, rtol=1e-10, atol=1e-10)

    def test_preserves_grid_shape_not_just_flat_size(self):
        rbf, n_source = self.make_rbf()
        grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 12), np.linspace(0, 100, 9))

        create_grid_fragment_surface_module._RBF_EVAL_CHUNK_BYTES = 50
        try:
            chunked = _evaluate_rbf_chunked(rbf, n_source, grid_x, grid_y)
        finally:
            create_grid_fragment_surface_module._RBF_EVAL_CHUNK_BYTES = 256 * 1024 ** 2

        self.assertEqual(chunked.shape, grid_x.shape)


###############################
if __name__ == "__main__":
    unittest.main()
