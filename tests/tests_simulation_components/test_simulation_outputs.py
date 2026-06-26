import os
import tempfile
import unittest
import numpy as np
import pyvista as pv
from core.object_components import SimulationResults
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results
from core.simulation_components.output_format.exodus.unified_format_exodus import load_exodus_results


############
# VTK helper
############
def _write_synthetic_vtk(path: str) -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([10])  # tetra
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["p"] = np.array([0.0, 1.0, 0.5, 0.2])
    mesh.save(path)


#################
# EXODUS helper
#################
def _write_synthetic_exodus(path: str) -> None:
    """
    PyVista does NOT reliably write full Exodus time series,
    so we create a minimal EXO file via pyvista EXODUS writer fallback.
    """

    try:
        # simple structured grid (Exodus-compatible writer path)
        grid = pv.StructuredGrid()

        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 0, 0, 0])

        grid.points = np.c_[x, y, z]
        grid.dimensions = (2, 2, 1)

        grid.point_data["p"] = np.array([1.0, 2.0, 3.0, 4.0])

        # PyVista will route this through VTK Exodus writer if available
        grid.save(path)

    except Exception as e:
        raise RuntimeError(
            "Cannot create synthetic Exodus file in this environment."
        ) from e


############
# TEST CLASS
############
class TestLoadResults(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        # VTK setup
        vtk_path = os.path.join(self._tmp.name, "result.0.vtk")
        _write_synthetic_vtk(vtk_path)
        self.vtk_input = {"output_dir": self._tmp.name, "is_temp": False}

        # EXODUS setup
        exo_path = os.path.join(self._tmp.name, "result.e")
        try:
            _write_synthetic_exodus(exo_path)
            self.exo_input = exo_path
        except Exception:
            self.exo_input = None  # skip if unsupported

    def tearDown(self):
        self._tmp.cleanup()

    # VTK TESTS
    def test_vtk_load(self):
        results = load_vtk_results(self.vtk_input)
        self.assertIsInstance(results, SimulationResults)

    def test_vtk_has_data(self):
        results = load_vtk_results(self.vtk_input)
        t0 = sorted(results.nodes_by_time.keys())[0]
        self.assertEqual(results.nodes_by_time[t0].shape[1], 3)
        self.assertIn("p", results.node_data_by_time[t0])

    # EXODUS TESTS
    def test_exodus_load(self):
        if self.exo_input is None:
            self.skipTest("Exodus writer not available in this environment")

        results = load_exodus_results(self.exo_input)
        self.assertIsInstance(results, SimulationResults)

    def test_exodus_has_time_steps(self):
        if self.exo_input is None:
            self.skipTest("Exodus writer not available")

        results = load_exodus_results(self.exo_input)
        self.assertGreater(len(results.nodes_by_time), 0)

    def test_exodus_node_data(self):
        if self.exo_input is None:
            self.skipTest("Exodus writer not available")

        results = load_exodus_results(self.exo_input)
        t0 = sorted(results.nodes_by_time.keys())[0]

        self.assertEqual(results.nodes_by_time[t0].shape[1], 3)
        self.assertIsInstance(results.node_data_by_time[t0], dict)


#########################
if __name__ == "__main__":
    unittest.main()
