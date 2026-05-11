import os
import tempfile
import unittest

import numpy as np
import pyvista as pv

from core.object_components import SimulationResults
from core.simulation_components.output_format.vtk.unified_format_vtk import load_vtk_results


def _write_synthetic_vtk(path: str) -> None:
    """Write a minimal valid VTK file with one tetra and scalar point data."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([10])  # VTK_TETRA
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["p"] = np.array([0.0, 1.0, 0.5, 0.2])
    mesh.save(path)


class TestLoadVTKResults(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        vtk_path = os.path.join(self._tmp.name, "result.0.vtk")
        _write_synthetic_vtk(vtk_path)
        self.sim_output = {"output_dir": self._tmp.name, "is_temp": False}

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_simulation_results(self):
        results = load_vtk_results(self.sim_output)
        self.assertIsInstance(results, SimulationResults)

    def test_time_steps_loaded(self):
        results = load_vtk_results(self.sim_output)
        self.assertGreater(len(results.nodes_by_time), 0)

    def test_node_shape(self):
        results = load_vtk_results(self.sim_output)
        first_time = sorted(results.nodes_by_time.keys())[0]
        self.assertEqual(results.nodes_by_time[first_time].shape[1], 3)

    def test_cell_data_present(self):
        results = load_vtk_results(self.sim_output)
        first_time = sorted(results.nodes_by_time.keys())[0]
        self.assertIn(first_time, results.cells_by_time)

    def test_point_data_present(self):
        results = load_vtk_results(self.sim_output)
        first_time = sorted(results.nodes_by_time.keys())[0]
        self.assertIsInstance(results.node_data_by_time[first_time], dict)
        self.assertIn("p", results.node_data_by_time[first_time])


if __name__ == "__main__":
    unittest.main()