import unittest
import os

from core.Simulation.Output_format.VTK.unified_format_vtk import load_vtk_results
from core.object_components import SimulationResults


class TestLoadVTKResultsRealFolder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__)

        # 👇 real output directory
        cls.output_dir = os.path.join(base_dir, "sfepy_test_output")

        assert os.path.exists(cls.output_dir), "sfepy_test_output directory does not exist"

        # ✅ FIX: use dict instead of removed class
        cls.sim_output = {
            "output_dir": cls.output_dir,
            "is_temp": False
        }

    def test_load_vtk_results_real(self):

        results = load_vtk_results(self.sim_output)

        # -----------------------------
        # Type check
        # -----------------------------
        self.assertIsInstance(results, SimulationResults)

        # -----------------------------
        # Ensure VTK files were found
        # -----------------------------
        self.assertTrue(len(results.nodes_by_time) > 0, "No time steps loaded")

        # Pick one time step
        first_time = sorted(results.nodes_by_time.keys())[0]

        # -----------------------------
        # Check node structure
        # -----------------------------
        nodes = results.nodes_by_time[first_time]
        self.assertEqual(nodes.shape[1], 3)

        # -----------------------------
        # Check cell data exists
        # -----------------------------
        self.assertIn(first_time, results.cells_by_time)

        # -----------------------------
        # Check PyVista data extraction
        # -----------------------------
        self.assertIsInstance(results.node_data_by_time[first_time], dict)

        print(f"[INFO] Loaded {len(results.nodes_by_time)} time steps successfully")


if __name__ == "__main__":
    unittest.main()
