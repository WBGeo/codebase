import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import meshio

from core.simulation_components.simulation_packages.sfepy.simulation_run import run_sfepy


class TestSfepyRun(unittest.TestCase):

    def create_mesh(self):
        """2 tetra blocks + 6 triangle blocks"""

        nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        cells = [
            meshio.CellBlock("tetra", np.array([[0, 1, 2, 3]])),
            meshio.CellBlock("tetra", np.array([[0, 1, 2, 3]])),

            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
        ]

        return meshio.Mesh(points=nodes, cells=cells)

    @patch("core.simulation_components.simulation_packages.sfepy.simulation_run.subprocess.Popen")
    @patch("core.simulation_components.simulation_packages.sfepy.simulation_run.meshio.read")
    @patch("core.simulation_components.simulation_packages.sfepy.simulation_run.export_mesh_results_to_exodus")
    def test_run_sfepy(self, mock_exodus, mock_read, mock_popen):

        # ------------------------
        # Mock Exodus export
        # ------------------------
        fake_buffer = MagicMock()
        fake_buffer.getbuffer.return_value = b"fake_exodus_data"
        mock_exodus.return_value = fake_buffer

        # ------------------------
        # Mock meshio.read output
        # ------------------------
        mock_mesh = MagicMock()

        mock_mesh.cells = [
            MagicMock(data=np.zeros((1, 4))),
            MagicMock(data=np.zeros((1, 4))),

            MagicMock(data=np.zeros((1, 3))),
            MagicMock(data=np.zeros((1, 3))),
            MagicMock(data=np.zeros((1, 3))),
            MagicMock(data=np.zeros((1, 3))),
            MagicMock(data=np.zeros((1, 3))),
            MagicMock(data=np.zeros((1, 3))),
        ]

        mock_read.return_value = mock_mesh

        # ------------------------
        # Mock subprocess
        # ------------------------
        process_mock = MagicMock()
        process_mock.wait.return_value = None
        mock_popen.return_value = process_mock

        # ------------------------
        # Input
        # ------------------------
        mesh = self.create_mesh()

        sfepy_input = {
            "input_file": "dummy_path/Hydro_thermal.py",
            "output_dir": None
        }

        # ------------------------
        # CALL (FIXED: mesh_type, NOT type)
        # ------------------------
        result = run_sfepy(
            sfepy_input_or_file=sfepy_input,
            mesh_test=mesh,
            mesh_type="unstr",
            output_dir=None
        )

        # ------------------------
        # Assertions
        # ------------------------
        self.assertIn("output_dir", result)
        self.assertIn("is_temp", result)
        self.assertTrue(result["is_temp"])

        mock_exodus.assert_called_once()
        mock_read.assert_called_once()
        mock_popen.assert_called_once()

        self.assertEqual(len(mock_mesh.cells), 8)


if __name__ == "__main__":
    unittest.main()
