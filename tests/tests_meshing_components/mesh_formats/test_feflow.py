import unittest
import numpy as np
import meshio
import os
import tempfile
from core.meshing_components.mesh_format.feflow.Feflow_format import FeflowInputs, export_mesh_results_to_feflow

from core.object_components import MeshResults


class TestExportMeshResultsToFeflow(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Simple tetra mesh
        # -----------------------------
        self.nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.elements = [
            meshio.CellBlock(
                "tetra",
                np.array([[0, 1, 2, 3]])
            )
        ]

        self.mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

    # -------------------------------------------------
    # 1. Test FeflowInputs mesh creation
    # -------------------------------------------------
    def test_create_mesh(self):
        feflow = FeflowInputs(self.nodes, self.elements)
        mesh = feflow.create_mesh()

        self.assertIsInstance(mesh, meshio.Mesh)
        self.assertEqual(mesh.points.shape[0], 4)
        self.assertEqual(len(mesh.cells), 1)
        self.assertEqual(mesh.cells[0].type, "tetra")

    # -------------------------------------------------
    # 2. Test write (.fem file creation)
    # -------------------------------------------------
    def test_write_fem_file(self):
        feflow = FeflowInputs(self.nodes, self.elements)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.fem")

            feflow.write(filepath)

            # file must exist
            self.assertTrue(os.path.exists(filepath))

            # file must not be empty
            self.assertTrue(os.path.getsize(filepath) > 0)

            # quick sanity check: header exists
            with open(filepath, "r") as f:
                content = f.read()

            self.assertIn("PROBLEM:", content)
            self.assertIn("XYZCOOR", content)
            self.assertIn("END", content)

    # -------------------------------------------------
    # 3. Test WBGeo export wrapper (buffer output)
    # -------------------------------------------------
    def test_export_mesh_results_to_feflow(self):
        buf = export_mesh_results_to_feflow(self.mesh)

        self.assertIsInstance(buf, (bytes, bytearray, type(buf)))
        self.assertTrue(hasattr(buf, "read"))

        # filename check
        self.assertTrue(buf.filename.endswith(".fem"))

        # ensure data exists
        data = buf.read()
        self.assertTrue(len(data) > 0)


if __name__ == "__main__":
    unittest.main()
