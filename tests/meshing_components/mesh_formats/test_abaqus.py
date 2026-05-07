import unittest
import numpy as np
import meshio
import os
import tempfile
from core.meshing_components.mesh_format.abaqus.Abaqus_format import   AbaqusInputs,export_mesh_results_to_abaqus
from core.object_components import MeshResults



class TestExportMeshResultsToAbaqus(unittest.TestCase):

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

    # =====================================================
    # 🧱 Test mesh creation
    # =====================================================
    def test_create_mesh(self):
        abaqus = AbaqusInputs(self.nodes, self.elements)
        mesh = abaqus.create_mesh()

        # -----------------------------
        # Type check
        # -----------------------------
        self.assertIsInstance(mesh, meshio.Mesh)

        # -----------------------------
        # Points check
        # -----------------------------
        self.assertEqual(mesh.points.shape, (4, 3))

        # -----------------------------
        # Cell check
        # -----------------------------
        self.assertEqual(len(mesh.cells), 1)
        self.assertEqual(mesh.cells[0].type, "tetra")

    # =====================================================
    # 🚫 Unsupported element handling
    # =====================================================
    def test_unsupported_elements(self):
        elements = [
            meshio.CellBlock("polygon", np.array([[0, 1, 2]]))  # unsupported
        ]

        abaqus = AbaqusInputs(self.nodes, elements)
        mesh = abaqus.create_mesh()

        # Should not crash, but result in empty cells
        self.assertEqual(len(mesh.cells), 0)

    # =====================================================
    # 💾 Test writing .inp file
    # =====================================================
    def test_write_inp_file(self):
        abaqus = AbaqusInputs(self.nodes, self.elements)

        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as tmp:
            tmp_path = tmp.name

        abaqus.write(tmp_path)

        # -----------------------------
        # Check file exists
        # -----------------------------
        self.assertTrue(os.path.exists(tmp_path))

        # -----------------------------
        # Check file content (text!)
        # -----------------------------
        with open(tmp_path, "r") as f:
            content = f.read()

        self.assertIn("*NODE", content)
        self.assertIn("*ELEMENT", content)
        self.assertIn("*MATERIAL", content)

        # Cleanup
        os.remove(tmp_path)

    # =====================================================
    # 🚀 Test export function
    # =====================================================
    def test_abaqus_export(self):
        buf = export_mesh_results_to_abaqus(self.mesh)

        # -----------------------------
        # Buffer checks
        # -----------------------------
        self.assertTrue(hasattr(buf, "filename"))
        self.assertTrue(buf.filename.endswith(".inp"))

        # -----------------------------
        # Save and inspect content
        # -----------------------------
        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name

        with open(tmp_path, "r") as f:
            content = f.read()

        self.assertIn("*NODE", content)
        self.assertIn("*ELEMENT", content)

        # Cleanup
        os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
