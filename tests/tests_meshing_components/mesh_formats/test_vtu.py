import unittest
import numpy as np
import meshio
import io

from core.meshing_components.mesh_format.vtu.VTU_format import VTUInputs, export_mesh_results_to_vtu
from core.object_components import MeshResults


class TestExportMeshResultsToVTU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # -----------------------------
        # Simple tetra mesh
        # -----------------------------
        cls.nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        cls.elements = [
            meshio.CellBlock(
                "tetra",
                np.array([[0, 1, 2, 3]])
            )
        ]

        cls.mesh = MeshResults(
            nodes=cls.nodes,
            elements=cls.elements
        )

    # =====================================================
    # Test VTUInputs initialization
    # =====================================================
    def test_vtu_inputs_init(self):

        vtu = VTUInputs(self.nodes, self.elements)

        self.assertEqual(vtu.nodes.shape, (4, 3))
        self.assertEqual(len(vtu.elements_block), 1)

    # =====================================================
    # Test mesh creation
    # =====================================================
    def test_create_mesh(self):

        vtu = VTUInputs(self.nodes, self.elements)
        mesh = vtu.create_mesh()

        self.assertIsInstance(mesh, meshio.Mesh)
        self.assertEqual(mesh.points.shape, (4, 3))
        self.assertEqual(len(mesh.cells), 1)

        # check cell data exists
        self.assertIn("gmsh:physical", mesh.cell_data)

    # =====================================================
    # Test full export pipeline
    # =====================================================
    def test_export_mesh_results_to_vtu(self):

        result = export_mesh_results_to_vtu(self.mesh)

        # -----------------------------
        # Check output type
        # -----------------------------
        self.assertIsInstance(result, io.BytesIO)

        # -----------------------------
        # Check filename
        # -----------------------------
        self.assertTrue(hasattr(result, "filename"))
        self.assertTrue(result.filename.endswith(".vtu"))

        # -----------------------------
        # Ensure file is not empty
        # -----------------------------
        result.seek(0, 2)
        size = result.tell()

        self.assertGreater(size, 0, "Exported VTU file is empty")

    # =====================================================
    # Test invalid node input
    # =====================================================
    def test_invalid_nodes(self):

        bad_nodes = np.array([[1.0, 2.0]])  # invalid Nx3 requirement

        with self.assertRaises(ValueError):
            VTUInputs(bad_nodes, self.elements)


if __name__ == "__main__":
    unittest.main()
