import unittest
import numpy as np
import meshio
import io

from core.meshing_components.mesh_format.exodus.Exo_format import (
    ExosInputs,
    export_mesh_results_to_exodus
)
from core.object_components import MeshResults


class TestExportMeshResultsToExodus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # -----------------------------
        # Create a simple tetra mesh
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

        cls.point_sets = {
            "left": np.array([0, 2], dtype=int)
        }

        # -----------------------------
        # Build MeshResults object
        # -----------------------------
        cls.mesh_results = MeshResults(
            nodes=cls.nodes,
            elements=cls.elements,
            point_sets=cls.point_sets
        )

    # =====================================================
    # Test ExosInputs initialization
    # =====================================================
    def test_exos_inputs_creation(self):

        exo = ExosInputs(
            self.nodes,
            self.elements,
            point_sets=self.point_sets
        )

        self.assertEqual(exo.nodes.shape, (4, 3))
        self.assertEqual(len(exo.elements), 1)
        self.assertIn("left", exo.point_sets)

    # =====================================================
    # Test mesh creation
    # =====================================================
    def test_create_mesh(self):

        exo = ExosInputs(
            self.nodes,
            self.elements,
            point_sets=self.point_sets
        )

        mesh = exo.create_mesh()

        self.assertIsInstance(mesh, meshio.Mesh)
        self.assertEqual(mesh.points.shape[0], 4)
        self.assertEqual(len(mesh.cells), 1)

    # =====================================================
    # Test full export pipeline
    # =====================================================
    def test_export_mesh_results_to_exodus(self):

        result = export_mesh_results_to_exodus(self.mesh_results)

        # -----------------------------
        # Check return type
        # -----------------------------
        self.assertIsInstance(result, io.BytesIO)

        # -----------------------------
        # Check filename attribute
        # -----------------------------
        self.assertTrue(hasattr(result, "filename"))
        self.assertTrue(result.filename.endswith(".exo"))

        # -----------------------------
        # Basic sanity: buffer not empty
        # -----------------------------
        result.seek(0, 2)  # go to end
        size = result.tell()

        self.assertGreater(size, 0, "Exported file is empty")

    # =====================================================
    # Test invalid node input
    # =====================================================
    def test_invalid_nodes(self):

        bad_nodes = np.array([[1.0, 2.0]])  # invalid shape

        with self.assertRaises(ValueError):
            ExosInputs(
                bad_nodes,
                self.elements,
                point_sets=self.point_sets
            )


if __name__ == "__main__":
    unittest.main()
