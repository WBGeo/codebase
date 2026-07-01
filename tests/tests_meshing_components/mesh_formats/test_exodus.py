import unittest
import numpy as np
import meshio
import io

from core.meshing_components.mesh_format.exodus.Exo_format import (
    ExodusInput,
    export_mesh_results_to_exodus,
    MeshType
)
from core.object_components import MeshResults


class TestExportMeshResultsToExodus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Simple tetra mesh
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

        cls.mesh_results = MeshResults(
            nodes=cls.nodes,
            elements=cls.elements,
            point_sets=cls.point_sets
        )

    # -----------------------------
    # ExodusInput creation test
    # -----------------------------
    def test_exodus_input_creation(self):

        exo = ExodusInput(
            mesh=self.mesh_results,
            mesh_type=MeshType.STRUCTURED
        )

        self.assertEqual(exo.mesh.nodes.shape, (4, 3))
        self.assertEqual(len(exo.mesh.elements), 1)

        self.assertEqual(exo.mesh_type, MeshType.STRUCTURED)
        self.assertEqual(exo.mesh_type.value, "str")

    # -----------------------------
    # Full export pipeline test
    # -----------------------------
    def test_export_mesh_results_to_exodus(self):

        result = export_mesh_results_to_exodus(
            self.mesh_results,
            type="str"   # KEEPING "type" as requested
        )

        self.assertIsInstance(result, io.BytesIO)
        self.assertTrue(hasattr(result, "filename"))
        self.assertTrue(result.filename.endswith(".exo"))

        result.seek(0, 2)
        size = result.tell()

        self.assertGreater(size, 0, "Exported file is empty")

    # -----------------------------
    # Invalid mesh type test
    # -----------------------------
    def test_invalid_mesh_type(self):

        with self.assertRaises(ValueError):

            ExodusInput(
                mesh=self.mesh_results,
                mesh_type="invalid_type"
            )


##############################################
if __name__ == "__main__":
    unittest.main()
