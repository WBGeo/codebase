import unittest
import numpy as np
import meshio
import os
import tempfile
from core.meshing_components.mesh_format.ansys.Ansys_format import   AnsysInputs,export_mesh_results_to_ansys
from core.object_components import MeshResults

class TestExportMeshResultsToAnsys(unittest.TestCase):

    def setUp(self):
        # Simple tetra mesh
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

    # Test mesh creation
    def test_create_mesh(self):
        ansys = AnsysInputs(self.nodes, self.elements)
        mesh = ansys.create_mesh()

        # Type check
        self.assertIsInstance(mesh, meshio.Mesh)

        # Points check
        self.assertEqual(mesh.points.shape, (4, 3))

        # Cell check
        self.assertEqual(len(mesh.cells), 1)
        self.assertEqual(mesh.cells[0].type, "tetra")

    # Unsupported element test
    def test_unsupported_elements(self):
        elements = [
            meshio.CellBlock("polygon", np.array([[0, 1, 2]]))  # unsupported
        ]

        ansys = AnsysInputs(self.nodes, elements)

        with self.assertRaises(ValueError):
            ansys.create_mesh()

    # Test file export
    def test_ansys_export(self):
        buf = export_mesh_results_to_ansys(self.mesh)

        # Buffer checks
        self.assertTrue(hasattr(buf, "filename"))
        self.assertTrue(buf.filename.endswith(".msh"))

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name

        # Read back with meshio
        loaded_mesh = meshio.read(tmp_path)

        self.assertEqual(len(loaded_mesh.points), 4)
        self.assertTrue(len(loaded_mesh.cells) > 0)

        # Cleanup
        os.remove(tmp_path)

########################################
if __name__ == "__main__":
    unittest.main()
