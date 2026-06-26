import unittest
import numpy as np
import meshio
import os
import tempfile
from core.meshing_components.mesh_format.gmsh.GMSH_format import     GMSHInputs, export_mesh_results_to_gmsh
from core.object_components import MeshResults

class TestExportMeshResultsToGmsh(unittest.TestCase):

    def setUp(self):
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

    def test_create_mesh(self):
        gmsh = GMSHInputs(self.nodes, self.elements)
        mesh = gmsh.create_mesh()

        self.assertIsInstance(mesh, meshio.Mesh)

        # points
        self.assertEqual(mesh.points.shape, (4, 3))

        # cells
        self.assertEqual(len(mesh.cells), 1)
        self.assertEqual(mesh.cells[0].type, "tetra")  # ✅ FIX

        # cell_data
        self.assertIn("gmsh:physical", mesh.cell_data)
        self.assertIn("gmsh:geometrical", mesh.cell_data)

        # point_data
        self.assertIn("gmsh:dim_tags", mesh.point_data)

    def test_gmsh_file_export(self):
        gmsh = GMSHInputs(self.nodes, self.elements)
        mesh = gmsh.create_mesh()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.msh")

            # Optional: force ASCII for easier debugging
            mesh.write(filepath, file_format="gmsh", binary=False)

            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(os.path.getsize(filepath) > 0)

            # Now safe to read as text
            with open(filepath, "r") as f:
                content = f.read()

            self.assertIn("$MeshFormat", content)

    def test_export_mesh_results_to_gmsh(self):
        buf = export_mesh_results_to_gmsh(self.mesh)

        self.assertTrue(hasattr(buf, "read"))

        data = buf.read()
        self.assertTrue(len(data) > 0)

        self.assertTrue(buf.filename.endswith(".msh"))

###########################################
if __name__ == "__main__":
    unittest.main()
