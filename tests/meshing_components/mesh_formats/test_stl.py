import unittest
import numpy as np
import meshio
import io
import zipfile
from core.meshing_components.mesh_format.stl.STL_format import STLInputs, export_mesh_results_to_stl


from core.object_components import MeshResults


class TestExportMeshResultsToSTL(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Simple tetra + triangle mesh
        # -----------------------------
        self.nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # surface triangles (group 1)
        self.triangles = meshio.CellBlock(
            "triangle",
            np.array([
                [0, 1, 2],
                [0, 2, 3],
            ])
        )

        # tetra volume (group 2)
        self.tetra = meshio.CellBlock(
            "tetra",
            np.array([
                [0, 1, 2, 4],
            ])
        )

        self.elements = [self.triangles, self.tetra]

    def test_stl_inputs_creation(self):
        stl = STLInputs(self.nodes, self.elements)

        # -----------------------------
        # Basic checks
        # -----------------------------
        self.assertEqual(stl.nodes.shape[1], 3)
        self.assertIn("gmsh:physical", stl.cell_data)

        # Each block should have physical tags
        self.assertEqual(len(stl.cell_data["gmsh:physical"]), 2)

    def test_surface_triangle_extraction(self):
        stl = STLInputs(self.nodes, self.elements)

        # must define output filename
        stl.output_filename = "test.stl"

        surface_nodes = stl._extract_and_save_triangle_groups()

        # -----------------------------
        # Should include triangle nodes
        # -----------------------------
        self.assertIsInstance(surface_nodes, set)
        self.assertTrue(len(surface_nodes) > 0)

    def test_interface_extraction(self):
        stl = STLInputs(self.nodes, self.elements)
        stl.output_filename = "test.stl"

        surface_nodes = stl._extract_and_save_triangle_groups()
        interfaces = stl._extract_interface_faces_by_group(surface_nodes)

        # -----------------------------
        # Interface structure check
        # -----------------------------
        self.assertIsInstance(interfaces, dict)

    def test_hexahedron_rejection(self):
        nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ])

        hexa = meshio.CellBlock(
            "hexahedron",
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        )

        stl = STLInputs(nodes, [hexa])
        stl.output_filename = "test.stl"

        with self.assertRaises(ValueError):
            stl.create_mesh()

    def test_full_export_zip(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets={}
        )


        buf = export_mesh_results_to_stl(mesh)

        # -----------------------------
        # Buffer check
        # -----------------------------
        self.assertIsInstance(buf, io.BytesIO)

        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            files = zf.namelist()

        # -----------------------------
        # Should contain STL files
        # -----------------------------
        self.assertTrue(any(f.endswith(".stl") for f in files))

    def test_stl_output_not_empty(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets={}
        )

        buf = export_mesh_results_to_stl(mesh)

        self.assertGreater(len(buf.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
