import unittest
import numpy as np
import meshio
import io
import zipfile
from core.meshing_components.mesh_format.VTK.VTK_format import VTKInputs, export_mesh_results_to_vtk


from core.object_components import MeshResults




class TestExportMeshResultsToVTK(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Simple cube-like mesh
        # -----------------------------
        self.nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])

        # two tetra blocks (different regions)
        self.elements = [
            meshio.CellBlock(
                "tetra",
                np.array([
                    [0, 1, 2, 4],
                    [1, 2, 5, 6],
                ])
            ),
            meshio.CellBlock(
                "tetra",
                np.array([
                    [0, 2, 3, 7],
                ])
            )
        ]

    def test_create_vtk_mesh(self):
        vtk_in = VTKInputs(self.nodes, self.elements)
        grid = vtk_in.create_mesh()

        # -----------------------------
        # Type check
        # -----------------------------
        import pyvista as pv
        self.assertIsInstance(grid, pv.UnstructuredGrid)

        # -----------------------------
        # Node check
        # -----------------------------
        self.assertEqual(grid.points.shape[1], 3)

        # -----------------------------
        # Cell data check
        # -----------------------------
        self.assertIn("RegionId", grid.cell_data)
        self.assertEqual(len(grid.cell_data["RegionId"]), grid.n_cells)

        # -----------------------------
        # Region ID correctness
        # -----------------------------
        self.assertTrue(np.all(grid.cell_data["RegionId"] > 0))

    def test_region_ids_match_blocks(self):
        vtk_in = VTKInputs(self.nodes, self.elements)
        grid = vtk_in.create_mesh()

        # We should have 2 regions (because 2 blocks)
        self.assertTrue(np.max(grid.cell_data["RegionId"]) == 2)

    def test_export_vtk_buffer(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets={}
        )

        buf = export_mesh_results_to_vtk(mesh)

        # -----------------------------
        # Buffer check
        # -----------------------------
        self.assertIsInstance(buf, io.BytesIO)

        buf.seek(0)
        content = buf.read()

        # VTK legacy files contain ASCII/binary header
        self.assertTrue(len(content) > 0)
        self.assertTrue(b"vtk" in content.lower() or b"dataset" in content.lower())

    def test_temp_file_cleanup(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets={}
        )

        buf = export_mesh_results_to_vtk(mesh)

        # buffer should be readable
        self.assertGreater(len(buf.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
