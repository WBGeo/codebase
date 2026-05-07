import unittest
import numpy as np
import meshio
import io
import zipfile
from core.meshing_components.mesh_format.vtm.VTM_format import VTMInputs, export_mesh_results_to_vtm


from core.object_components import MeshResults


class TestExportMeshResultsToVTM(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Simple synthetic mesh
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

        self.tetra = meshio.CellBlock(
            "tetra",
            np.array([
                [0, 1, 2, 4],
                [1, 2, 5, 6],
            ])
        )

        self.triangle = meshio.CellBlock(
            "triangle",
            np.array([
                [0, 1, 2],
                [0, 2, 3],
            ])
        )

        self.elements = [self.tetra, self.triangle]

    def test_create_multiblock(self):
        vtm = VTMInputs(self.nodes, self.elements)
        multi_block = vtm.create_mesh()

        # -----------------------------
        # Basic type check
        # -----------------------------
        import pyvista as pv
        self.assertIsInstance(multi_block, pv.MultiBlock)

        # -----------------------------
        # Correct way to check content
        # -----------------------------
        self.assertGreater(len(multi_block), 0)

        # -----------------------------
        # Check block names exist
        # -----------------------------
        names = list(multi_block.keys())
        self.assertTrue(any("Block_" in n for n in names))

        # -----------------------------
        # Check at least one UnstructuredGrid
        # -----------------------------
        grids = [multi_block[i] for i in range(len(multi_block))]
        self.assertTrue(all(g.n_points > 0 for g in grids))

    def test_block_structure(self):
        vtm = VTMInputs(self.nodes, self.elements)
        multi_block = vtm.create_mesh()

        # Each block should be a PyVista dataset
        for i in range(len(multi_block)):
            block = multi_block[i]
            self.assertTrue(hasattr(block, "cells"))
            self.assertTrue(hasattr(block, "points"))

    def test_vtm_save_to_disk(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "test.vtm")

            vtm = VTMInputs(self.nodes, self.elements, output_filename=file_path)
            vtm.create_mesh()

            self.assertTrue(os.path.exists(file_path))

    def test_zip_export(self):
        # Fake MeshResults
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets={}
        )


        buf = export_mesh_results_to_vtm(mesh)

        self.assertIsInstance(buf, io.BytesIO)

        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            files = zf.namelist()

        self.assertTrue(any(f.endswith(".vtm") for f in files))


if __name__ == "__main__":
    unittest.main()
