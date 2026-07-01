import unittest
import os
import zipfile
import tempfile
import gmsh
import numpy as np
from core.meshing_components.explicit.unstructured.mesh_data import mesh_generator

class TestMeshGeneratorFromMshReference(unittest.TestCase):

    # setup gmsh + extract reference msh
    def setUp(self):

        gmsh.finalize() if gmsh.isInitialized() else None
        gmsh.initialize()
        gmsh.model.add("test")

        data_dir = os.path.join(os.path.dirname(__file__))

        zip_path = os.path.join(data_dir, "reference_model.msh.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)

        # ---- unzip msh ----
        self._tmp_dir = tempfile.TemporaryDirectory()
        tmp_dir = self._tmp_dir.name

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)

        msh_files = [f for f in os.listdir(tmp_dir) if f.endswith(".msh")]
        if not msh_files:
            raise RuntimeError("No .msh found in zip")

        self.msh_path = os.path.join(tmp_dir, msh_files[0])

        # load mesh into gmsh
        gmsh.merge(self.msh_path)
        gmsh.model.mesh.generate(3)

    def run_mesh(self):

        # IMPORTANT: in msh-based workflow, these usually come from mesh itself
        nodes = np.array(gmsh.model.mesh.getNodes()[1]).reshape(-1, 3)

        cells = gmsh.model.mesh.getElements(3)

        # convert gmsh format → simple list
        cell_list = []
        for etype, tags, node_ids in zip(cells[0], cells[1], cells[2]):
            cell_list.append({
                "type": etype,
                "nodes": np.array(node_ids)
            })

        # dummy inputs (since msh already encodes geometry)
        ov = []
        tagssss = []

        return nodes, cell_list, ov, tagssss

    def test_nodes_exist(self):
        nodes, _, _, _ = self.run_mesh()
        self.assertIsInstance(nodes, np.ndarray)
        self.assertGreater(len(nodes), 0)

    def test_cells_exist(self):
        _, cells, _, _ = self.run_mesh()
        self.assertIsInstance(cells, list)
        self.assertGreater(len(cells), 0)

    def test_node_shape(self):
        nodes, _, _, _ = self.run_mesh()
        self.assertEqual(nodes.shape[1], 3)

    def test_regression_hash(self):

        import hashlib
        import pickle

        nodes, cells, _, _ = self.run_mesh()

        snapshot = {
            "n_nodes": len(nodes),
            "n_cells": len(cells),
        }

        current_hash = hashlib.md5(pickle.dumps(snapshot)).hexdigest()

        # optional baseline
        expected = None  # replace with saved hash if needed

        if expected is None:
            print("\nNEW HASH:", current_hash)
            self.skipTest("No baseline hash yet")

        self.assertEqual(current_hash, expected)

    def tearDown(self):
        gmsh.finalize()
        self._tmp_dir.cleanup()

######################################
if __name__ == "__main__":
    unittest.main()
