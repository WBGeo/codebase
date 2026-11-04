import unittest
import os
import zipfile
import tempfile
import gmsh
import numpy as np
from unittest.mock import patch
from core.meshing_components.explicit.unstructured.mesh_data import mesh_generator
from core.meshing_components.explicit.unstructured.refinement_fields import apply_background_fields, build_refinement_fields
from core.meshing_components.explicit.unstructured.refinement_mesh import LinearSourceRefinement, Refinement

class TestMeshGeneratorSources(unittest.TestCase):
    """Sources are passed as physical group tags (2000, 2001, ... as in fragment_surfaces);
    their point entities must be embedded, not points with the group tag's number."""

    def setUp(self):
        gmsh.finalize() if gmsh.isInitialized() else None
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

    def tearDown(self):
        gmsh.finalize()

    def test_source_point_is_a_mesh_node(self):
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 10)
        plane = occ.addRectangle(0, 0, 5, 10, 10)  # two volumes
        point = occ.addPoint(3, 3, 2)
        ov, ovv = occ.fragment([(3, box)], [(2, plane), (0, point)])
        occ.synchronize()
        source_point = [t for d, t in ovv[2] if d == 0][0]
        group = gmsh.model.addPhysicalGroup(0, [source_point], 2000)
        self.assertEqual(gmsh.model.getEntities(0).count((0, 2000)), 0)  # no point entity with the group's number

        nodes, cells, _ = mesh_generator(ov, [0, 10, 0, 10, 0, 10], [], [group], [], [], [],
                                         grid_litho=None, mesh_size=2.0, mapping_litho="none")

        self.assertLess(np.linalg.norm(np.asarray(nodes) - [3.0, 3.0, 2.0], axis=1).min(), 1e-9)

    def test_source_mesh_size_is_applied_to_source_point(self):
        """The source physical group must receive the requested local mesh size.

        Checking the Gmsh size assignment directly is more reliable than
        asserting a fixed ratio of nearby nodes, which varies with meshing
        algorithms and the global size settings.
        """
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 100, 100, 100)
        point = occ.addPoint(30, 30, 30)
        volumes, mapping = occ.fragment([(3, box)], [(0, point)])
        occ.synchronize()

        source_points = [tag for dim, tag in mapping[1] if dim == 0]
        self.assertTrue(source_points)
        group = gmsh.model.addPhysicalGroup(0, source_points, 2000)

        # Record the real Gmsh API calls without altering mesh generation.
        real_set_size = gmsh.model.mesh.setSize
        size_calls = []

        def record_size(entities, size):
            size_calls.append((list(entities), float(size)))
            return real_set_size(entities, size)

        with patch.object(gmsh.model.mesh, "setSize", side_effect=record_size):
            nodes, cells, _ = mesh_generator(
                volumes, [0, 100, 0, 100, 0, 100], [], [group], [], [], [],
                grid_litho=None, mesh_size=30.0, mapping_litho="none",
                source_mesh_size=2.0,
            )

        expected_points = {(0, tag) for tag in source_points}
        self.assertTrue(
            any(expected_points.issubset(set(entities)) and size == 2.0
                for entities, size in size_calls),
            f"Expected source points {expected_points} to receive mesh size 2.0; "
            f"observed setSize calls: {size_calls}",
        )
        self.assertGreater(len(nodes), 0)
        self.assertLess(
            np.linalg.norm(np.asarray(nodes) - [30.0, 30.0, 30.0], axis=1).min(),
            1e-9,
        )


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

    def tearDown(self):
        gmsh.finalize()
        self._tmp_dir.cleanup()

######################################
if __name__ == "__main__":
    unittest.main()
