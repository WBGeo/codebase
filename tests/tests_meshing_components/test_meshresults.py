import unittest
import numpy as np
import meshio

from core.object_components import MeshResults


class TestMeshResults(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Simple tetra mesh
        # -----------------------------
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

    # =====================================================
    # 🧱 Initialization test
    # =====================================================
    def test_init(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

        self.assertEqual(mesh.nodes.shape, (4, 3))
        self.assertEqual(len(mesh.elements), 1)

    # =====================================================
    # 🔹 to_meshio conversion
    # =====================================================
    def test_to_meshio(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

        meshio_mesh = mesh.to_meshio()

        self.assertIsInstance(meshio_mesh, meshio.Mesh)
        self.assertEqual(meshio_mesh.points.shape, (4, 3))
        self.assertEqual(len(meshio_mesh.cells), 1)
        self.assertEqual(meshio_mesh.cells[0].type, "tetra")

    # =====================================================
    # 🔹 point_sets + cell_data handling
    # =====================================================
    def test_optional_data(self):
        point_sets = {"boundary": np.array([0, 1], dtype=np.int64)}
        cell_data = {"region": [np.array([1], dtype=np.int64)]}

        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            point_sets=point_sets,
            cell_data=cell_data
        )

        meshio_mesh = mesh.to_meshio()

        self.assertIn("boundary", meshio_mesh.point_sets)
        self.assertIn("region", meshio_mesh.cell_data)

    # =====================================================
    # 🔹 VTMInputs builder
    # =====================================================
    def test_vtm_in_property(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

        vtm = mesh.vtm_in

        # Lazy init check
        self.assertIsNotNone(vtm)

        # Ensure it's reused (cached)
        self.assertIs(mesh.vtm_in, vtm)

    # =====================================================
    # 🔹 mesh property (PyVista MultiBlock)
    # =====================================================
    def test_mesh_property(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

        pv_mesh = mesh.mesh

        # Should not be None
        self.assertIsNotNone(pv_mesh)

        # Should be cached
        self.assertIs(mesh.mesh, pv_mesh)

    # =====================================================
    # 🔹 mesh setter
    # =====================================================
    def test_mesh_setter(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements
        )

        dummy = "custom_mesh"
        mesh.mesh = dummy

        self.assertEqual(mesh.mesh, dummy)


if __name__ == "__main__":
    unittest.main()
