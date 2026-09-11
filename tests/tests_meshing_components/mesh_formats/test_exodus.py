import io
import os
import tempfile
import unittest

import meshio
import numpy as np

from core.meshing_components.mesh_format.exodus.Exo_format import (
    ExodusInput,
    export_mesh_results_to_exodus,
    MeshType,
)
from core.object_components import MeshResults


class TestExportMeshResultsToExodus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
            point_sets=cls.point_sets,
        )

    def test_exodus_input_creation(self):
        exo = ExodusInput(
            mesh=self.mesh_results,
            mesh_type=MeshType.STRUCTURED,
        )

        self.assertEqual(exo.mesh.nodes.shape, (4, 3))
        self.assertEqual(len(exo.mesh.elements), 1)
        self.assertEqual(exo.mesh_type, MeshType.STRUCTURED)
        self.assertEqual(exo.mesh_type.value, "str")

    def test_export_mesh_results_to_exodus(self):
        result = export_mesh_results_to_exodus(
            self.mesh_results,
            type="str",
        )

        self.assertIsInstance(result, io.BytesIO)
        self.assertTrue(hasattr(result, "filename"))
        self.assertTrue(result.filename.endswith(".exo"))

        result.seek(0, 2)
        self.assertGreater(result.tell(), 0, "Exported file is empty")

    def test_invalid_mesh_type(self):
        with self.assertRaises(ValueError):
            ExodusInput(
                mesh=self.mesh_results,
                mesh_type="invalid_type",
            )

    def test_export_rejects_invalid_mesh_type(self):
        with self.assertRaises(ValueError):
            export_mesh_results_to_exodus(
                self.mesh_results,
                type="invalid_type",
            )


class TestUnstructuredPositionalBoundarySplit(unittest.TestCase):
    """
    The current UNSTRUCTURED implementation reserves exactly the last six
    mesh blocks as boundary blocks. Everything before those six blocks is
    written as Exodus element blocks.
    """

    def setUp(self):
        self.nodes = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ])

        self.volume = meshio.CellBlock(
            "tetra",
            np.array([[0, 1, 2, 3]])
        )

        # All four tetra faces are valid surface blocks. Duplicate two of
        # them so that the current API receives exactly six boundary blocks.
        self.boundaries = [
            meshio.CellBlock("triangle", np.array([[0, 1, 3]])),
            meshio.CellBlock("triangle", np.array([[1, 2, 3]])),
            meshio.CellBlock("triangle", np.array([[0, 3, 2]])),
            meshio.CellBlock("triangle", np.array([[0, 2, 1]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 3]])),
            meshio.CellBlock("triangle", np.array([[1, 2, 3]])),
        ]

    def test_last_six_blocks_are_reserved_as_boundaries(self):
        mesh_results = MeshResults(
            nodes=self.nodes,
            elements=[self.volume, *self.boundaries],
        )

        exo = ExodusInput(
            mesh=mesh_results,
            mesh_type=MeshType.UNSTRUCTURED,
        )

        volume_blocks, geometry_blocks, side_sets = exo.classify_blocks(
            list(mesh_results.elements)
        )

        self.assertEqual(volume_blocks, [self.volume])
        self.assertEqual(geometry_blocks, [])
        self.assertGreater(len(side_sets), 0)

    def test_unstructured_requires_at_least_six_blocks(self):
        mesh_results = MeshResults(
            nodes=self.nodes,
            elements=[self.volume],
        )

        exo = ExodusInput(
            mesh=mesh_results,
            mesh_type=MeshType.UNSTRUCTURED,
        )

        with self.assertRaisesRegex(
            ValueError,
            "at least 6 blocks are required",
        ):
            exo.classify_blocks(list(mesh_results.elements))

    def test_seven_blocks_write_only_first_block(self):
        # This verifies the current positional contract directly:
        # with seven total blocks, the first is writable and the final six
        # are treated as boundary blocks.
        first_block = meshio.CellBlock(
            "tetra",
            np.array([[0, 1, 2, 3]])
        )

        mesh_results = MeshResults(
            nodes=self.nodes,
            elements=[first_block, *self.boundaries],
        )

        exo = ExodusInput(
            mesh=mesh_results,
            mesh_type=MeshType.UNSTRUCTURED,
        )

        volume_blocks, geometry_blocks, _ = exo.classify_blocks(
            list(mesh_results.elements)
        )

        self.assertEqual(len(volume_blocks), 1)
        self.assertIs(volume_blocks[0], first_block)
        self.assertEqual(geometry_blocks, [])


class TestStructuredAndImplicitClassification(unittest.TestCase):

    def setUp(self):
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
        self.block = meshio.CellBlock(
            "hexahedron",
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        )

    def _assert_all_blocks_are_written(self, mesh_type):
        mesh_results = MeshResults(
            nodes=self.nodes,
            elements=[self.block],
        )
        exo = ExodusInput(
            mesh=mesh_results,
            mesh_type=mesh_type,
        )

        volume_blocks, geometry_blocks, side_sets = exo.classify_blocks(
            list(mesh_results.elements)
        )

        self.assertEqual(volume_blocks, [self.block])
        self.assertEqual(geometry_blocks, [])
        self.assertGreater(len(side_sets), 0)

    def test_structured_writes_all_blocks(self):
        self._assert_all_blocks_are_written(MeshType.STRUCTURED)

    def test_implicit_writes_all_blocks(self):
        self._assert_all_blocks_are_written(MeshType.IMPLICIT)


if __name__ == "__main__":
    unittest.main()
