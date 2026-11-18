import unittest
from unittest.mock import patch
import io

import meshio
import numpy as np

from core.meshing_components.mesh_format.mesh_export import export_mesh_results, MeshFormatType
from core.object_components import MeshResults, MeshType
from core.meshing_components.mesh_format.exodus.Exo_format import MeshType as ExodusMeshType


class TestExportMeshResultsExodus(unittest.TestCase):
    """The generic dispatcher must preserve MeshResults mesh provenance."""

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
        self.elements = [
            meshio.CellBlock(
                "hexahedron",
                np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
            )
        ]

    def test_dispatches_each_mesh_type_to_exodus(self):
        for origin, expected in (
            (MeshType.IMPLICIT, ExodusMeshType.IMPLICIT),
            (MeshType.STRUCTURED, ExodusMeshType.STRUCTURED),
            (MeshType.UNSTRUCTURED, ExodusMeshType.UNSTRUCTURED),
        ):
            with self.subTest(mesh_type=origin):
                mesh = MeshResults(nodes=self.nodes, elements=self.elements, mesh_type=origin)
                with patch(
                    "core.meshing_components.mesh_format.mesh_export.export_mesh_results_to_exodus"
                ) as mock_export:
                    export_mesh_results(mesh, format=MeshFormatType.Exodus)
                mock_export.assert_called_once_with(mesh, type=expected)

    def test_missing_mesh_type_keeps_legacy_default(self):
        mesh = MeshResults(nodes=self.nodes, elements=self.elements)
        with patch(
            "core.meshing_components.mesh_format.mesh_export.export_mesh_results_to_exodus"
        ) as mock_export:
            export_mesh_results(mesh, format=MeshFormatType.Exodus)
        mock_export.assert_called_once_with(mesh, type=ExodusMeshType.UNSTRUCTURED)

    def test_generic_dispatcher_writes_implicit_mesh_with_fewer_than_six_blocks(self):
        """Regression: generic Exodus export must not use unstructured block rules."""
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            mesh_type=MeshType.IMPLICIT,
        )
        result = export_mesh_results(mesh, format=MeshFormatType.Exodus)
        self.assertIsInstance(result, io.BytesIO)
        self.assertTrue(result.filename.endswith(".exo"))
        self.assertGreater(result.getbuffer().nbytes, 0)


if __name__ == "__main__":
    unittest.main()
