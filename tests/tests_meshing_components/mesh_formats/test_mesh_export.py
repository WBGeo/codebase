import unittest
from unittest.mock import patch

import meshio
import numpy as np

from core.meshing_components.mesh_format.mesh_export import export_mesh_results, MeshFormatType
from core.object_components import MeshResults, MeshType


class TestExportMeshResultsExodus(unittest.TestCase):
    """
    Tests for the current Download Mesh -> Exodus integration.

    The current mesh_export.export_mesh_results() delegates directly to
    export_mesh_results_to_exodus(mesh) without translating MeshResults.mesh_type
    or passing an explicit Exodus mesh type.
    """

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

    def test_default_format_calls_exodus_export_without_explicit_type(self):
        mesh = MeshResults(
            nodes=self.nodes,
            elements=self.elements,
            mesh_type=MeshType.IMPLICIT,
        )

        with patch(
            "core.meshing_components.mesh_format.mesh_export.export_mesh_results_to_exodus"
        ) as mock_export:
            export_mesh_results(mesh, format=MeshFormatType.Exodus)

        mock_export.assert_called_once_with(mesh)


if __name__ == "__main__":
    unittest.main()
