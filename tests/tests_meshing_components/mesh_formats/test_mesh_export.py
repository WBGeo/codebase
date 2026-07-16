import unittest
from unittest.mock import patch

import meshio
import numpy as np

from core.meshing_components.mesh_format.mesh_export import export_mesh_results, MeshFormatType
from core.meshing_components.mesh_format.exodus.Exo_format import exodus_type_for, MeshType as ExodusMeshType
from core.object_components import MeshResults, MeshType


class TestExportMeshResultsExodusType(unittest.TestCase):
    """
    export_mesh_results(format=Exodus) used to call export_mesh_results_to_exodus(mesh)
    with no explicit type -- defaulting to MeshType.UNSTRUCTURED regardless
    of the mesh's true type. For an implicit/structured mesh (always
    exactly 1 element block) that crashed outright pre-fix (1 < the old
    hardcoded NUM_SIDE_BLOCKS=6 minimum). These tests confirm the fix:
    export_mesh_results now passes the mesh's own recorded type.
    """

    def setUp(self):
        self.nodes = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ])
        self.elements = [meshio.CellBlock("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]

    def test_default_format_passes_correct_exodus_type_for_implicit_mesh(self):
        mesh = MeshResults(nodes=self.nodes, elements=self.elements, mesh_type=MeshType.IMPLICIT)
        with patch(
            "core.meshing_components.mesh_format.mesh_export.export_mesh_results_to_exodus"
        ) as mock_export:
            export_mesh_results(mesh, format=MeshFormatType.Exodus)
        mock_export.assert_called_once_with(mesh, type=exodus_type_for(mesh))
        self.assertEqual(exodus_type_for(mesh), ExodusMeshType.IMPLICIT)

    def test_implicit_mesh_through_default_download_mesh_does_not_raise(self):
        # Regression test for the historical crash: a 1-block implicit
        # mesh through the real (non-mocked) export pipeline.
        mesh = MeshResults(nodes=self.nodes, elements=self.elements, mesh_type=MeshType.IMPLICIT)
        result = export_mesh_results(mesh, format=MeshFormatType.Exodus)
        result.seek(0, 2)
        self.assertGreater(result.tell(), 0, "Exported file is empty")


if __name__ == "__main__":
    unittest.main()
