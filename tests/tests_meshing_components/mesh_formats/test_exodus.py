import unittest
import tempfile
import os
import numpy as np
import meshio
import io

from core.meshing_components.mesh_format.exodus.Exo_format import (
    ExodusInput,
    export_mesh_results_to_exodus,
    exodus_type_for,
    MeshType
)
from core.object_components import MeshResults, MeshType as ObjectMeshType


class TestExportMeshResultsToExodus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Simple tetra mesh
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
            point_sets=cls.point_sets
        )

    # -----------------------------
    # ExodusInput creation test
    # -----------------------------
    def test_exodus_input_creation(self):

        exo = ExodusInput(
            mesh=self.mesh_results,
            mesh_type=MeshType.STRUCTURED
        )

        self.assertEqual(exo.mesh.nodes.shape, (4, 3))
        self.assertEqual(len(exo.mesh.elements), 1)

        self.assertEqual(exo.mesh_type, MeshType.STRUCTURED)
        self.assertEqual(exo.mesh_type.value, "str")

    # -----------------------------
    # Full export pipeline test
    # -----------------------------
    def test_export_mesh_results_to_exodus(self):

        result = export_mesh_results_to_exodus(
            self.mesh_results,
            type="str"   # KEEPING "type" as requested
        )

        self.assertIsInstance(result, io.BytesIO)
        self.assertTrue(hasattr(result, "filename"))
        self.assertTrue(result.filename.endswith(".exo"))

        result.seek(0, 2)
        size = result.tell()

        self.assertGreater(size, 0, "Exported file is empty")

    # -----------------------------
    # Invalid mesh type test
    # -----------------------------
    def test_invalid_mesh_type(self):

        with self.assertRaises(ValueError):

            ExodusInput(
                mesh=self.mesh_results,
                mesh_type="invalid_type"
            )


class TestUnstructuredDimensionBasedSplit(unittest.TestCase):
    """
    ExodusInput.write()'s UNSTRUCTURED branch used to slice the last
    NUM_SIDE_BLOCKS=6 blocks off positionally as "boundary" -- wrong for
    any mesh without exactly 6 non-volume blocks (e.g. any fault count
    other than the one originally tested, or wells/sources present). These
    tests build meshes with a non-6 non-volume block count and verify the
    real fix (splitting by block.dim instead of position).
    """

    @staticmethod
    def _write_and_reload(mesh_results):
        """Writes mesh_results via ExodusInput(mesh_type=UNSTRUCTURED) to a
        real temp .exo file and reads it back with meshio, so assertions
        can inspect what actually ended up in the exported element blocks
        (not just that write() didn't raise)."""
        fd, path = tempfile.mkstemp(suffix=".exo")
        os.close(fd)
        try:
            ExodusInput(mesh=mesh_results, mesh_type=MeshType.UNSTRUCTURED).write(path)
            return meshio.read(path, file_format="exodus")
        finally:
            # meshio's netCDF4-backed exodus reader can still hold the file
            # open briefly on Windows -- best-effort cleanup, not a test
            # failure condition (matches this codebase's existing
            # try/except OSError cleanup convention elsewhere, e.g.
            # sfepy_hydrothermal_run.py's _run_sfepy_input_file).
            try:
                os.remove(path)
            except OSError:
                pass

    def test_split_is_dimension_based_not_positional(self):
        # 2 tetra (volume) + 7 triangle (boundary) = 9 total blocks, 7 of
        # them non-volume -- more than the old hardcoded 6. Under the old
        # code, volume_blocks = all_blocks[:-6] would wrongly include one
        # triangle block (position 2) alongside the 2 real tetra blocks.
        # Triangles are each a real face of tetra1=[0,1,2,3] or
        # tetra2=[4,5,6,7] (per ExodusInput.FACE_TABLES["tetra"]) so every
        # side-set has exactly 1 matched element -- a triangle with no
        # matching face gives a zero-length side-set dimension, which hits
        # an unrelated, pre-existing netCDF4 quirk (0-length dimension
        # treated as a second "unlimited" dimension) that isn't what this
        # test is about.
        nodes = np.arange(12 * 3, dtype=float).reshape(12, 3)
        elements = [
            meshio.CellBlock("tetra", np.array([[0, 1, 2, 3]])),
            meshio.CellBlock("tetra", np.array([[4, 5, 6, 7]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 3]])),
            meshio.CellBlock("triangle", np.array([[1, 2, 3]])),
            meshio.CellBlock("triangle", np.array([[0, 2, 3]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("triangle", np.array([[4, 5, 7]])),
            meshio.CellBlock("triangle", np.array([[5, 6, 7]])),
            meshio.CellBlock("triangle", np.array([[4, 6, 7]])),
        ]
        mesh_results = MeshResults(nodes=nodes, elements=elements)

        reloaded = TestUnstructuredDimensionBasedSplit._write_and_reload(mesh_results)

        # Only the 2 real tetra elements should have been written -- no
        # triangle block leaked into the exported volume elements.
        self.assertEqual(set(cb.type for cb in reloaded.cells), {"tetra"})
        self.assertEqual(sum(len(cb.data) for cb in reloaded.cells), 2)

    def test_wells_and_sources_excluded_from_both_buckets(self):
        # 1 tetra (volume) + 1 triangle (boundary) + 1 line (well,
        # dim==1) + 1 vertex (source, dim==0). Neither the well nor the
        # source should end up in the exported volume elements -- matches
        # today's de facto behavior (they never matched anything in
        # build_side_sets_unstructured's face lookup either), just no
        # longer coincidentally dependent on a fixed block count.
        nodes = np.arange(6 * 3, dtype=float).reshape(6, 3)
        elements = [
            meshio.CellBlock("tetra", np.array([[0, 1, 2, 3]])),
            meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
            meshio.CellBlock("line", np.array([[3, 4]])),
            meshio.CellBlock("vertex", np.array([[5]])),
        ]
        mesh_results = MeshResults(nodes=nodes, elements=elements)

        reloaded = TestUnstructuredDimensionBasedSplit._write_and_reload(mesh_results)

        self.assertEqual(set(cb.type for cb in reloaded.cells), {"tetra"})
        self.assertEqual(sum(len(cb.data) for cb in reloaded.cells), 1)


class TestExodusTypeFor(unittest.TestCase):
    """exodus_type_for() -- translates a mesh's true provenance
    (core.object_components.MeshType, readable values) to Exo_format's own
    short-code MeshType, used so callers like mesh_export.py's default
    "Download Mesh" component don't have to hardcode/guess the mapping."""

    def setUp(self):
        self.nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.elements = [meshio.CellBlock("hexahedron", np.array([[0, 1, 2, 3, 0, 1, 2, 3]]))]

    def _mesh_with_type(self, mesh_type):
        return MeshResults(nodes=self.nodes, elements=self.elements, mesh_type=mesh_type)

    def test_implicit(self):
        self.assertEqual(exodus_type_for(self._mesh_with_type(ObjectMeshType.IMPLICIT)), MeshType.IMPLICIT)

    def test_structured(self):
        self.assertEqual(exodus_type_for(self._mesh_with_type(ObjectMeshType.STRUCTURED)), MeshType.STRUCTURED)

    def test_unstructured(self):
        self.assertEqual(exodus_type_for(self._mesh_with_type(ObjectMeshType.UNSTRUCTURED)), MeshType.UNSTRUCTURED)

    def test_none_falls_back_to_unstructured(self):
        # Matches export_mesh_results_to_exodus's own pre-existing default,
        # for a manually-constructed MeshResults with no provenance set.
        self.assertEqual(exodus_type_for(self._mesh_with_type(None)), MeshType.UNSTRUCTURED)


##############################################
if __name__ == "__main__":
    unittest.main()
