
import io
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


EXPORTERS = [
    ("abaqus.abaqus_format", "export_meshes_abaqus", "export_mesh_results_to_abaqus", ".inp", False),
    ("ansys.ansys_format", "export_meshes_ansys", "export_mesh_results_to_ansys", ".cdb", False),
    ("exodus.exo_format", "export_meshes_exodus", "export_mesh_results_to_exodus", ".exo", True),
    ("feflow.feflow_format", "export_meshes_feflow", "export_mesh_results_to_feflow", ".fem", False),
    ("gmsh.gmsh_format", "export_meshes_gmsh", "export_mesh_results_to_gmsh", ".msh", False),
    ("vtk.vtk_format", "export_meshes_vtk", "export_mesh_results_to_vtk", ".vtk", False),
    ("vtm.vtm_format", "export_meshes_vtm", "export_mesh_results_to_vtm", ".vtm", False),
    ("vtu.vtu_format", "export_meshes_vtu", "export_mesh_results_to_vtu", ".vtu", False),
]


def fake_mesh():
    return SimpleNamespace(
        nodes=np.array([[0., 0., 0.], [1., 1., 1.]]),
        elements=[object()],
    )


class TestHierarchicalExporters(unittest.TestCase):

    def test_all_exporters_write_two_meshes_and_two_boundary_frames(self):
        big = {
            "top": np.array([[0, 0., 0., 1.], [1, 1., 1., 1.]]),
            "bottom": np.array([[0, 0., 0., 0.]]),
        }
        small = {
            "top": np.array([[0, 10., 20., 30.], [1, 11., 21., 31.]]),
            "bottom": np.array([[0, 10., 20., 29.]]),
        }
        boundary = {"big_frame": big, "small_frame": small}

        for module_suffix, function_name, low_level_name, extension, takes_type in EXPORTERS:
            with self.subTest(exporter=function_name), tempfile.TemporaryDirectory() as td:
                module_name = "core.hierarchical_simulation.export_data." + module_suffix
                module = importlib.import_module(module_name)
                function = getattr(module, function_name)

                m1 = Path(td) / ("mesh1" + extension)
                m2 = Path(td) / ("mesh2" + extension)
                folder = Path(td) / "boundary"

                fake_buffer = io.BytesIO(b"mesh-data")

                with patch.object(module, low_level_name, return_value=fake_buffer) as mocked:
                    function(fake_mesh(), fake_mesh(), boundary, str(m1), str(m2), str(folder))

                self.assertEqual(m1.read_bytes(), b"mesh-data")
                self.assertEqual(m2.read_bytes(), b"mesh-data")
                for frame in ("big_frame", "small_frame"):
                    self.assertTrue((folder / frame / "top_boundary_nodes.csv").exists())
                    self.assertTrue((folder / frame / "bottom_boundary_nodes.csv").exists())

                # Same mesh1 node IDs must be preserved in both coordinate frames.
                top_big = np.loadtxt(
                    folder / "big_frame" / "top_boundary_nodes.csv",
                    delimiter=",", skiprows=1,
                )
                top_small = np.loadtxt(
                    folder / "small_frame" / "top_boundary_nodes.csv",
                    delimiter=",", skiprows=1,
                )
                np.testing.assert_array_equal(
                    np.atleast_2d(top_big)[:, 0],
                    np.atleast_2d(top_small)[:, 0],
                )
                self.assertEqual(mocked.call_count, 2)

    def test_all_exporters_keep_legacy_flat_boundary_format(self):
        boundary = {
            "top": np.array([[0, 0., 0., 1.], [1, 1., 1., 1.]]),
            "bottom": np.array([[0, 0., 0., 0.]]),
        }

        for module_suffix, function_name, low_level_name, extension, takes_type in EXPORTERS:
            with self.subTest(exporter=function_name), tempfile.TemporaryDirectory() as td:
                module = importlib.import_module(
                    "core.hierarchical_simulation.export_data." + module_suffix
                )
                function = getattr(module, function_name)
                m1 = Path(td) / ("mesh1" + extension)
                m2 = Path(td) / ("mesh2" + extension)
                folder = Path(td) / "boundary"

                with patch.object(
                    module, low_level_name, return_value=io.BytesIO(b"mesh-data")
                ):
                    function(fake_mesh(), fake_mesh(), boundary, str(m1), str(m2), str(folder))

                self.assertTrue((folder / "top_boundary_nodes.csv").exists())
                self.assertTrue((folder / "bottom_boundary_nodes.csv").exists())

    def test_exodus_forwards_mesh_type(self):
        module = importlib.import_module(
            "core.hierarchical_simulation.export_data.exodus.exo_format"
        )
        boundary = {"top": np.array([[0, 0., 0., 1.]])}

        with tempfile.TemporaryDirectory() as td:
            with patch.object(
                module,
                "export_mesh_results_to_exodus",
                return_value=io.BytesIO(b"x"),
            ) as mocked:
                module.export_meshes_exodus(
                    fake_mesh(), fake_mesh(), boundary,
                    str(Path(td) / "a.exo"),
                    str(Path(td) / "b.exo"),
                    str(Path(td) / "boundary"),
                    mesh_type="str",
                )

            self.assertEqual(mocked.call_args_list[0].args[1], "str")
            self.assertEqual(mocked.call_args_list[1].args[1], "str")


if __name__ == "__main__":
    unittest.main()
