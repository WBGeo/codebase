import tempfile

import netCDF4 as nc
import unittest
from pathlib import Path
from unittest.mock import patch

import meshio
import numpy as np
import pandas as pd

from core.meshing_components.mesh_format.exodus.Exo_format import ExodusInput
from core.meshing_components.mesh_format.mesh_export import (
    MeshFormatType, export_mesh_results,
)
from core.meshing_components.explicit.unstructured.mesh_data import (
    LithoMappingMode,
    load_csv_surface_coordinates,
    create_unstructured_mesh_data,
)


MODULE = "core.meshing_components.explicit.unstructured.mesh_data"
EXTENT = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)


def _write_surface(path: Path, z: float = 5.0):
    pd.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "z": [z, z, z, z],
        }
    ).to_csv(path, index=False)


class SurfaceCsvInputTestCase(unittest.TestCase):
    """Regression tests for direct x/y/z CSV geological-surface input."""

    def _run_csv_route(self, csv_dir, mapping_litho="none", extent=EXTENT, extent_z=None, export_ready=False, faulted_blocks=False):
        # Match the mocked mesh to the domain resolved by the CSV route.
        # The implementation now clips cells to that domain, so a fixed
        # 0..10 tetrahedron is invalid when CSV bounds are inferred.
        if extent is None:
            csv_files = sorted(Path(csv_dir).glob("*.csv"))
            xyz = np.concatenate(
                [pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=float)
                 for path in csv_files],
                axis=0,
            )
            low, high = xyz.min(axis=0), xyz.max(axis=0)
            if extent_z is not None:
                low[2], high[2] = map(float, extent_z)
        else:
            bounds = np.asarray(extent, dtype=float)
            low = bounds[[0, 2, 4]]
            high = bounds[[1, 3, 5]]

        x0, y0, z0 = low
        x1, y1, z1 = high
        cube_nodes = np.array(
            [
                [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
            ],
            dtype=float,
        )
        tetra = meshio.CellBlock("tetra", np.array([[0, 1, 2, 4]], dtype=int))

        if export_ready:
            # One volume plus six positional boundary blocks required by
            # the current UNSTRUCTURED Exodus exporter.
            cells = [tetra] + [
                meshio.CellBlock("triangle", np.array([face], dtype=int))
                for face in (
                    [0, 1, 2], [0, 1, 4], [0, 2, 4],
                    [1, 2, 4], [0, 1, 2], [0, 1, 4],
                )
            ]
            if faulted_blocks:
                # Real fault triangle block immediately before the six exterior
                # boundary blocks: the 7+-block layout missing from the old test.
                cells.insert(1, meshio.CellBlock(
                    "triangle", np.array([[0, 1, 2]], dtype=int)
                ))
        else:
            cells = [tetra]

        with (
            patch(f"{MODULE}.import_structural_surfaces") as import_structural_surfaces,
            patch(f"{MODULE}.create_surface_grid", return_value=[np.zeros((4, 3))]) as create_grid,
            patch(f"{MODULE}.import_surfaces", return_value=([101, 102], EXTENT)),
            patch(
                f"{MODULE}.fragment_surfaces",
                return_value=([(3, 1)], [], [], [], [], [], []),
            ) as fragment,
            patch(
                f"{MODULE}.mesh_generator",
                return_value=(cube_nodes, cells, {}),
            ) as mesh_generator,
            patch(f"{MODULE}.gmsh.initialize"),
            patch(f"{MODULE}.gmsh.finalize"),
            patch(f"{MODULE}.gmsh.model.occ.synchronize"),
        ):
            kwargs = {
                "geomodel_result": load_csv_surface_coordinates(csv_dir),
                "mapping_litho": mapping_litho,
            }
            if extent is not None:
                kwargs["extent"] = extent
            if extent_z is not None:
                kwargs["extent_z"] = extent_z
            result = create_unstructured_mesh_data(**kwargs)

        return result, import_structural_surfaces, create_grid, fragment, mesh_generator

    def test_surface_csv_dir_to_generic_exodus_export(self):
        """CSV input -> MeshResults -> generic dispatcher -> real Exodus writer.

        Geometry/meshing are mocked to keep the test deterministic, but CSV
        parsing, mesh provenance, dispatch, and Exodus serialization are real.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 4.0)
            _write_surface(root / "fault1.csv", 6.0)
            mesh, _, _, _, _ = self._run_csv_route(root, export_ready=True)
            self.assertEqual(str(mesh.mesh_type), "unstructured")
            exported = export_mesh_results(mesh, format=MeshFormatType.Exodus)
            self.assertTrue(exported.filename.endswith(".exo"))
            self.assertGreater(exported.getbuffer().nbytes, 0)

    def test_faulted_csv_exodus_boundary_modes(self):
        """CSV -> mesh -> real Exodus writer with a fault and six box faces.

        The mesh generator is mocked, but CSV parsing, mesh assembly, block
        classification and NetCDF Exodus serialization are exercised.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 4.0)
            _write_surface(root / "fault1.csv", 6.0)
            mesh, _, create_grid, fragment, _ = self._run_csv_route(
                root, export_ready=True, faulted_blocks=True
            )
            self.assertEqual(len(create_grid.call_args.args[0]), 2)
            self.assertIn("fault1.csv", fragment.call_args.args[2])
            self.assertEqual(len(mesh.elements), 8)  # tetra + fault + six faces

            for mode, expected_elements, expected_sidesets in (
                ("fixed_trailing", 2, 6),
                ("dim", 1, 7),
            ):
                with self.subTest(boundary_mode=mode):
                    path = root / f"faulted_{mode}.exo"
                    ExodusInput(mesh, boundary_mode=mode).write(str(path))
                    with nc.Dataset(str(path), "r") as ds:
                        self.assertEqual(len(ds.dimensions["num_elem"]), expected_elements)
                        self.assertEqual(len(ds.dimensions["num_side_sets"]), expected_sidesets)
                        self.assertEqual(len(ds.dimensions["num_el_blk"]), expected_elements)
                        block_types = [
                            ds.variables[f"connect{i}"].getncattr("elem_type")
                            for i in range(1, expected_elements + 1)
                        ]
                        if mode == "fixed_trailing":
                            self.assertEqual(block_types, ["TETRA4", "TRI3"])
                        else:
                            self.assertEqual(block_types, ["TETRA4"])
                        # All boundary faces must match the real tetrahedron.
                        for i in range(1, expected_sidesets + 1):
                            self.assertEqual(len(ds.variables[f"elem_ss{i}"][:]), 1)
                            self.assertEqual(int(ds.variables[f"elem_ss{i}"][0]), 1)

    def test_csv_mode_bypasses_structural_surface_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 4.0)
            self._run_csv_route(root)[1].assert_not_called()

    def test_create_surface_grid_receives_only_surface_id_and_xyz(self):
        """Guards against the previous 'too many values to unpack' regression."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 4.0)
            _write_surface(root / "fault1.csv", 6.0)

            _, _, create_grid, _, _ = self._run_csv_route(root)
            csv_surfaces = create_grid.call_args.args[0]

            self.assertEqual(len(csv_surfaces), 2)
            for item in csv_surfaces:
                self.assertEqual(len(item), 2)
                surface_id, xyz = item
                self.assertIsInstance(surface_id, str)
                self.assertEqual(xyz.shape, (4, 3))

    def test_fault_filename_is_stored_as_reference_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 4.0)
            _write_surface(root / "fault1.csv", 6.0)
            _write_surface(root / "fault_main.csv", 7.0)

            _, _, create_grid, fragment, _ = self._run_csv_route(root)
            csv_surfaces = create_grid.call_args.args[0]
            index_by_name = {name: i for i, (name, _) in enumerate(csv_surfaces)}

            reference_surfaces = fragment.call_args.args[2]
            self.assertEqual(
                reference_surfaces,
                {
                    "fault1.csv": index_by_name["fault1.csv"],
                    "fault_main.csv": index_by_name["fault_main.csv"],
                },
            )
            self.assertNotIn("surface1.csv", reference_surfaces)

    def test_fault_recognition_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "Fault2.csv", 6.0)
            _write_surface(root / "surface1.csv", 4.0)

            _, _, _, fragment, _ = self._run_csv_route(root)
            self.assertIn("Fault2.csv", fragment.call_args.args[2])

    def test_csv_mode_forces_mapping_litho_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")

            _, _, _, _, mesh_generator = self._run_csv_route(
                root, mapping_litho="automatic_centers"
            )
            self.assertEqual(
                mesh_generator.call_args.kwargs["mapping_litho"],
                LithoMappingMode.NONE,
            )

    def test_csv_mode_infers_extent_from_all_surface_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "x": [-5.0, 2.0, 8.0],
                    "y": [3.0, -4.0, 6.0],
                    "z": [100.0, 120.0, 110.0],
                }
            ).to_csv(root / "surface1.csv", index=False)
            pd.DataFrame(
                {
                    "x": [1.0, 20.0, 10.0],
                    "y": [-10.0, 12.0, 1.0],
                    "z": [90.0, 150.0, 120.0],
                }
            ).to_csv(root / "surface2.csv", index=False)

            _, _, _, fragment, _ = self._run_csv_route(root, extent=None)

            expected = np.array(
                [-5.0, 20.0, -10.0, 12.0, 90.0, 150.0],
                dtype=float,
            )
            np.testing.assert_allclose(fragment.call_args.args[1], expected)

    def test_csv_mode_explicit_extent_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv", 5.0)
            explicit = (-2.0, 12.0, -3.0, 13.0, -4.0, 14.0)

            _, _, _, fragment, _ = self._run_csv_route(
                root, extent=explicit
            )

            np.testing.assert_allclose(
                fragment.call_args.args[1],
                np.asarray(explicit, dtype=float),
            )


    def test_csv_inferred_extent_uses_absolute_extent_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "x": [0.0, 5.0, 10.0],
                    "y": [1.0, 5.0, 9.0],
                    "z": [-7.0, 0.0, 10.0],
                }
            ).to_csv(root / "surface1.csv", index=False)

            _, _, _, fragment, _ = self._run_csv_route(
                root,
                extent=None,
                extent_z=[-9.0, 5.0],
            )

            np.testing.assert_allclose(
                fragment.call_args.args[1],
                np.array([0.0, 10.0, 1.0, 9.0, -9.0, 5.0]),
            )

    def test_csv_extent_z_replaces_only_inferred_z_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "x": [-5.0, 7.0, 20.0],
                    "y": [-10.0, 1.0, 12.0],
                    "z": [90.0, 120.0, 150.0],
                }
            ).to_csv(root / "surface1.csv", index=False)

            _, _, _, fragment, _ = self._run_csv_route(
                root,
                extent=None,
                extent_z=[-20.0, 30.0],
            )

            np.testing.assert_allclose(
                fragment.call_args.args[1],
                np.array([-5.0, 20.0, -10.0, 12.0, -20.0, 30.0]),
            )

    def test_csv_explicit_extent_takes_precedence_over_extent_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")
            explicit = (-2.0, 12.0, -3.0, 13.0, -4.0, 14.0)

            _, _, _, fragment, _ = self._run_csv_route(
                root,
                extent=explicit,
                extent_z=[-100.0, 200.0],
            )

            np.testing.assert_allclose(
                fragment.call_args.args[1],
                np.asarray(explicit, dtype=float),
            )

    def test_extent_z_requires_z_min_less_than_z_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")

            with self.assertRaisesRegex(ValueError, "z_min < z_max"):
                create_unstructured_mesh_data(
                    geomodel_result=load_csv_surface_coordinates(root),
                    extent_z=[5.0, -9.0],
                    mapping_litho="none",
                )

    def test_extent_z_requires_two_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")

            with self.assertRaisesRegex(ValueError, "exactly two"):
                create_unstructured_mesh_data(
                    geomodel_result=load_csv_surface_coordinates(root),
                    extent_z=[-9.0],
                    mapping_litho="none",
                )

    def test_csv_model_loader_returns_surface_coordinates(self):
        """The current API accepts one geomodel object, not two input routes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")
            model = load_csv_surface_coordinates(root)
            self.assertEqual(len(model.csv_surface_coordinates), 1)
            csv_path, xyz = model.csv_surface_coordinates[0]
            self.assertEqual(csv_path.name, "surface1.csv")
            self.assertEqual(xyz.shape, (4, 3))

    def test_csv_requires_xyz_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}).to_csv(
                root / "bad.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "must contain x, y, z columns"):
                load_csv_surface_coordinates(root)

    def test_hidden_csv_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")
            _write_surface(root / "._fault_hidden.csv")

            model = load_csv_surface_coordinates(root)
            self.assertEqual([path.name for path, _ in model.csv_surface_coordinates], ["surface1.csv"])
            _, _, create_grid, fragment, _ = self._run_csv_route(root)
            csv_surfaces = create_grid.call_args.args[0]
            self.assertEqual([name for name, _ in csv_surfaces], ["surface1.csv"])
            self.assertEqual(fragment.call_args.args[2], {})


if __name__ == "__main__":
    unittest.main()
