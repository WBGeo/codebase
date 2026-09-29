import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import meshio
import numpy as np
import pandas as pd

from core.meshing_components.explicit.unstructured.mesh_data import (
    LithoMappingMode,
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

    def _run_csv_route(self, csv_dir, mapping_litho="none", extent=EXTENT, extent_z=None):
        cube_nodes = np.array(
            [
                [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
                [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
            ],
            dtype=float,
        )
        tetra = meshio.CellBlock("tetra", np.array([[0, 1, 2, 4]], dtype=int))

        with (
            patch(f"{MODULE}.data_preparation") as data_preparation,
            patch(f"{MODULE}.create_surface_grid", return_value=[np.zeros((4, 3))]) as create_grid,
            patch(f"{MODULE}.import_surfaces", return_value=([101, 102], EXTENT)),
            patch(
                f"{MODULE}.fragment_surfaces",
                return_value=([(3, 1)], [], [], [], [], [], []),
            ) as fragment,
            patch(
                f"{MODULE}.mesh_generator",
                return_value=(cube_nodes, [tetra], {}),
            ) as mesh_generator,
            patch(f"{MODULE}.gmsh.initialize"),
            patch(f"{MODULE}.gmsh.finalize"),
            patch(f"{MODULE}.gmsh.model.occ.synchronize"),
        ):
            kwargs = {
                "surface_csv_dir": str(csv_dir),
                "mapping_litho": mapping_litho,
            }
            if extent is not None:
                kwargs["extent"] = extent
            if extent_z is not None:
                kwargs["extent_z"] = extent_z
            result = create_unstructured_mesh_data(**kwargs)

        return result, data_preparation, create_grid, fragment, mesh_generator

    def test_csv_mode_bypasses_structural_model_cleaning(self):
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
            cleaned_surfaces = create_grid.call_args.args[0]

            self.assertEqual(len(cleaned_surfaces), 2)
            for item in cleaned_surfaces:
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
            cleaned_surfaces = create_grid.call_args.args[0]
            index_by_name = {name: i for i, (name, _) in enumerate(cleaned_surfaces)}

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
                    "x": [1.0, 20.0],
                    "y": [-10.0, 12.0],
                    "z": [90.0, 150.0],
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
                    surface_csv_dir=str(root),
                    extent_z=[5.0, -9.0],
                    mapping_litho="none",
                )

    def test_extent_z_requires_two_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")

            with self.assertRaisesRegex(ValueError, "exactly two"):
                create_unstructured_mesh_data(
                    surface_csv_dir=str(root),
                    extent_z=[-9.0],
                    mapping_litho="none",
                )

    def test_geomodel_and_csv_directory_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")
            with self.assertRaisesRegex(ValueError, "either geomodel_result or surface_csv_dir"):
                create_unstructured_mesh_data(
                    geomodel_result=object(),
                    surface_csv_dir=str(root),
                    extent=EXTENT,
                    mapping_litho="none",
                )

    def test_csv_requires_xyz_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}).to_csv(
                root / "bad.csv", index=False
            )
            with (
                patch(f"{MODULE}.gmsh.initialize"),
                patch(f"{MODULE}.gmsh.finalize"),
            ):
                with self.assertRaisesRegex(ValueError, "must contain x, y, z columns"):
                    create_unstructured_mesh_data(
                        surface_csv_dir=str(root),
                        extent=EXTENT,
                        mapping_litho="none",
                    )

    def test_hidden_csv_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_surface(root / "surface1.csv")
            _write_surface(root / "._fault_hidden.csv")

            _, _, create_grid, fragment, _ = self._run_csv_route(root)
            cleaned_surfaces = create_grid.call_args.args[0]
            self.assertEqual([name for name, _ in cleaned_surfaces], ["surface1.csv"])
            self.assertEqual(fragment.call_args.args[2], {})


if __name__ == "__main__":
    unittest.main()
