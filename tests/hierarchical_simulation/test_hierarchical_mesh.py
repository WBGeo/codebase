
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.hierarchical_simulation.generate_mesh.hierarchical_mesh import hierarchical_mesh


def geomodel(extent=(0, 10, 0, 10, 0, 10)):
    grid = SimpleNamespace(extent=extent)
    frame = SimpleNamespace(grid=grid)
    return SimpleNamespace(structural_frame=frame)


def mesh(nodes, point_sets=None):
    return SimpleNamespace(
        nodes=np.asarray(nodes, dtype=float),
        elements=[0],
        point_sets=point_sets or {},
    )


BIG_NODES = [
    [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
    [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
    [2, 2, 2], [8, 2, 2], [2, 8, 2], [8, 8, 2],
    [2, 2, 8], [8, 2, 8], [2, 8, 8], [8, 8, 8],
]

SMALL_NODES = [
    [2, 2, 2], [8, 2, 2], [2, 8, 2], [8, 8, 2],
    [2, 2, 8], [8, 2, 8], [2, 8, 8], [8, 8, 8],
]


SMALL_POINT_SETS = {
    "left": [0, 2, 4, 6],
    "right": [1, 3, 5, 7],
    "front": [0, 1, 4, 5],
    "back": [2, 3, 6, 7],
    "bottom": [0, 1, 2, 3],
    "top": [4, 5, 6, 7],
}


def big_mesh():
    return mesh(BIG_NODES)


def small_mesh(nodes=SMALL_NODES):
    return mesh(nodes, point_sets=SMALL_POINT_SETS)


class TestHierarchicalMeshValidation(unittest.TestCase):

    def test_invalid_mesh_type(self):
        with self.assertRaises(ValueError):
            hierarchical_mesh(geomodel(), mesh_type="bad", extent2=[2, 8, 2, 8, 2, 8])

    def test_extent2_is_required(self):
        with self.assertRaisesRegex(ValueError, "extent2"):
            hierarchical_mesh(geomodel(), mesh_type="Un", mesh_size1=2, extent2=None)

    def test_extent2_must_be_inside_extent1(self):
        with self.assertRaisesRegex(ValueError, "fully inside"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Un",
                mesh_size1=2,
                extent1=[0, 10, 0, 10, 0, 10],
                extent2=[-1, 8, 2, 8, 2, 8],
            )

    def test_unstructured_requires_mesh_size1(self):
        with self.assertRaisesRegex(ValueError, "mesh_size1"):
            hierarchical_mesh(
                geomodel(), mesh_type="Un", extent2=[2, 8, 2, 8, 2, 8]
            )

    def test_unstructured_rejects_structured_controls(self):
        with self.assertRaisesRegex(ValueError, "only valid for structured"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Un",
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
                refinement1=[1, 1, 1],
            )

    def test_structured_requires_controls(self):
        with self.assertRaisesRegex(ValueError, "must be provided"):
            hierarchical_mesh(
                geomodel(), mesh_type="Str", extent2=[2, 8, 2, 8, 2, 8]
            )

    def test_structured_rejects_mesh_sizes(self):
        with self.assertRaisesRegex(ValueError, "only valid for unstructured"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Str",
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
                refinement1=[1, 1, 1],
                mesh_devision1=(4, 4),
            )

    def test_implicit_rejects_explicit_controls(self):
        with self.assertRaisesRegex(ValueError, "do not accept"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Im",
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
            )

    def test_requires_geomodel_or_surface_csv_dir(self):
        with self.assertRaisesRegex(ValueError, "geomodel_result or surface_csv_dir"):
            hierarchical_mesh(
                mesh_type="Un",
                extent1=[0, 10, 0, 10, 0, 10],
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
            )

    def test_csv_only_requires_explicit_outer_extent(self):
        with self.assertRaisesRegex(ValueError, "explicit extent"):
            hierarchical_mesh(
                mesh_type="Un",
                surface_csv_dir="surfaces",
                extent1="whole",
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
            )

    def test_csv_surfaces_are_rejected_for_structured_mesh(self):
        with self.assertRaisesRegex(ValueError, "only supported for unstructured"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Str",
                surface_csv_dir="surfaces",
                extent2=[2, 8, 2, 8, 2, 8],
                refinement1=[1, 1, 1],
                mesh_devision1=(4, 4),
            )

    def test_csv_surfaces_are_rejected_for_implicit_mesh(self):
        with self.assertRaisesRegex(ValueError, "only supported for unstructured"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Im",
                surface_csv_dir="surfaces",
                extent2=[2, 8, 2, 8, 2, 8],
            )


class TestHierarchicalUnstructured(unittest.TestCase):

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_hierarchical_objects_are_selected_per_mesh(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        wells = {
            1: [(1, 1, 1, 4, 4, 4)],
            2: [(3, 3, 3, 7, 7, 7)],
        }
        sources = {
            1: [(1, 1, 1)],
            2: [(5, 5, 5)],
        }

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent1="whole",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=3,
            mesh_size2=1,
            wells=wells,
            sources=sources,
        )

        self.assertEqual(create.call_count, 2)
        first = create.call_args_list[0].kwargs
        second = create.call_args_list[1].kwargs

        self.assertEqual(first["mesh_size"], 3)
        self.assertEqual(second["mesh_size"], 1)
        self.assertEqual(first["sources"], [(1, 1, 1)])
        self.assertEqual(second["sources"], [(5, 5, 5)])
        self.assertEqual(first["wells"], wells[1])
        self.assertEqual(second["wells"], wells[2])

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_nonhierarchical_objects_are_used_for_both_and_clipped(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            sources=[(1, 1, 1), (5, 5, 5), (9, 9, 9)],
            wells=[(0, 5, 5, 10, 5, 5)],
        )

        outer = create.call_args_list[0].kwargs
        inner = create.call_args_list[1].kwargs

        self.assertEqual(outer["sources"], [(1, 1, 1), (5, 5, 5), (9, 9, 9)])
        self.assertEqual(inner["sources"], [(5, 5, 5)])

        np.testing.assert_allclose(
            np.asarray(inner["wells"][0]).reshape(-1, 3),
            [[2, 5, 5], [8, 5, 5]],
        )

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_all_engineering_object_types_are_unwrapped(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        tri1 = np.array([[1, 1, 1, 11], [2, 2, 2, 11], [3, 3, 3, 11]], dtype=float)
        tri2 = np.array([[3, 3, 3, 22], [4, 4, 4, 22], [5, 5, 5, 22]], dtype=float)

        kwargs = dict(
            wells={1: [(1,1,1, 2,2,2)], 2: [(3,3,3, 4,4,4)]},
            sources={1: [(1,1,1)], 2: [(4,4,4)]},
            shafts={
                1: [(1,1,1, 1,0,0, 0.2)],
                2: [(3,3,3, 1,0,0, 0.2)],
            },
            extra_planes={
                1: [(1,1,1, 2,1,1, 2,2,1, 1,2,1)],
                2: [(3,3,3, 4,3,3, 4,4,3, 3,4,3)],
            },
            ellipses={
                1: [{"center": (2,2,2), "radii": (0.2,0.2), "normal": (0,0,1)}],
                2: [{"center": (4,4,4), "radii": (0.2,0.2), "normal": (0,0,1)}],
            },
            triangulations={1: tri1, 2: tri2},
        )

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            **kwargs,
        )

        outer = create.call_args_list[0].kwargs
        inner = create.call_args_list[1].kwargs

        self.assertEqual(outer["sources"], kwargs["sources"][1])
        self.assertEqual(inner["sources"], kwargs["sources"][2])
        self.assertEqual(inner["shafts"][0][-1], 0.2)
        self.assertEqual(len(inner["extra_planes"]), 1)
        self.assertEqual(len(inner["ellipses"]), 1)
        np.testing.assert_array_equal(inner["triangulations"], tri2)

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_mesh_size2_defaults_to_mesh_size1(self, create):
        create.side_effect = [big_mesh(), small_mesh()]
        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=7,
        )
        self.assertEqual(create.call_args_list[0].kwargs["mesh_size"], 7)
        self.assertEqual(create.call_args_list[1].kwargs["mesh_size"], 7)


    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_common_unstructured_params_are_applied_to_both_meshes(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            unstructured_params={"curve_mesh_size": 5, "gmsh_flag": True},
        )

        first = create.call_args_list[0].kwargs
        second = create.call_args_list[1].kwargs
        self.assertEqual(first["curve_mesh_size"], 5)
        self.assertEqual(second["curve_mesh_size"], 5)
        self.assertTrue(first["gmsh_flag"])
        self.assertTrue(second["gmsh_flag"])

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_csv_only_input_is_forwarded_to_both_meshes(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            mesh_type="Un",
            surface_csv_dir="surfaces",
            extent1=[0, 10, 0, 10, 0, 10],
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=3,
            mesh_size2=1,
        )

        self.assertEqual(create.call_count, 2)
        for call in create.call_args_list:
            self.assertIsNone(call.args[0])
            self.assertEqual(call.kwargs["surface_csv_dir"], "surfaces")

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_geomodel_workflow_does_not_inject_surface_csv_dir(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        model = geomodel()
        hierarchical_mesh(
            model,
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
        )

        self.assertIs(create.call_args_list[0].args[0], model)
        self.assertIs(create.call_args_list[1].args[0], model)
        self.assertNotIn("surface_csv_dir", create.call_args_list[0].kwargs)
        self.assertNotIn("surface_csv_dir", create.call_args_list[1].kwargs)

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_duplicate_surface_csv_dir_is_rejected(self, create):
        with self.assertRaisesRegex(ValueError, "both to hierarchical_mesh"):
            hierarchical_mesh(
                mesh_type="Un",
                surface_csv_dir="surfaces",
                extent1=[0, 10, 0, 10, 0, 10],
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
                unstructured_params={"surface_csv_dir": "other_surfaces"},
            )
        create.assert_not_called()

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_csv_input_keeps_level_specific_unstructured_params(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            mesh_type="Un",
            surface_csv_dir="surfaces",
            extent1=[0, 10, 0, 10, 0, 10],
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            unstructured_params={"curve_mesh_size": 5},
            unstructured_params1={"rotation_angle": 20},
            unstructured_params2={"rotation_angle": 45},
        )

        first = create.call_args_list[0].kwargs
        second = create.call_args_list[1].kwargs
        self.assertEqual(first["surface_csv_dir"], "surfaces")
        self.assertEqual(second["surface_csv_dir"], "surfaces")
        self.assertEqual(first["curve_mesh_size"], 5)
        self.assertEqual(second["curve_mesh_size"], 5)
        self.assertEqual(first["rotation_angle"], 20)
        self.assertEqual(second["rotation_angle"], 45)

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_level_specific_unstructured_params_override_common_values(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            unstructured_params={"curve_mesh_size": 5, "rotation_angle": 10},
            unstructured_params1={"rotation_angle": 20},
            unstructured_params2={"rotation_angle": 45},
        )

        first = create.call_args_list[0].kwargs
        second = create.call_args_list[1].kwargs
        self.assertEqual(first["curve_mesh_size"], 5)
        self.assertEqual(second["curve_mesh_size"], 5)
        self.assertEqual(first["rotation_angle"], 20)
        self.assertEqual(second["rotation_angle"], 45)


class TestStructuredAndImplicit(unittest.TestCase):

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_structured_mesh_data")
    def test_structured_defaults_second_controls_to_first(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Str",
            extent2=[2, 8, 2, 8, 2, 8],
            refinement1=[2, 2, 2],
            mesh_devision1=(4, 4),
        )

        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[1].kwargs["refinement_data"], [2, 2, 2])
        self.assertEqual(create.call_args_list[1].kwargs["mesh_devision"], (4, 4))

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_implicit_structured_mesh")
    def test_implicit_creates_two_extents(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Im",
            extent1="whole",
            extent2=[2, 8, 2, 8, 2, 8],
            implicit_params={"example": 1},
        )

        self.assertEqual(create.call_count, 2)
        np.testing.assert_array_equal(
            create.call_args_list[0].kwargs["extent"],
            [0, 10, 0, 10, 0, 10],
        )
        np.testing.assert_array_equal(
            create.call_args_list[1].kwargs["extent"],
            [2, 8, 2, 8, 2, 8],
        )
        self.assertEqual(create.call_args_list[0].kwargs["example"], 1)


class TestBoundaryMatching(unittest.TestCase):

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_returns_six_boundary_faces_with_big_mesh_node_ids(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        _, _, boundary = hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
        )

        self.assertEqual(
            set(boundary),
            {"left", "right", "front", "back", "bottom", "top"},
        )
        for values in boundary.values():
            self.assertEqual(values.shape[1], 4)
            self.assertTrue(np.all(values[:, 0] >= 0))
            self.assertTrue(np.all(values[:, 0] < len(BIG_NODES)))

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_boundary_nodes_are_taken_from_mesh2_point_sets(self, create):
        # Add an interior node whose coordinates would confuse a min/max-based
        # boundary detector. Only the explicitly listed point-set nodes should
        # participate in boundary matching.
        small_nodes = np.vstack([np.asarray(SMALL_NODES, dtype=float), [2, 5, 5]])
        create.side_effect = [big_mesh(), small_mesh(small_nodes)]

        _, _, boundary = hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
        )

        self.assertEqual(len(boundary["left"]), len(SMALL_POINT_SETS["left"]))
        self.assertEqual(len(boundary["right"]), len(SMALL_POINT_SETS["right"]))

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_missing_point_sets_raise_clear_error(self, create):
        create.side_effect = [big_mesh(), mesh(SMALL_NODES)]

        with self.assertRaisesRegex(ValueError, "point_sets"):
            hierarchical_mesh(
                geomodel(),
                mesh_type="Un",
                extent2=[2, 8, 2, 8, 2, 8],
                mesh_size1=2,
            )

    @patch("core.hierarchical_simulation.generate_mesh.hierarchical_mesh.create_unstructured_mesh_data")
    def test_different_rotation_angles_are_accepted_for_each_level(self, create):
        create.side_effect = [big_mesh(), small_mesh()]

        hierarchical_mesh(
            geomodel(),
            mesh_type="Un",
            extent2=[2, 8, 2, 8, 2, 8],
            mesh_size1=2,
            unstructured_params1={"rotation_angle": 20},
            unstructured_params2={"rotation_angle": 45},
        )

        self.assertEqual(create.call_args_list[0].kwargs["rotation_angle"], 20)
        self.assertEqual(create.call_args_list[1].kwargs["rotation_angle"], 45)


if __name__ == "__main__":
    unittest.main()
