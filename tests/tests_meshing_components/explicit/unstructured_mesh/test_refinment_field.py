import unittest
from unittest.mock import patch

from core.meshing_components.explicit.unstructured.refinement_fields import (
    FieldIDGenerator,
    build_triangulation_mesh_callback,
    build_ellipse_mesh_callback,
    build_combined_mesh_callback,
    build_refinement_fields,
    apply_background_fields,
)


class DummyCfg:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.hmin = 1.0
        self.hmax = 10.0
        self.d1 = 5.0
        self.d2 = 20.0


class TestRefinementFields(unittest.TestCase):

    def setUp(self):
        self.cfg = DummyCfg()

    # --------------------------------------------------
    # FieldIDGenerator
    # --------------------------------------------------

    def test_field_id_generator(self):
        gen = FieldIDGenerator()

        self.assertEqual(gen.next(), 1)
        self.assertEqual(gen.next(), 2)
        self.assertEqual(gen.next(), 3)

    # --------------------------------------------------
    # Triangulation callback
    # --------------------------------------------------

    def test_triangulation_callback_inside_d1(self):
        points = [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]

        cb = build_triangulation_mesh_callback(
            points,
            hmin=1.0,
            hmax=10.0,
            d1=5.0,
            d2=20.0,
        )

        size = cb(0, 0, 0.0, 0.0, 0.0, 50.0)

        self.assertEqual(size, 1.0)

    def test_triangulation_callback_between_d1_d2(self):
        cb = build_triangulation_mesh_callback(
            [[0.0, 0.0, 0.0]],
            hmin=2.0,
            hmax=10.0,
            d1=10.0,
            d2=20.0,
        )

        size = cb(0, 0, 15.0, 0.0, 0.0, 50.0)

        self.assertGreater(size, 2.0)
        self.assertLess(size, 10.0)

    def test_triangulation_callback_outside_d2(self):
        cb = build_triangulation_mesh_callback(
            [[0.0, 0.0, 0.0]],
            hmin=2.0,
            hmax=10.0,
            d1=10.0,
            d2=20.0,
        )

        size = cb(0, 0, 50.0, 0.0, 0.0, 99.0)

        self.assertEqual(size, 99.0)

    # --------------------------------------------------
    # Ellipse callback
    # --------------------------------------------------

    def test_ellipse_callback_none(self):
        self.assertIsNone(
            build_ellipse_mesh_callback([])
        )

    def test_ellipse_callback_inside(self):
        ellipses = [
            {
                "center": [0.0, 0.0, 0.0],
                "radii": [10.0, 5.0],
            }
        ]

        cb = build_ellipse_mesh_callback(
            ellipses,
            hmin=1.0,
            hmax=20.0,
            d1=5.0,
            d2=15.0,
        )

        size = cb(0, 0, 0.0, 0.0, 0.0, 50.0)

        self.assertEqual(size, 1.0)

    # --------------------------------------------------
    # Combined callback
    # --------------------------------------------------

    def test_combined_callback_returns_none_when_disabled(self):
        cb = build_combined_mesh_callback(
            triangulations=None,
            ellipses=None,
            fault_fragments=None,
            tri_cfg=None,
            ell_cfg=None,
            fault_cfg=None,
        )

        self.assertIsNone(cb)

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_triangulation_mesh_callback"
    )
    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_ellipse_mesh_callback"
    )
    def test_combined_callback_takes_minimum(
        self,
        mock_ellipse,
        mock_tri,
    ):
        mock_tri.return_value = lambda *args: 5.0
        mock_ellipse.return_value = lambda *args: 2.0

        cb = build_combined_mesh_callback(
            triangulations=[[0, 0, 0]],
            ellipses=[{"center": [0, 0, 0], "radii": [1, 1]}],
            tri_cfg=self.cfg,
            ell_cfg=self.cfg,
        )

        result = cb(0, 0, 0, 0, 0, 20.0)

        self.assertEqual(result, 2.0)

    # --------------------------------------------------
    # build_refinement_fields
    # --------------------------------------------------

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_well_refinement"
    )
    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_source_refinement"
    )
    def test_build_refinement_fields(
        self,
        mock_source,
        mock_well,
    ):
        mock_well.return_value = 10
        mock_source.return_value = 20

        result = build_refinement_fields(
            zip_info=[],
            well_lines=[],
            source_points=[],
            refinement=None,
        )

        self.assertEqual(result, [10, 20])

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_well_refinement",
        return_value=None,
    )
    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.build_source_refinement",
        return_value=None,
    )
    def test_build_refinement_fields_empty(
        self,
        mock_source,
        mock_well,
    ):
        result = build_refinement_fields(
            zip_info=[],
            well_lines=[],
            source_points=[],
            refinement=None,
        )

        self.assertEqual(result, [])

    # --------------------------------------------------
    # apply_background_fields
    # --------------------------------------------------

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.gmsh"
    )
    def test_apply_background_fields_single(
        self,
        mock_gmsh,
    ):
        apply_background_fields([5])

        mock_gmsh.model.mesh.field.setAsBackgroundMesh.assert_called_once_with(5)

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.gmsh"
    )
    def test_apply_background_fields_multiple(
        self,
        mock_gmsh,
    ):
        apply_background_fields([2, 4])

        mock_gmsh.model.mesh.field.add.assert_called_once_with(
            "Min",
            5,
        )

        mock_gmsh.model.mesh.field.setNumbers.assert_called_once_with(
            5,
            "FieldsList",
            [2, 4],
        )

        mock_gmsh.model.mesh.field.setAsBackgroundMesh.assert_called_once_with(5)

    @patch(
        "core.meshing_components.explicit.unstructured.refinement_fields.gmsh"
    )
    def test_apply_background_fields_empty(
        self,
        mock_gmsh,
    ):
        apply_background_fields([])

        mock_gmsh.model.mesh.field.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
