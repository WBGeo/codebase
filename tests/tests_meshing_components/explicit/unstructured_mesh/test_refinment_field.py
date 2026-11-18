import unittest

import numpy as np
from scipy.spatial import cKDTree
from unittest.mock import patch

from core.meshing_components.explicit.unstructured.refinement_fields import (
    FieldIDGenerator,
    build_triangulation_mesh_callback,
    build_ellipse_mesh_callback,
    build_extra_plane_mesh_callback,
    build_combined_mesh_callback,
    build_refinement_fields,
    build_shaft_refinement,
    build_general_refinement,
    apply_background_fields,
)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement,
    LinearShaftRefinement,
    FunctionShaftRefinement,
    GeneralRefinement,
    TriangulationRefinement,
    EllipseRefinement,
    FaultRefinement,
    ExtraPlaneRefinement,
)


class TestRefinementFields(unittest.TestCase):

    # --------------------------------------------------
    # FieldIDGenerator
    # --------------------------------------------------
    def test_field_id_generator(self):
        gen = FieldIDGenerator()
        self.assertEqual(gen.next(), 1)
        self.assertEqual(gen.next(), 2)
        self.assertEqual(gen.next(), 3)

    # --------------------------------------------------
    # Triangulation callback -- current parameter names
    # --------------------------------------------------
    def test_triangulation_callback_inside_distmin(self):
        cb = build_triangulation_mesh_callback(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            SizeMin=1.0,
            SizeMax=10.0,
            DistMin=5.0,
            DistMax=20.0,
        )
        self.assertEqual(cb(0, 0, 0.0, 0.0, 0.0, 50.0), 1.0)

    def test_triangulation_callback_between_distmin_distmax(self):
        cb = build_triangulation_mesh_callback(
            [[0.0, 0.0, 0.0]],
            SizeMin=2.0,
            SizeMax=10.0,
            DistMin=10.0,
            DistMax=20.0,
        )
        size = cb(0, 0, 15.0, 0.0, 0.0, 50.0)
        self.assertGreater(size, 2.0)
        self.assertLess(size, 10.0)

    def test_triangulation_callback_outside_distmax_returns_sizemax(self):
        cb = build_triangulation_mesh_callback(
            [[0.0, 0.0, 0.0]],
            SizeMin=2.0,
            SizeMax=10.0,
            DistMin=10.0,
            DistMax=20.0,
        )
        self.assertEqual(cb(0, 0, 50.0, 0.0, 0.0, 99.0), 10.0)

    # --------------------------------------------------
    # Ellipse callback
    # --------------------------------------------------
    def test_ellipse_callback_none(self):
        self.assertIsNone(build_ellipse_mesh_callback([]))

    def test_ellipse_callback_inside(self):
        ellipses = [{"center": [0.0, 0.0, 0.0], "radii": [10.0, 5.0]}]
        cb = build_ellipse_mesh_callback(
            ellipses,
            SizeMin=1.0,
            SizeMax=20.0,
            DistMin=5.0,
            DistMax=15.0,
        )
        self.assertEqual(cb(0, 0, 0.0, 0.0, 0.0, 50.0), 1.0)

    # --------------------------------------------------
    # Extra-plane callback -- recently added refinement
    # --------------------------------------------------
    def test_extra_plane_callback_none(self):
        self.assertIsNone(build_extra_plane_mesh_callback([]))

    def test_extra_plane_callback_on_plane(self):
        plane = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ]
        cb = build_extra_plane_mesh_callback(
            [plane],
            SizeMin=1.0,
            SizeMax=10.0,
            DistMin=0.5,
            DistMax=2.0,
        )
        self.assertEqual(cb(0, 0, 0.5, 0.5, 0.0, 20.0), 1.0)

    def test_extra_plane_callback_exact_finite_triangle_distance(self):
        """Distance is to the finite surface, not its vertices or plane extension."""
        square = [0., 0., 0., 2., 0., 0., 2., 2., 0., 0., 2., 0.]
        cb = build_extra_plane_mesh_callback(
            [square], SizeMin=2., SizeMax=10., DistMin=1., DistMax=3.
        )
        # Directly above the face: distance 0.5, within DistMin.
        self.assertAlmostEqual(cb(0, 0, 1., 1., .5, 100.), 2.)
        # Above the face: distance 2, halfway through transition.
        self.assertAlmostEqual(cb(0, 0, 1., 1., 2., 100.), 6.)
        # Beyond the edge: distance 2 to the nearest finite edge.
        self.assertAlmostEqual(cb(0, 0, 4., 1., 0., 100.), 6.)
        self.assertAlmostEqual(cb(0, 0, 20., 1., 0., 100.), 10.)

    def test_extra_plane_spatial_index_preserves_nearest_large_triangle(self):
        """The nearest centroid need not belong to the nearest triangle."""
        large = [0., 0., 0., 100., 0., 0., 100., 100., 0., 0., 100., 0.]
        small = [2., 2., 3., 3., 2., 3., 3., 3., 3., 2., 3., 3.]
        cb = build_extra_plane_mesh_callback(
            [large, small], SizeMin=1., SizeMax=9., DistMin=0., DistMax=4.
        )
        # The large plane is nearest even though the small plane's
        # centroid is much closer to the query point.
        self.assertAlmostEqual(cb(0, 0, 2.5, 2.5, 0., 100.), 1.)
        self.assertAlmostEqual(cb(0, 0, 2.5, 2.5, 1., 100.), 3.)

    def test_extra_plane_spatial_index_matches_brute_force_on_parallel_planes(self):
        """For points inside aligned squares, exact distance is the minimum |dz|."""
        heights = [-20., -5., 7., 25.]
        planes = [
            [-10., -10., h, 10., -10., h, 10., 10., h, -10., 10., h]
            for h in heights
        ]
        cb = build_extra_plane_mesh_callback(
            planes, SizeMin=2., SizeMax=14., DistMin=1., DistMax=7.
        )
        rng = np.random.default_rng(42)
        for x, y, z in rng.uniform([-9., -9., -30.], [9., 9., 35.], (200, 3)):
            distance = min(abs(z - h) for h in heights)
            expected = (
                2. if distance <= 1. else
                14. if distance >= 7. else
                2. + (distance - 1.) / 6. * 12.
            )
            self.assertAlmostEqual(cb(0, 0, x, y, z, 100.), expected)

    def test_extra_plane_spatial_index_constructed_once(self):
        """The KD-tree is built at callback creation, not on every Gmsh query."""
        from unittest.mock import patch as mock_patch
        module = "core.meshing_components.explicit.unstructured.refinement_fields"
        square = [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.]
        with mock_patch(f"{module}.cKDTree", wraps=cKDTree) as kd_tree:
            cb = build_extra_plane_mesh_callback([square])
            self.assertEqual(kd_tree.call_count, 1)
            for z in (0., 1., 10.):
                cb(0, 0, .5, .5, z, 20.)
            self.assertEqual(kd_tree.call_count, 1)

    def test_extra_plane_skips_invalid_coordinate_count(self):
        self.assertIsNone(build_extra_plane_mesh_callback([[0., 1., 2.]]))

    def test_refinement_accepts_extra_planes_snake_case(self):
        cfg = ExtraPlaneRefinement(SizeMin=2., SizeMax=10.)
        refinement = Refinement(extra_planes=cfg)
        self.assertIsNotNone(refinement.extra_planes)
        self.assertEqual(refinement.extra_planes.SizeMin, 2.)

    def test_refinement_accepts_legacy_extraplanes_alias(self):
        cfg = ExtraPlaneRefinement(SizeMin=2., SizeMax=10.)
        refinement = Refinement(extraplanes=cfg)
        self.assertIsNotNone(refinement.extra_planes)
        self.assertEqual(refinement.extraplanes.SizeMin, 2.)

    # --------------------------------------------------
    # Combined callback -- triangulation / ellipse /
    # fault / extra-plane all participate via minimum
    # --------------------------------------------------
    def test_combined_callback_returns_none_when_disabled(self):
        cb = build_combined_mesh_callback()
        self.assertIsNone(cb)

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_extra_plane_mesh_callback")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_fault_mesh_callback_from_fragments")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_ellipse_mesh_callback")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_triangulation_mesh_callback")
    def test_combined_callback_takes_minimum_across_all_refinements(
        self, mock_tri, mock_ellipse, mock_fault, mock_extra
    ):
        mock_tri.return_value = lambda *args: 5.0
        mock_ellipse.return_value = lambda *args: 4.0
        mock_fault.return_value = lambda *args: 3.0
        mock_extra.return_value = lambda *args: 2.0

        cb = build_combined_mesh_callback(
            triangulations=[[0, 0, 0]],
            ellipses=[{"center": [0, 0, 0], "radii": [1, 1]}],
            fault_fragments=[(2, 1)],
            extra_planes=[[0] * 12],
            tri_cfg=TriangulationRefinement(),
            ell_cfg=EllipseRefinement(),
            fault_cfg=FaultRefinement(),
            extra_plane_cfg=ExtraPlaneRefinement(),
        )
        self.assertEqual(cb(0, 0, 0, 0, 0, 20.0), 2.0)

    # --------------------------------------------------
    # Shaft refinement -- recently added refinement
    # --------------------------------------------------
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_linear_shaft_refinement_builds_threshold_field(self, mock_gmsh):
        refinement = Refinement(
            shafts=LinearShaftRefinement(
                SizeMin=1.0, SizeMax=10.0, DistMin=5.0, DistMax=20.0
            )
        )
        shaft = {"center": [0, 0, 0], "axis": [0, 0, 1]}
        field_id = build_shaft_refinement([shaft], refinement, FieldIDGenerator())
        self.assertEqual(field_id, 2)
        mock_gmsh.model.mesh.field.add.assert_any_call("MathEval", 1)
        mock_gmsh.model.mesh.field.add.assert_any_call("Threshold", 2)
        mock_gmsh.model.mesh.field.setNumber.assert_any_call(2, "SizeMin", 1.0)
        mock_gmsh.model.mesh.field.setNumber.assert_any_call(2, "SizeMax", 10.0)

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_function_shaft_refinement_builds_math_eval(self, mock_gmsh):
        refinement = Refinement(
            shafts=FunctionShaftRefinement(expression="1.0 + DIST")
        )
        shaft = {"center": [0, 0, 0], "axis": [0, 0, 1]}
        field_id = build_shaft_refinement([shaft], refinement, FieldIDGenerator())
        self.assertEqual(field_id, 1)
        mock_gmsh.model.mesh.field.add.assert_called_with("MathEval", 1)
        expr = mock_gmsh.model.mesh.field.setString.call_args.args[2]
        self.assertIn("sqrt", expr)
        self.assertNotIn("DIST", expr)

    # --------------------------------------------------
    # General point / line refinement -- recently added
    # --------------------------------------------------
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_general_point_linear_refinement(self, mock_gmsh):
        mock_gmsh.model.occ.addPoint.return_value = 11
        refinement = Refinement(
            general=GeneralRefinement(
                coordinates=[(1.0, 2.0, 3.0)],
                SizeMin=2.0,
                DistMax=25.0,
            )
        )
        fields = build_general_refinement(refinement, FieldIDGenerator(), mesh_size=30.0)
        self.assertEqual(fields, [2])
        mock_gmsh.model.mesh.field.setNumbers.assert_any_call(1, "NodesList", [11])
        mock_gmsh.model.mesh.field.setNumber.assert_any_call(2, "SizeMax", 30.0)

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_general_line_function_refinement(self, mock_gmsh):
        mock_gmsh.model.occ.addPoint.side_effect = [11, 12]
        mock_gmsh.model.occ.addLine.return_value = 21
        refinement = Refinement(
            general=GeneralRefinement(
                coordinates=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                expression="2.0 + 0.5*DIST",
            )
        )
        fields = build_general_refinement(refinement, FieldIDGenerator(), mesh_size=30.0)
        self.assertEqual(fields, [2])
        mock_gmsh.model.mesh.field.setNumbers.assert_any_call(1, "CurvesList", [21])
        mock_gmsh.model.mesh.field.setString.assert_called_with(2, "F", "2.0 + 0.5*F1")

    def test_general_refinement_rejects_mixed_linear_and_expression(self):
        refinement = Refinement(
            general=GeneralRefinement(
                coordinates=[(0.0, 0.0, 0.0)],
                SizeMin=1.0,
                DistMax=10.0,
                expression="DIST",
            )
        )
        with self.assertRaisesRegex(ValueError, "either 'expression'"):
            build_general_refinement(refinement, FieldIDGenerator(), mesh_size=20.0)

    # --------------------------------------------------
    # build_refinement_fields -- now includes shaft and
    # general fields in addition to wells and sources
    # --------------------------------------------------
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_general_refinement")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_shaft_refinement")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_source_refinement")
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_well_refinement")
    def test_build_refinement_fields_includes_all_active_field_types(
        self, mock_well, mock_source, mock_shaft, mock_general
    ):
        mock_well.return_value = 10
        mock_source.return_value = 20
        mock_shaft.return_value = 30
        mock_general.return_value = [40, 50]

        result = build_refinement_fields(
            zip_info=[],
            well_lines=[],
            source_points=[],
            mine_shafts=[],
            triangulations=[],
            refinement=None,
            mesh_size=25.0,
        )
        self.assertEqual(result, [10, 20, 30, 40, 50])

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_general_refinement", return_value=[])
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_shaft_refinement", return_value=None)
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_source_refinement", return_value=None)
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.build_well_refinement", return_value=None)
    def test_build_refinement_fields_empty(self, mock_well, mock_source, mock_shaft, mock_general):
        result = build_refinement_fields(
            zip_info=[],
            well_lines=[],
            source_points=[],
            mine_shafts=[],
            triangulations=[],
            refinement=None,
        )
        self.assertEqual(result, [])

    # --------------------------------------------------
    # apply_background_fields
    # --------------------------------------------------
    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_apply_background_fields_single(self, mock_gmsh):
        apply_background_fields([5])
        mock_gmsh.model.mesh.field.setAsBackgroundMesh.assert_called_once_with(5)

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_apply_background_fields_multiple(self, mock_gmsh):
        apply_background_fields([2, 4])
        mock_gmsh.model.mesh.field.add.assert_called_once_with("Min", 5)
        mock_gmsh.model.mesh.field.setNumbers.assert_called_once_with(5, "FieldsList", [2, 4])
        mock_gmsh.model.mesh.field.setAsBackgroundMesh.assert_called_once_with(5)

    @patch("core.meshing_components.explicit.unstructured.refinement_fields.gmsh")
    def test_apply_background_fields_empty(self, mock_gmsh):
        apply_background_fields([])
        mock_gmsh.model.mesh.field.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestSourcePointSize(unittest.TestCase):

    def test_linear_refinement_uses_size_min(self):
        from core.meshing_components.explicit.unstructured.refinement_fields import source_point_size
        from core.meshing_components.explicit.unstructured.refinement_mesh import LinearSourceRefinement
        refinement = Refinement(sources=LinearSourceRefinement(SizeMin=4.0, SizeMax=50.0, DistMin=10.0, DistMax=80.0))
        self.assertEqual(source_point_size(refinement), 4.0)

    def test_function_refinement_is_evaluated_at_distance_zero(self):
        from core.meshing_components.explicit.unstructured.refinement_fields import source_point_size
        from core.meshing_components.explicit.unstructured.refinement_mesh import FunctionSourceRefinement
        refinement = Refinement(sources=FunctionSourceRefinement(expression="5 + 75*(1 - exp(-DIST/80))"))
        self.assertAlmostEqual(source_point_size(refinement), 5.0)

    def test_no_source_refinement_or_unknown_function_gives_none(self):
        from core.meshing_components.explicit.unstructured.refinement_fields import source_point_size
        from core.meshing_components.explicit.unstructured.refinement_mesh import FunctionSourceRefinement
        self.assertIsNone(source_point_size(None))
        self.assertIsNone(source_point_size(Refinement()))
        self.assertIsNone(source_point_size(Refinement(sources=FunctionSourceRefinement(expression="Step(DIST)"))))


class TestFaultRefinementUsesAllFaults(unittest.TestCase):

    def setUp(self):
        import gmsh
        gmsh.finalize() if gmsh.isInitialized() else None
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

    def tearDown(self):
        import gmsh
        gmsh.finalize()

    def test_fragments_of_every_fault_reach_the_refinement(self):
        import gmsh
        import core.meshing_components.explicit.unstructured.create_grid_fragment_surface as cgfs

        def vertical_plane(x):
            occ = gmsh.model.occ
            pts = [occ.addPoint(x, y, z) for y, z in ((-1, -1), (11, -1), (11, 11), (-1, 11))]
            loop = occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)])
            return occ.addPlaneSurface([loop])

        surfaces = [vertical_plane(3.0), vertical_plane(7.0)]
        gmsh.model.occ.synchronize()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return None

        with patch.object(cgfs, "build_combined_mesh_callback", side_effect=capture):
            cgfs.fragment_surfaces(surfaces, [0, 10, 0, 10, 0, 10], {"F1": 0, "F2": 1}, [], [], [], [],
                                   mesh_size=5.0, curve_mesh_size=5.0,
                                   refinement=Refinement(faults=FaultRefinement(SizeMin=1.0, SizeMax=5.0)))

        fault_x = sorted({round(gmsh.model.getBoundingBox(2, tag)[0], 6) for dim, tag in captured["fault_fragments"]})
        self.assertEqual(fault_x, [3.0, 7.0])
