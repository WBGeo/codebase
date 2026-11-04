import unittest
from types import SimpleNamespace

import gmsh
import numpy as np

from core.meshing_components.explicit.unstructured import structural_surfaces as ss


def grid_mesh(nx, ny, height=lambda x, y: 0.0 * x, skip=()):
    """Triangulated nx x ny grid of unit cells in the xy-plane, z = height(x, y); cells in `skip` left out."""
    xs, ys = np.meshgrid(np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float), indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), height(xs, ys).ravel()])
    idx = lambda i, j: i * (ny + 1) + j
    faces = []
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return vertices, np.asarray(faces, dtype=np.int64)


def fake_result(elements, faults=(), spacing=(1.0, 1.0, 1.0)):
    """Minimal structural result: elements/faults as (name, vertices, faces)."""
    def obj(name, v, f):
        return SimpleNamespace(name=name, vertices={"extended_masked": v}, edges={"extended_masked": f})
    frame = SimpleNamespace(
        grid=SimpleNamespace(spacing=spacing),
        structural_groups=[SimpleNamespace(structural_elements=[obj(*e) for e in elements])],
        fault_frame=SimpleNamespace(fault_elements=[obj(*f) for f in faults]) if faults else None,
    )
    return SimpleNamespace(structural_frame=frame)


class SplitConnectedPiecesTestCase(unittest.TestCase):

    def test_two_separate_patches_give_two_pieces(self):
        v1, f1 = grid_mesh(2, 2)
        v2, f2 = grid_mesh(3, 1)
        vertices = np.vstack([v1, v2 + [10.0, 0.0, 0.0]])
        faces = np.vstack([f1, f2 + len(v1)])

        pieces = ss.split_connected_pieces(vertices, faces)

        self.assertEqual(sorted(len(f) for _, f in pieces), [6, 8])
        for v, f in pieces:
            self.assertEqual(f.max(), len(v) - 1)  # re-indexed, no unused vertices

    def test_empty_mesh_gives_no_pieces(self):
        self.assertEqual(ss.split_connected_pieces(np.empty((0, 3)), np.empty((0, 3), dtype=int)), [])


class BoundaryLoopsTestCase(unittest.TestCase):

    def test_square_with_hole_has_outer_loop_and_hole(self):
        vertices, faces = grid_mesh(3, 3, skip={(1, 1)})

        loops = ss._boundary_loops(faces, vertices[:, :2])

        self.assertEqual(sorted(len(loop) for loop in loops), [4, 12])

    def test_squares_touching_in_one_vertex_give_two_simple_loops(self):
        vertices, faces = grid_mesh(2, 2, skip={(0, 1), (1, 0)})

        loops = ss._boundary_loops(faces, vertices[:, :2])

        self.assertEqual(sorted(len(loop) for loop in loops), [4, 4])

    def test_mirrored_frame_gives_the_same_loops(self):
        vertices, faces = grid_mesh(2, 2, skip={(0, 1), (1, 0)})

        loops = ss._boundary_loops(faces, vertices[:, :2] * [1.0, -1.0])

        self.assertEqual(sorted(len(loop) for loop in loops), [4, 4])


class SimplifyClosedTestCase(unittest.TestCase):

    def test_staircase_within_tolerance_becomes_a_rectangle(self):
        bottom = np.column_stack([np.arange(11.0), 0.05 * (np.arange(11) % 2)])
        top = np.column_stack([np.arange(10.0, -1.0, -1.0), np.full(11, 5.0)])
        polygon = np.vstack([bottom, top])

        simplified = ss._simplify_closed(polygon, tolerance=0.1)

        self.assertEqual(len(simplified), 4)

    def test_keeps_corners_beyond_tolerance(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]])

        self.assertEqual(len(ss._simplify_closed(square, tolerance=0.1)), 5)


class ImportStructuralSurfacesTestCase(unittest.TestCase):

    def setUp(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

    def tearDown(self):
        gmsh.finalize()

    def test_flat_curved_and_fault_pieces(self):
        flat = grid_mesh(8, 6, height=lambda x, y: 0.0 * x + 2.0)
        curved = grid_mesh(12, 8, height=lambda x, y: 5.0 + 2.0 * np.sin(x / 3.0))
        fault_v, fault_f = grid_mesh(6, 6)
        fault_v = fault_v[:, [2, 0, 1]] + [4.0, 0.0, 0.0]  # vertical plane x = 4
        result = fake_result([("flat", *flat), ("curved", *curved)], faults=[("F1", fault_v, fault_f)])

        surfaces, refs = ss.import_structural_surfaces(result)

        self.assertEqual(len(surfaces), 3)
        # the fault is the only reference surface, at its position in the surface list
        self.assertEqual(list(refs.values()), [2])
        types = [gmsh.model.getType(2, tag) for tag in surfaces]
        self.assertEqual(types[0], "Plane")
        self.assertEqual(types[1], "BSpline surface")
        # each surface covers the area of its piece (trimmed to the outline, not the bounding box)
        for tag, (v, f) in zip(surfaces, [flat, curved, (fault_v, fault_f)]):
            tri = v[f]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
            self.assertAlmostEqual(gmsh.model.occ.getMass(2, tag) / area, 1.0, delta=0.02)

    def test_non_convex_piece_is_trimmed_to_its_outline(self):
        # L-shaped curved piece: its convex hull would be 25 % larger
        v, f = grid_mesh(8, 8, height=lambda x, y: 0.3 * np.sin(x / 2.0), skip={(i, j) for i in range(4, 8) for j in range(4, 8)})
        result = fake_result([("L", v, f)])

        surfaces, _ = ss.import_structural_surfaces(result)

        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        self.assertAlmostEqual(sum(gmsh.model.occ.getMass(2, t) for t in surfaces) / area, 1.0, delta=0.02)

    def test_staircase_edge_is_trimmed_to_the_convex_hull(self):
        # one-cell staircase along a diagonal edge: the hull is the clean straight cut
        v, f = grid_mesh(10, 10, skip={(i, j) for i in range(10) for j in range(10) if i + j > 12})
        result = fake_result([("stairs", v, f)])

        surfaces, _ = ss.import_structural_surfaces(result)

        used = v[np.unique(f)]
        hull_area = ss.ConvexHull(used[:, :2]).volume
        stairs_area = 0.5 * len(f)  # unit cells, two triangles each
        self.assertEqual(len(surfaces), 1)
        self.assertGreater(hull_area, stairs_area + 2.0)  # the hull bridges the steps
        self.assertAlmostEqual(gmsh.model.occ.getMass(2, surfaces[0]), hull_area, delta=1e-6 * hull_area)

    def test_tiny_fragments_are_skipped(self):
        big = grid_mesh(8, 8)
        tiny_v, tiny_f = grid_mesh(1, 1)
        result = fake_result([("big", *big), ("tiny", tiny_v + [20.0, 0.0, 0.0], tiny_f)])

        surfaces, _ = ss.import_structural_surfaces(result)

        self.assertEqual(len(surfaces), 1)


class MissingExtendedMaskedTestCase(unittest.TestCase):

    def test_old_structural_result_raises_a_clear_error(self):
        result = fake_result([("old", *grid_mesh(2, 2))])
        result.structural_frame.structural_groups[0].structural_elements[0].vertices = {}

        with self.assertRaisesRegex(ValueError, "surface_source='reconstructed'"):
            ss.collect_surface_pieces(result)

    def test_fault_without_extended_masked_mesh_raises(self):
        """Faults, like geological surfaces, require extended_masked meshes."""
        fault_vertices, fault_faces = grid_mesh(3, 3)
        result = fake_result([], faults=[("fault", fault_vertices, fault_faces)])
        result.structural_frame.fault_frame.fault_elements[0].vertices = {}

        with self.assertRaisesRegex(ValueError, "fault.*extended_masked"):
            ss.collect_surface_pieces(result)




class CollectSurfacePiecesRegressionTestCase(unittest.TestCase):

    def test_collects_geological_pieces_before_fault_pieces(self):
        geological = grid_mesh(3, 3)
        fault_vertices, fault_faces = grid_mesh(3, 3)
        fault_vertices = fault_vertices[:, [2, 0, 1]]
        result = fake_result([("layer", *geological)], faults=[("fault", fault_vertices, fault_faces)])
        pieces = ss.collect_surface_pieces(result)
        self.assertEqual([(p.name, p.is_fault) for p in pieces],
                         [("layer", False), ("fault", True)])

    def test_discards_small_extended_masked_fragments_without_cleaning(self):
        """Small surface patches are filtered directly from extended_masked meshes."""
        large = grid_mesh(3, 3)  # area 9, above the 4-cell minimum
        tiny_v, tiny_f = grid_mesh(1, 1)  # area 1, below the minimum
        result = fake_result([
            ("main", *large),
            ("tiny", tiny_v + [20.0, 0.0, 0.0], tiny_f),
        ])
        pieces = ss.collect_surface_pieces(result)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].name, "main")
        self.assertFalse(pieces[0].is_fault)
        self.assertEqual(len(pieces[0].faces), 18)

    def test_collects_multiple_disconnected_pieces_from_one_element(self):
        first_v, first_f = grid_mesh(3, 3)
        second_v, second_f = grid_mesh(3, 3)
        vertices = np.vstack((first_v, second_v + [10.0, 0.0, 0.0]))
        faces = np.vstack((first_f, second_f + len(first_v)))
        result = fake_result([("layer", vertices, faces)])
        pieces = ss.collect_surface_pieces(result)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(p.name == "layer" for p in pieces))
        self.assertTrue(all(len(p.faces) == 18 for p in pieces))


if __name__ == "__main__":
    unittest.main()
