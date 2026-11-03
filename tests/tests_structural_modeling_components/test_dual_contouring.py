"""
Dual contouring on octree cells: the DC core on analytic fields, the gap
mitigation, groups (fault domains), and the UNMASKED meshes of the octree
fault and structural steps.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general
from core.object_components import InputData_StructuralElements
import core.structural_modeling_components.general_faults as gf
import core.structural_modeling_components.general_octree as general_octree
from core.structural_modeling_components.structural_modeling_utility.dual_contouring import (
    contour_cells,
    dual_contour,
    interior_open_edges,
    restrict_to_kept_cells,
    select_faces,
    padding_shell,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    balance_mesh_depth_adjacency,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    GapMitigation,
    OctreeCells,
    OctreeGrid,
)
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultMeshType,
    GeoMeshType,
)


def grid(gap_mitigation=GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY, max_level=4) -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 1.0, 0.0, 2.0, 0.0, 1.0), base_resolution=(2, 4, 2), max_level=max_level,
                      gap_mitigation=gap_mitigation)


def plane(p):
    return 0.3 * p[:, 0] + 0.2 * p[:, 1] + p[:, 2]


CENTER = np.array([0.5, 1.0, 0.5])


def sphere(p):
    return np.linalg.norm(p - CENTER, axis=1)


def contour(g: OctreeGrid, field, isovalue, strictly_above=False):
    cells = refine_cells(g, lambda p: (field(p) > isovalue if strictly_above else field(p) >= isovalue).astype(int))
    dc_cells = contour_cells(g, cells)
    return dual_contour(g, dc_cells.ijk, np.zeros(len(dc_cells), dtype=int), lambda p, _: field(p), [isovalue],
                        strictly_above=strictly_above)[0]


def contains(points, subset, atol=1e-9):
    """Whether every point of ``subset`` is one of ``points`` up to rounding."""
    from scipy.spatial import cKDTree
    return bool(len(subset) == 0 or cKDTree(points).query(subset)[0].max() <= atol)


def face_normals(mesh):
    v, f = mesh.vertices, mesh.faces
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), v[f].mean(axis=1)


# -----------------------------------------------------------------------------
# DC core
# -----------------------------------------------------------------------------

def test_a_plane_is_reproduced_exactly():
    mesh = contour(grid(), plane, 0.55)
    assert mesh.faces.shape[0] > 0
    np.testing.assert_allclose(plane(mesh.vertices), 0.55, atol=1e-12)
    assert mesh.n_missing_quads == 0
    assert interior_open_edges(mesh, grid()) == 0


def test_a_sphere_is_closed_and_its_normals_point_outward():
    mesh = contour(grid(), sphere, 0.33)
    assert interior_open_edges(mesh, grid()) == 0
    # Closed: every edge is used by exactly two triangles.
    edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    np.testing.assert_allclose(sphere(mesh.vertices), 0.33, atol=0.01 * 0.33)
    normals, centers = face_normals(mesh)
    assert np.all(np.sum(normals * (centers - CENTER), axis=1) > 0)  # toward increasing values


def test_vertices_stay_inside_their_cells():
    g = grid()
    mesh = contour(g, sphere, 0.33)
    lo = g.origin + mesh.vertex_cells * g.finest_spacing
    assert np.all(mesh.vertices >= lo - 1e-12) and np.all(mesh.vertices <= lo + g.finest_spacing + 1e-12)


def test_the_side_rule_must_match_the_refinement_labels():
    """A surface exactly on a lattice plane: the crossings lie where the refinement put max-level cells."""
    g = grid(GapMitigation.NONE)
    x_plane = lambda p: p[:, 0]  # noqa: E731
    isovalue = 0.5  # a lattice plane at every level
    strict = contour(g, x_plane, isovalue, strictly_above=True)
    assert strict.faces.shape[0] > 0 and strict.n_missing_quads == 0
    np.testing.assert_allclose(strict.vertices[:, 0], isovalue, atol=g.finest_spacing[0])
    # Refined with ">" but contoured with ">=": the crossings fall into unrefined cells.
    cells = refine_cells(g, lambda p: (x_plane(p) > isovalue).astype(int))
    dc_cells = contour_cells(g, cells)
    mismatched = dual_contour(g, dc_cells.ijk, np.zeros(len(dc_cells), dtype=int), lambda p, _: x_plane(p),
                              [isovalue])[0]
    assert mismatched.faces.shape[0] == 0


def test_several_isovalues_at_once_equal_one_at_a_time():
    g = grid()
    cells = contour_cells(g, refine_cells(g, lambda p: np.searchsorted([0.4, 0.7], plane(p), side="right")))
    groups = np.zeros(len(cells), dtype=int)
    both = dual_contour(g, cells.ijk, groups, lambda p, _: plane(p), [0.4, 0.7])
    for mesh, isovalue in zip(both, [0.4, 0.7]):
        single = dual_contour(g, cells.ijk, groups, lambda p, _: plane(p), [isovalue])[0]
        np.testing.assert_array_equal(mesh.vertices, single.vertices)
        np.testing.assert_array_equal(mesh.faces, single.faces)


# -----------------------------------------------------------------------------
# Cell masks (MASKED meshes)
# -----------------------------------------------------------------------------

def _sphere_cells(g):
    cells = contour_cells(g, refine_cells(g, lambda p: (sphere(p) >= 0.33).astype(int)))
    return cells, np.zeros(len(cells), dtype=int)


def test_keeping_every_cell_gives_the_unmasked_mesh():
    g = grid()
    cells, groups = _sphere_cells(g)
    plain = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33])[0]
    kept = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33],
                        cell_kept=np.ones(len(cells), dtype=bool))[0]
    np.testing.assert_array_equal(kept.vertices, plain.vertices)
    np.testing.assert_array_equal(kept.faces, plain.faces)


def test_restricting_the_unmasked_mesh_equals_contouring_with_the_cell_mask():
    """MASKED is taken from UNMASKED; it must be exactly what contouring with the mask gives."""
    g = grid()
    cells, groups = _sphere_cells(g)
    kept_cells = cells.centers(g)[:, 0] < 0.5
    plain = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33])[0]
    masked = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33], cell_kept=kept_cells)[0]
    restricted = restrict_to_kept_cells(plain, kept_cells)
    np.testing.assert_array_equal(restricted.vertices, masked.vertices)
    np.testing.assert_array_equal(restricted.faces, masked.faces)
    np.testing.assert_array_equal(restricted.vertex_cells, masked.vertex_cells)
    np.testing.assert_array_equal(restricted.vertex_rows, masked.vertex_rows)
    np.testing.assert_array_equal(restricted.vertex_at_group_boundary, masked.vertex_at_group_boundary)


def test_a_cell_mask_keeps_only_quads_whose_four_cells_are_kept():
    g = grid()
    cells, groups = _sphere_cells(g)
    kept_cells = cells.centers(g)[:, 0] < 0.5
    plain = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33])[0]
    masked = dual_contour(g, cells.ijk, groups, lambda p, _: sphere(p), [0.33], cell_kept=kept_cells)[0]
    assert 0 < masked.faces.shape[0] < plain.faces.shape[0]
    assert masked.n_missing_quads == 0  # the cut is no gap
    # Every vertex is used, comes from a kept cell, and is one of the unmasked vertices.
    assert np.array_equal(np.unique(masked.faces), np.arange(len(masked.vertices)))
    assert np.all(g.cell_centers(g.max_level, masked.vertex_cells)[:, 0] < 0.5)
    plain_rows = {tuple(v) for v in plain.vertices}
    assert all(tuple(v) in plain_rows for v in masked.vertices)
    # The kept half of the sphere: everything strictly inside x < 0.5 - one cell is still there.
    deep = plain.vertices[:, 0] < 0.5 - 1.5 * g.finest_spacing[0]
    masked_rows = {tuple(v) for v in masked.vertices}
    assert all(tuple(v) in masked_rows for v in plain.vertices[deep])


def test_select_faces_keeps_the_used_vertices_in_order():
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    faces = np.array([[0, 1, 2], [2, 3, 4], [1, 3, 4]])
    v, f, used = select_faces(vertices, faces, np.array([False, True, True]))
    np.testing.assert_array_equal(used, [1, 2, 3, 4])
    np.testing.assert_array_equal(v, vertices[1:])
    np.testing.assert_array_equal(v[f], vertices[faces[1:]])


# -----------------------------------------------------------------------------
# Padding shell (EXTENDED meshes)
# -----------------------------------------------------------------------------

def test_the_padding_shell_is_the_padded_box_minus_the_extent():
    g = grid(max_level=2)  # finest resolution (8, 16, 8)
    shell = padding_shell(g, 2)
    n = np.array(g.finest_resolution)
    assert len(shell) == np.prod(n + 4) - np.prod(n)
    assert len({tuple(c) for c in shell}) == len(shell)
    assert np.all(np.any((shell < 0) | (shell >= n), axis=1))
    assert np.all((shell >= -2) & (shell < n + 2))


def _extended(g, field, isovalue, padding=2):
    cells = contour_cells(g, refine_cells(g, lambda p: (field(p) >= isovalue).astype(int)))
    ijk = np.concatenate([cells.ijk, padding_shell(g, padding)])
    plain = dual_contour(g, cells.ijk, np.zeros(len(cells), dtype=int), lambda p, _: field(p), [isovalue])[0]
    extended = dual_contour(g, ijk, np.zeros(len(ijk), dtype=int), lambda p, _: field(p), [isovalue],
                            padding=padding)[0]
    return plain, extended


def test_an_extended_plane_crosses_the_box_and_contains_the_unmasked_plane():
    g = grid()
    plain, extended = _extended(g, plane, 0.55)
    np.testing.assert_allclose(plane(extended.vertices), 0.55, atol=1e-12)
    assert extended.n_missing_quads == 0 and interior_open_edges(extended, g, padding=2) == 0
    lo, hi = np.array(g.extent[0::2]), np.array(g.extent[1::2])
    for axis in (0, 1):  # the plane runs through the x and y walls
        assert extended.vertices[:, axis].min() < lo[axis] - g.finest_spacing[axis]
        assert extended.vertices[:, axis].max() > hi[axis] + g.finest_spacing[axis]
    # The vertex of a cell depends only on its own edges: the unmasked vertices reappear unchanged.
    assert {tuple(v) for v in plain.vertices} <= {tuple(v) for v in extended.vertices}


def test_normal_stencils_are_clamped_into_the_field_bounds():
    """A field that is NaN outside the padded box still gives finite vertices up to its outer face."""
    g = grid()
    pad = 2 * g.finest_spacing
    bounds = (-pad[0], 1 + pad[0], -pad[1], 2 + pad[1], -pad[2], 1 + pad[2])
    lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])

    def bounded(p):
        inside = np.all((p >= lo - 1e-12) & (p <= hi + 1e-12), axis=1)
        return np.where(inside, plane(p), np.nan)

    cells = contour_cells(g, refine_cells(g, lambda p: (plane(p) >= 0.55).astype(int)))
    ijk = np.concatenate([cells.ijk, padding_shell(g, 2)])
    mesh = dual_contour(g, ijk, np.zeros(len(ijk), dtype=int), lambda p, _: bounded(p), [0.55], padding=2,
                        field_bounds=bounds)[0]
    assert mesh.faces.shape[0] > 0 and np.all(np.isfinite(mesh.vertices))
    np.testing.assert_allclose(plane(mesh.vertices), 0.55, atol=1e-12)


# -----------------------------------------------------------------------------
# Groups (fault domains)
# -----------------------------------------------------------------------------

def test_each_group_is_contoured_with_its_own_field_and_ends_at_the_group_boundary():
    """Two groups split at y = 1, each with its own horizontal surface (z = 0.3 and z = 0.6)."""
    g = grid()
    group_of = lambda p: (p[:, 1] >= 1.0).astype(int)  # noqa: E731
    field = lambda p, d: p[:, 2] - np.where(d == 0, 0.3, 0.6)  # noqa: E731
    cells = refine_cells(g, lambda p, d: (field(p, d) >= 0).astype(int),
                         cell_domains=lambda level, ijk: group_of(g.cell_centers(level, ijk)))
    dc_cells = contour_cells(g, cells)
    groups = group_of(dc_cells.centers(g))
    mesh = dual_contour(g, dc_cells.ijk, groups, field, [0.0])[0]

    vertex_groups = group_of(g.cell_centers(g.max_level, mesh.vertex_cells))
    np.testing.assert_allclose(mesh.vertices[vertex_groups == 0, 2], 0.3, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[vertex_groups == 1, 2], 0.6, atol=1e-12)
    # No triangle joins the two groups, and the open edges at the boundary are no gaps.
    assert np.all(vertex_groups[mesh.faces].min(axis=1) == vertex_groups[mesh.faces].max(axis=1))
    assert mesh.n_missing_quads == 0
    assert interior_open_edges(mesh, g) == 0
    assert mesh.vertex_at_group_boundary.any()


# -----------------------------------------------------------------------------
# Gap mitigation
# -----------------------------------------------------------------------------

def test_balancing_splits_the_coarse_face_neighbours_of_max_level_cells():
    g = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(1, 1, 1), max_level=2)
    # One max-level cell in the corner, the rest coarse.
    cells = refine_cells(g, lambda p: (np.all(p <= 0.25, axis=1)).astype(int))
    balanced = balance_mesh_depth_adjacency(g, cells)
    assert (8 ** (g.max_level - balanced.level.astype(int))).sum() == np.prod(g.finest_resolution)
    fine = {tuple(c) for c in balanced.ijk[balanced.level == g.max_level]}
    # Face neighbours of the corner cell (0, 0, 0) are at max level now ...
    assert {(1, 0, 0), (0, 1, 0), (0, 0, 1)} <= fine
    # ... but the far level-1 cells are not.
    assert np.sum(balanced.level == 1) > 0


def test_balancing_leaves_an_octree_without_max_level_cells_alone():
    g = grid(max_level=2)
    coarse = refine_cells(g, lambda p: np.zeros(len(p), dtype=int))
    assert balance_mesh_depth_adjacency(g, coarse) is coarse


def test_without_mitigation_a_grazing_surface_leaves_gaps_that_balancing_closes():
    """A surface that only touches corners of coarse cells: refinement misses it there."""
    field = lambda p: p[:, 2] + 0.08 * np.sin(9.0 * p[:, 0]) * np.sin(7.0 * p[:, 1])  # noqa: E731
    gaps = {m: contour(grid(m, max_level=5), field, 0.5) for m in GapMitigation}
    plain = gaps[GapMitigation.NONE]
    balanced = gaps[GapMitigation.BALANCE_MESH_DEPTH_ADJACENCY]
    assert plain.n_missing_quads > 0 and interior_open_edges(plain, grid(max_level=5)) > 0
    assert balanced.n_missing_quads == 0 and interior_open_edges(balanced, grid(max_level=5)) == 0


# -----------------------------------------------------------------------------
# Fault and structural steps
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dipping_fault_result():
    pytest.importorskip("gempy")
    from test_octree_fault_model import octree_grid, two_faults_input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.compute_fault_domains(gf.build_fault_frame(two_faults_input(), octree_grid()))


def test_fault_step_stores_unmasked_and_extended_fault_surfaces_on_the_fault_planes(dipping_fault_result):
    frame = dipping_fault_result.fault_frame
    cell = frame.grid.finest_spacing
    planes = {"F1": lambda v: v[:, 0] - (0.27 + 0.4 * v[:, 2]), "F2": lambda v: v[:, 0] - 0.85}
    for fault in frame.fault_elements:
        assert set(fault.vertices) == {FaultMeshType.UNMASKED, FaultMeshType.EXTENDED}
        for kind in (FaultMeshType.UNMASKED, FaultMeshType.EXTENDED):
            vertices = fault.vertices[kind]
            faces = np.asarray(fault.edges[kind], dtype=np.int64)
            assert faces.shape[0] > 0 and faces.max() < len(vertices)
            # GemPy's fit of planar input data; within a fraction of a finest cell.
            assert np.abs(planes[fault.name](vertices)).max() < 0.3 * cell[0]
        extended = fault.vertices[FaultMeshType.EXTENDED]
        # Through the y and z walls of the box, and the unmasked vertices reappear unchanged.
        for axis in (1, 2):
            assert extended[:, axis].min() < -cell[axis] and extended[:, axis].max() > 1 + cell[axis]
        assert contains(extended, fault.vertices[FaultMeshType.UNMASKED])


@pytest.fixture(scope="module")
def unconformity_result():
    """Upper group horizontal at z = 0.82, 0.62; lower group dipping 45 degrees, eroded above z = 0.62."""
    from test_octree_structural_model import octree_grid, unconformity_input
    logger = logging.getLogger("core.structural_modeling_components.general_octree")
    records = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=octree_grid())
            result = general.compute_structural_model(frame, extract_meshes=True).structural_frame
    finally:
        logger.removeHandler(handler)
    return result, records


def test_structural_step_stores_gap_free_unmasked_surfaces(unconformity_result):
    result, records = unconformity_result
    assert not [r for r in records if "gaps" in r.getMessage()]
    expected_z = {"rock4": 0.82, "rock3": 0.62}
    for group in result.structural_groups:
        for elem in group.structural_elements:
            assert set(elem.vertices) == set(GeoMeshType)
            vertices = elem.vertices[GeoMeshType.UNMASKED]
            faces = elem.edges[GeoMeshType.UNMASKED]
            assert len(vertices) > 0 and faces.dtype == np.int32 and faces.max() < len(vertices)
            if elem.name in expected_z:  # the horizontal surfaces of the upper group
                np.testing.assert_allclose(vertices[:, 2], expected_z[elem.name], atol=0.01)


def test_masked_surfaces_stop_at_the_unconformity(unconformity_result):
    result, _ = unconformity_result
    cell = result.grid.finest_spacing
    upper, lower = result.structural_groups
    for elem in upper.structural_elements:  # nothing younger: MASKED is UNMASKED
        np.testing.assert_array_equal(elem.vertices[GeoMeshType.MASKED], elem.vertices[GeoMeshType.UNMASKED])
        np.testing.assert_array_equal(elem.edges[GeoMeshType.MASKED], elem.edges[GeoMeshType.UNMASKED])
    for elem in lower.structural_elements:
        unmasked = elem.vertices[GeoMeshType.UNMASKED]
        masked = elem.vertices[GeoMeshType.MASKED]
        faces = elem.edges[GeoMeshType.MASKED]
        assert faces.dtype == np.int32 and 0 < len(masked) < len(unmasked)
        assert {tuple(v) for v in masked} <= {tuple(v) for v in unmasked}
        # Below the unconformity at z = 0.62 (not even a vertex past it, within the fit's
        # accuracy), up to it within a cell, and complete beneath it.
        assert masked[:, 2].max() < 0.62 + 0.01
        assert masked[:, 2].max() > 0.62 - 2 * cell[2]
        below = unmasked[:, 2] < 0.62 - 2 * cell[2]
        assert {tuple(v) for v in unmasked[below]} <= {tuple(v) for v in masked}


def test_flat_surfaces_on_a_corner_plane_give_a_regular_mesh():
    """
    Horizons at z = 0.25 and 0.5 lie exactly on planes of cell corners, where the
    fitted field equals the isovalue only up to rounding. Snapped to the isovalue,
    every crossing falls into the same cell layer: each vertex sits at its cell's
    center in x and y (a regular triangulation), none is pulled toward a neighbour.
    """
    xs = np.linspace(0.05, 0.95, 7)
    rows = [{"X": float(x), "Y": float(y), "Z": z, "formation": name}
            for name, z in (("top", 0.5), ("bottom", 0.25)) for x in xs for y in xs]
    data = InputData_StructuralElements(
        name="flat", mapping_object={"Series": ("top", "bottom")}, surface_points=pd.DataFrame(rows),
        orientations=pd.DataFrame([dict(row, G_x=0.0, G_y=0.0, G_z=1.0) for row in rows]))
    grid = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(2, 2, 2), max_level=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=data, grid=grid)
        result = general.compute_structural_model(frame, extract_meshes=True).structural_frame
    cell = result.grid.finest_spacing
    for elem in result.structural_groups[0].structural_elements:
        vertices = elem.vertices[GeoMeshType.UNMASKED]
        np.testing.assert_allclose(vertices[:, 2], {"top": 0.5, "bottom": 0.25}[elem.name], atol=1e-6)
        in_cell = vertices[:, :2] / cell[:2] - 0.5
        np.testing.assert_allclose(in_cell, np.round(in_cell), atol=1e-6)


def test_extended_surfaces_cross_the_box_and_contain_the_unmasked_surfaces(unconformity_result):
    result, _ = unconformity_result
    cell = result.grid.finest_spacing
    for group in result.structural_groups:
        for elem in group.structural_elements:
            extended = elem.vertices[GeoMeshType.EXTENDED]
            faces = elem.edges[GeoMeshType.EXTENDED]
            assert faces.dtype == np.int32 and faces.max() < len(extended)
            assert contains(extended, elem.vertices[GeoMeshType.UNMASKED])
            assert extended[:, 1].min() < -cell[1] and extended[:, 1].max() > 1 + cell[1]  # all run along y
            if elem.name in ("rock4", "rock3"):  # horizontal: through the x walls, and still flat
                assert extended[:, 0].min() < -cell[0] and extended[:, 0].max() > 1 + cell[0]
                np.testing.assert_allclose(extended[:, 2], {"rock4": 0.82, "rock3": 0.62}[elem.name], atol=0.01)


def mesh_pieces(vertices, faces):
    """The connected pieces of a mesh, as arrays of vertex indices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    _, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == label) for label in np.unique(labels[np.unique(faces)])]


def test_extended_masked_surfaces_stop_past_the_base_of_the_younger_groups(unconformity_result):
    """Without faults: one piece per element, EXTENDED for the youngest group, and EXTENDED
    trimmed just above the unconformity (z = 0.62) for the eroded group, as on the regular grid."""
    result, _ = unconformity_result
    cell = result.grid.finest_spacing
    o = general._EXTENDED_MASKED_OVERSHOOT_CELLS
    upper, lower = result.structural_groups
    for elem in upper.structural_elements:  # nothing younger: the whole EXTENDED mesh
        np.testing.assert_array_equal(elem.vertices[GeoMeshType.EXTENDED_MASKED], elem.vertices[GeoMeshType.EXTENDED])
        np.testing.assert_array_equal(elem.edges[GeoMeshType.EXTENDED_MASKED], elem.edges[GeoMeshType.EXTENDED])
    for elem in lower.structural_elements:
        extended = elem.vertices[GeoMeshType.EXTENDED]
        kept = elem.vertices[GeoMeshType.EXTENDED_MASKED]
        faces = elem.edges[GeoMeshType.EXTENDED_MASKED]
        assert faces.dtype == np.int32 and 0 < len(kept) < len(extended) and faces.max() < len(kept)
        assert contains(extended, kept)
        # past the unconformity by at least about the overshoot, by at most the overshoot + one cell
        assert 0.62 < kept[:, 2].max() <= 0.62 + (o + 1) * cell[2] + 0.01
        assert contains(kept, extended[extended[:, 2] < 0.62 - cell[2]])
        assert kept[:, 1].min() < -cell[1] and kept[:, 1].max() > 1 + cell[1]  # still crosses the box
        assert len(mesh_pieces(kept, faces)) == 1


def test_extended_masked_surfaces_are_one_piece_per_side_of_a_fault_that_cuts_them(model2_frames):
    """Model 2, F1 (x = 0.5) cutting only the lower group: each lower element has a piece per side,
    from that side's field, overshooting the fault by at most the overshoot (+ one cell) and not
    sharing vertices; the upper group, which F1 doesn't cut, keeps its whole EXTENDED mesh."""
    _, frames = model2_frames
    frame = frames["LowerSeries"]
    cell = frame.grid.finest_spacing
    o = general._EXTENDED_MASKED_OVERSHOOT_CELLS
    upper, lower = frame.structural_groups
    for elem in upper.structural_elements:
        np.testing.assert_array_equal(elem.vertices[GeoMeshType.EXTENDED_MASKED], elem.vertices[GeoMeshType.EXTENDED])
    for elem in lower.structural_elements:
        vertices = elem.vertices[GeoMeshType.EXTENDED_MASKED]
        faces = elem.edges[GeoMeshType.EXTENDED_MASKED]
        assert faces.dtype == np.int32 and len(faces) > 0
        pieces = mesh_pieces(vertices, faces)
        sides = set()
        for piece in pieces:
            x = vertices[piece, 0]
            if x.min() < 0.0:  # the footwall piece, from the left wall across the fault a little
                assert 0.5 + o * cell[0] <= x.max() <= 0.5 + (o + 1) * cell[0]
                sides.add("left")
            else:              # the hanging-wall piece, from the right wall
                assert x.max() > 1.0
                assert 0.5 - (o + 1) * cell[0] <= x.min() <= 0.5 - o * cell[0]
                sides.add("right")
        assert sides == {"left", "right"}, elem.name
        # and they stop just above the unconformity at z = 0.68
        assert vertices[:, 2].max() <= 0.68 + (o + 1) * cell[2] + 0.01


def test_combined_surfaces_are_the_lithology_contacts(unconformity_result):
    """The boundary of 'lithology id >= the element's id', with the erosion: from the lithology itself."""
    from scipy.spatial import cKDTree
    result, _ = unconformity_result
    cell = result.grid.finest_spacing
    elements = {e.name: e for g in result.structural_groups for e in g.structural_elements}
    for name in elements:
        faces = elements[name].edges[GeoMeshType.COMBINED]
        assert faces.dtype == np.int32 and len(faces) > 0
    combined = {name: e.vertices[GeoMeshType.COMBINED] for name, e in elements.items()}
    # The upper group is not eroded: its bases are the horizontal surfaces, within a cell.
    np.testing.assert_allclose(combined["rock4"][:, 2], 0.82, atol=cell[2])
    np.testing.assert_allclose(combined["rock3"][:, 2], 0.62, atol=cell[2])
    # The lower group: its own surface below the unconformity, and the unconformity above
    # the older rock it erodes down to.
    for name in ("rock2", "rock1"):
        to_surface = cKDTree(elements[name].vertices[GeoMeshType.MASKED]).query(combined[name])[0]
        on_unconformity = np.abs(combined[name][:, 2] - 0.62) <= cell[2]
        assert np.all((to_surface <= 1.5 * cell.max()) | on_unconformity), name
        assert np.any(to_surface <= 1.5 * cell.max()), name
    # rock2 continues along the unconformity over the eroded rock1, beyond the end of its own surface
    # (x ~ 0.7) to the wall; rock1 meets the unconformity only next to the end of its own (x > 0.92).
    rock2 = combined["rock2"]
    assert rock2[np.abs(rock2[:, 2] - 0.62) <= cell[2], 0].max() > 0.9


@pytest.fixture(scope="module")
def model2_frames():
    """Model 2 of the integration tests: vertical fault at x = 0.5, unconformity at z = 0.68."""
    from test_octree_faulted_structural_model import MODELS, _integration_module, octree_grid
    elements, faults = _integration_module(MODELS["model2"][0])._build_input_data()
    grid = octree_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fault_result = gf.compute_fault_domains(gf.build_fault_frame(faults, grid))
        frames = {}
        for youngest in ["UpperSeries", "LowerSeries"]:
            frame = general.build_structural_frame(
                input_data_elements=elements, grid=grid, fault_model_results=fault_result)
            frame.set_fault_activity_by_group("F1", youngest)
            frames[youngest] = general.compute_structural_model(frame, verbose=False).structural_frame
    return fault_result, frames


def test_fault_masked_meshes_stop_at_the_youngest_unaffected_group(model2_frames):
    fault_result, frames = model2_frames
    # The fault model's own result is not changed.
    assert set(fault_result.fault_frame.fault_elements[0].vertices) == {FaultMeshType.UNMASKED,
                                                                        FaultMeshType.EXTENDED}

    cuts_all = frames["UpperSeries"].fault_frame.fault_elements[0]  # no unaffected group: kept whole
    for source, target in [(FaultMeshType.UNMASKED, FaultMeshType.MASKED),
                           (FaultMeshType.EXTENDED, FaultMeshType.EXTENDED_MASKED)]:
        np.testing.assert_array_equal(cuts_all.vertices[target], cuts_all.vertices[source])
        np.testing.assert_array_equal(cuts_all.edges[target], cuts_all.edges[source])

    fault = frames["LowerSeries"].fault_frame.fault_elements[0]  # cut off where UpperSeries is
    cell = frames["LowerSeries"].grid.finest_spacing
    o = general._EXTENDED_MASKED_OVERSHOOT_CELLS
    for source, target in [(FaultMeshType.UNMASKED, FaultMeshType.MASKED),
                           (FaultMeshType.EXTENDED, FaultMeshType.EXTENDED_MASKED)]:
        full = fault.vertices[source]
        kept = fault.vertices[target]
        faces = fault.edges[target]
        assert faces.dtype == np.int32 and faces.max() < len(kept)
        if target == FaultMeshType.MASKED:  # stops at the unconformity
            assert 0.68 - 2 * cell[2] < kept[:, 2].max() < 0.68 + cell[2]
        else:  # overshoots it, as on the regular grid, but by no more than the overshoot (+ one cell)
            assert 0.68 < kept[:, 2].max() <= 0.68 + (o + 1) * cell[2]
        below = full[:, 2] < 0.68 - 2 * cell[2]
        assert {tuple(v) for v in full[below]} <= {tuple(v) for v in kept}
    # EXTENDED_MASKED still crosses the box below the unconformity.
    extended_masked = fault.vertices[FaultMeshType.EXTENDED_MASKED]
    assert extended_masked[:, 1].min() < -cell[1] and extended_masked[:, 1].max() > 1 + cell[1]
    assert extended_masked[:, 2].min() < -cell[2]


def test_surfaces_run_through_faults_that_do_not_affect_their_group(model2_frames):
    _, frames = model2_frames
    upper, lower = frames["LowerSeries"].structural_groups
    for group, crosses in [(upper, True), (lower, False)]:
        for elem in group.structural_elements:
            for kind in (GeoMeshType.UNMASKED, GeoMeshType.MASKED, GeoMeshType.EXTENDED):
                x = elem.vertices[kind][elem.edges[kind]][..., 0]
                assert np.any((x.min(axis=1) < 0.5) & (x.max(axis=1) > 0.5)) == crosses, (elem.name, kind)


class _HeightField:
    """A fitted field that is the height z, with the given element scalar values."""

    def __init__(self, scalar_values):
        self.scalar_values = scalar_values

    def evaluate(self, xyz):
        return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, 2].copy()


def test_fault_extended_masked_overshoots_the_base_of_all_younger_groups(monkeypatch):
    """As on the regular grid: the fault's EXTENDED_MASKED mesh ends at the base of every group
    younger than the youngest group it cuts, plus the overshoot, even if that base is not the
    one directly above."""
    from core.structural_modeling_components.structural_objects.structural_objects import (
        FaultElement, StructuralElement, StructuralFrame, StructuralGroup)

    g = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(3, 2, 3), max_level=3)
    groups, fields = [], []
    for name, sval in [("Top", 0.3), ("Mid", 0.5), ("Cut", 0.2)]:  # Top's base lies below Mid's
        elem = StructuralElement(name=f"{name}_base")
        elem.set_scalar_value(sval)
        groups.append(StructuralGroup(name=name, structural_elements=[elem]))
        fields.append({0: _HeightField({elem.name: sval})})
    frame = StructuralFrame(structural_groups=groups)
    frame.grid = g
    fault = FaultElement("F")
    # A vertical plane at x = 0.5 through the padded box, on a lattice finer than the cells.
    ys, zs = np.meshgrid(np.linspace(-0.08, 1.08, 57), np.linspace(-0.08, 1.08, 57), indexing="ij")
    vertices = np.column_stack([np.full(ys.size, 0.5), ys.ravel(), zs.ravel()])
    idx = np.arange(ys.size).reshape(ys.shape)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    fault.set_mesh(FaultMeshType.EXTENDED, vertices, np.concatenate([np.column_stack([a, b, c]),
                                                                     np.column_stack([a, c, d])]))
    frame.fault_frame = type("FakeFaultFrame", (), {"fault_elements": [fault]})()
    frame.fault_activity = {"F": 2}  # cuts "Cut" only
    monkeypatch.setattr(general_octree, "_point_domains",
                        lambda frame, points: np.zeros(len(points), dtype=np.int64))

    general_octree._extract_fault_masked_meshes(frame, fields, stored_masks=np.zeros((3, 1), dtype=bool))

    kept = fault.vertices[FaultMeshType.EXTENDED_MASKED]
    cell = g.finest_spacing
    o = general._EXTENDED_MASKED_OVERSHOOT_CELLS
    assert 0.3 < kept[:, 2].max() <= 0.3 + (o + 1) * cell[2]
    assert kept[:, 2].min() < 0.0                                # still crosses the bottom of the box
    assert kept[:, 1].min() < 0.0 and kept[:, 1].max() > 1.0     # and the box along y
    # everything well below the base is kept
    below = vertices[:, 2] < 0.3 - cell[2]
    assert contains(kept, vertices[below])


def test_a_fault_under_a_group_covering_the_model_has_no_extended_masked_mesh(model2_frames, monkeypatch):
    """As on the regular grid: MASKED falls back to the whole fault, EXTENDED_MASKED does not."""
    _, frames = model2_frames
    frame = frames["LowerSeries"].model_copy(deep=True)
    n_groups = len(frame.structural_groups)
    monkeypatch.setattr(general_octree, "_age_mask_at_vertex_cells",
                        lambda frame, fields, vertices: np.ones((n_groups, len(vertices)), dtype=bool))
    monkeypatch.setattr(general_octree, "_below_younger_groups",
                        lambda frame, fields, group_idx, points: np.zeros(len(points), dtype=bool))
    stored_masks = np.ones((n_groups, len(frame.cells)), dtype=bool)
    general_octree._extract_fault_masked_meshes(frame, fields=[], stored_masks=stored_masks)
    fault = frame.fault_frame.fault_elements[0]
    np.testing.assert_array_equal(fault.vertices[FaultMeshType.MASKED], fault.vertices[FaultMeshType.UNMASKED])
    assert len(fault.vertices[FaultMeshType.EXTENDED_MASKED]) == 0
    assert len(fault.edges[FaultMeshType.EXTENDED_MASKED]) == 0
