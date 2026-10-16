"""
Dual contouring on octree cells: the DC core on analytic fields, the gap
mitigation, groups (fault domains), and the UNMASKED meshes of the octree
fault and structural steps.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

import core.structural_modeling_components.general as general
import core.structural_modeling_components.general_faults as gf
from core.structural_modeling_components.structural_modeling_utility.dual_contouring import (
    contour_cells,
    dual_contour,
    interior_open_edges,
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

def _no_gap_warnings(caplog):
    return not [r for r in caplog.records if "gaps" in r.getMessage()]


@pytest.fixture(scope="module")
def dipping_fault_result():
    pytest.importorskip("gempy")
    from test_octree_fault_model import octree_grid, two_faults_input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.compute_fault_domains(gf.build_fault_frame(two_faults_input(), octree_grid()))


def test_fault_step_stores_unmasked_fault_surfaces_on_the_fault_planes(dipping_fault_result):
    frame = dipping_fault_result.fault_frame
    planes = {"F1": lambda v: v[:, 0] - (0.27 + 0.4 * v[:, 2]), "F2": lambda v: v[:, 0] - 0.85}
    for fault in frame.fault_elements:
        assert set(fault.vertices) == {FaultMeshType.UNMASKED}
        vertices = fault.vertices[FaultMeshType.UNMASKED]
        faces = np.asarray(fault.edges[FaultMeshType.UNMASKED], dtype=np.int64)
        assert faces.shape[0] > 0 and faces.max() < len(vertices)
        # GemPy's fit of planar input data; within a fraction of a finest cell.
        assert np.abs(planes[fault.name](vertices)).max() < 0.3 * frame.grid.finest_spacing[0]


def test_structural_step_stores_gap_free_unmasked_surfaces(caplog):
    from test_octree_structural_model import octree_grid, unconformity_input
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=octree_grid())
        result = general.compute_structural_model(frame, extract_meshes=True).structural_frame
    assert _no_gap_warnings(caplog)
    expected_z = {"rock4": 0.82, "rock3": 0.62}
    for group in result.structural_groups:
        for elem in group.structural_elements:
            assert set(elem.vertices) == {GeoMeshType.UNMASKED}
            vertices = elem.vertices[GeoMeshType.UNMASKED]
            faces = elem.edges[GeoMeshType.UNMASKED]
            assert len(vertices) > 0 and faces.dtype == np.int32 and faces.max() < len(vertices)
            if elem.name in expected_z:  # the horizontal surfaces of the upper group
                np.testing.assert_allclose(vertices[:, 2], expected_z[elem.name], atol=0.01)
