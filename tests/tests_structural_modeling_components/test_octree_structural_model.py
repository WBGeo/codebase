"""
Octree path of the structural model: RBF fit/evaluate, octree refinement, and the
full pipeline compared with the regular grid at the equivalent finest resolution.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic_core import to_jsonable_python

import core.structural_modeling_components.general as general
import core.structural_modeling_components.general_octree as general_octree
from core.object_components import InputData_StructuralElements, StructuralModelResults
from core.structural_modeling_components.interpolator_functions.finite_difference import (
    fit_group_finite_differences,
    interpolate_group_finite_differences,
)
from core.structural_modeling_components.interpolator_functions.geoinr import fit_group_geo_inr
from core.structural_modeling_components.interpolator_functions.ordinary_kriging import (
    fit_group_ordinary_kriging,
    interpolate_group_ordinary_kriging,
)
from core.structural_modeling_components.interpolator_functions.piecewise_linear import (
    fit_group_piecewise_linear,
    interpolate_group_piecewise_linear,
)
from core.structural_modeling_components.interpolator_functions.radial_basis_function import (
    fit_group_radial_basis_function,
    interpolate_group_radial_basis_function,
)
from core.structural_modeling_components.interpolator_functions.universal_cokriging import (
    fit_group_universal_cokriging,
    interpolate_group_universal_cokriging,
)
from core.structural_modeling_components.interpolator_functions.universal_kriging import (
    fit_group_universal_kriging,
    interpolate_group_universal_kriging,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    balance_2to1,
    base_cells,
    coarsen_cells,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    OctreeCells,
    OctreeGrid,
    RegularGrid,
)
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod


# -----------------------------------------------------------------------------
# Input: two groups with an unconformity, no faults (see test_unconformities_analytical)
#   UpperSeries: rock4, rock3 (horizontal at z = 0.82, 0.62)
#   LowerSeries: rock2, rock1 (45 degrees in x, eroded by UpperSeries)
# -----------------------------------------------------------------------------

def unconformity_input() -> InputData_StructuralElements:
    xs = np.linspace(0.05, 0.95, 5)
    surfaces = {
        "rock1": lambda x: 0.20 + (x - 0.5),
        "rock2": lambda x: 0.40 + (x - 0.5),
        "rock3": lambda x: 0.62,
        "rock4": lambda x: 0.82,
    }
    rows = [{"X": float(x), "Y": float(y), "Z": float(z_of(x)), "formation": name}
            for name, z_of in surfaces.items() for x in xs for y in xs]
    # build_structural_frame needs orientations; RBF ignores them.
    orientations = [dict(row, G_x=0.0, G_y=0.0, G_z=1.0) for row in rows]
    return InputData_StructuralElements(
        name="unconformity",
        mapping_object={"UpperSeries": ("rock4", "rock3"), "LowerSeries": ("rock2", "rock1")},
        surface_points=pd.DataFrame(rows),
        orientations=pd.DataFrame(orientations),
    )


def octree_grid(base=(2, 2, 2), max_level=3) -> OctreeGrid:
    return OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=base, max_level=max_level)


def compute(grid, method=InterpolationMethod.RADIAL_BASIS_FUNCTION) -> StructuralModelResults:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
        for group in frame.structural_groups:
            group.set_interpolation_method(method)
        return general.compute_structural_model(frame, extract_meshes=False)


# Interpolators whose octree result must equal the regular grid at the finest resolution.
DETERMINISTIC_METHODS = [
    InterpolationMethod.RADIAL_BASIS_FUNCTION,
    InterpolationMethod.FINITE_DIFFERENCES,
    InterpolationMethod.ORDINARY_KRIGING,
    InterpolationMethod.UNIVERSAL_KRIGING,
    InterpolationMethod.UNIVERSAL_COKRIGING,
]


@pytest.fixture(scope="module", params=DETERMINISTIC_METHODS, ids=lambda m: m.name)
def results(request) -> tuple[StructuralModelResults, StructuralModelResults]:
    grid = octree_grid()
    return compute(grid, request.param), compute(grid.equivalent_regular_grid(), request.param)


# -----------------------------------------------------------------------------
# RBF fit / evaluate
# -----------------------------------------------------------------------------

def test_rbf_fit_evaluates_to_the_regular_grid_interpolation():
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    sp = frame.surface_points[frame.surface_points["formation"].isin(["rock1", "rock2"])]

    on_grid, values_on_grid = interpolate_group_radial_basis_function(
        group=group, grid=grid, group_surface_points_df=sp)
    field = fit_group_radial_basis_function(group=group, group_surface_points_df=sp)

    assert field.scalar_values == values_on_grid
    # interpolate_group_* returns reshape(resolution).T; undo the transpose to compare.
    np.testing.assert_array_equal(field.evaluate(grid.grid_coordinates).reshape(grid.resolution), on_grid.T)


@pytest.mark.parametrize("interpolate, fit", [
    (interpolate_group_finite_differences, fit_group_finite_differences),
    (interpolate_group_piecewise_linear, fit_group_piecewise_linear),
], ids=["FDI", "PLI"])
def test_loopstructural_fit_evaluates_to_the_regular_grid_interpolation(interpolate, fit):
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    group.set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)  # FDI and PLI share these params
    formations = ["rock1", "rock2"]
    sp = frame.surface_points[frame.surface_points["formation"].isin(formations)]
    ori = frame.orientations[frame.orientations["formation"].isin(formations)]

    on_grid, values_on_grid = interpolate(
        group=group, grid=grid, group_surface_points_df=sp, group_orientations_points_df=ori)
    field = fit(group=group, extent=grid.extent, group_surface_points_df=sp, group_orientations_points_df=ori)

    assert field.scalar_values == values_on_grid
    np.testing.assert_allclose(field.evaluate(grid.grid_coordinates).reshape(grid.resolution), on_grid.T,
                               rtol=0, atol=1e-10)


def test_geo_inr_fit_gives_a_field_and_the_usual_scalar_values():
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    group.set_interpolation_method(InterpolationMethod.GEOINR)
    group.configure_interpolation_params(epochs=50)
    formations = ["rock1", "rock2"]
    sp = frame.surface_points[frame.surface_points["formation"].isin(formations)]
    ori = frame.orientations[frame.orientations["formation"].isin(formations)]

    field = fit_group_geo_inr(group=group, extent=grid.extent, group_surface_points_df=sp,
                              group_orientations_points_df=ori)

    # Labels run from -1 (oldest) to +1 (youngest), as in interpolate_group_geo_inr.
    assert field.scalar_values == {"rock1": -1.0, "rock2": 1.0}
    values = field.evaluate(grid.grid_coordinates)
    assert values.shape == (grid.grid_coordinates.shape[0],) and values.dtype == np.float64
    # The network runs in float32, so a point's value depends slightly (~1e-7) on the
    # batch it is evaluated in; the octree evaluates the same point in different batches.
    np.testing.assert_allclose(field.evaluate(grid.grid_coordinates[:7]), values[:7], rtol=0, atol=1e-5)


# -----------------------------------------------------------------------------
# Kriging fit / evaluate (dual kriging vs. pykrige's execute)
# -----------------------------------------------------------------------------

OK = (InterpolationMethod.ORDINARY_KRIGING, interpolate_group_ordinary_kriging, fit_group_ordinary_kriging)
UK = (InterpolationMethod.UNIVERSAL_KRIGING, interpolate_group_universal_kriging, fit_group_universal_kriging)


# A Gaussian variogram without nugget gives a kriging matrix with a condition number
# around 1e13 here, so solving it once (dual kriging) and pykrige's per-point inverse
# agree only to ~1e-3; both carry that rounding noise.
@pytest.mark.parametrize("method, interpolate, fit, params, atol", [
    (*OK, dict(neighbors=None, variogram_model="exponential"), 1e-10),
    (*OK, dict(neighbors=None, nugget=0.1), 1e-10),
    (*OK, dict(neighbors=None), 1e-3),
    (*OK, dict(neighbors=12), 1e-10),
    (*UK, dict(), 1e-10),
    (*UK, dict(nugget=0.1), 1e-10),
], ids=["OK", "OK-nugget", "OK-gaussian-no-nugget", "OK-neighbors", "UK", "UK-nugget"])
def test_kriging_fit_evaluates_to_the_regular_grid_interpolation(method, interpolate, fit, params, atol):
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    group.set_interpolation_method(method)
    group.configure_interpolation_params(**params)
    sp = frame.surface_points[frame.surface_points["formation"].isin(["rock1", "rock2"])]

    on_grid, values_on_grid = interpolate(group=group, grid=grid, group_surface_points_df=sp)
    field = fit(group=group, extent=grid.extent, group_surface_points_df=sp)

    assert field.scalar_values == values_on_grid
    # interpolate_group_* returns pykrige's (z, y, x) grid; transpose to compare.
    points = grid.grid_coordinates.copy()
    np.testing.assert_allclose(field.evaluate(points).reshape(grid.resolution), on_grid.T, rtol=0, atol=atol)
    np.testing.assert_array_equal(points, grid.grid_coordinates)  # evaluate leaves its input alone

    # At the data points kriging returns the data values, also with a nugget.
    on_data = field.evaluate(sp[["X", "Y", "Z"]].to_numpy())
    np.testing.assert_allclose(on_data, sp["formation"].map(values_on_grid).to_numpy(), rtol=0, atol=max(atol, 1e-8))


def test_kriging_with_unsupported_drift_is_refused_on_points():
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = frame.structural_groups[1]
    group.set_interpolation_method(InterpolationMethod.UNIVERSAL_KRIGING)
    group.configure_interpolation_params(drift_terms=["specified"])
    sp = frame.surface_points[frame.surface_points["formation"].isin(["rock1", "rock2"])]
    with pytest.raises((NotImplementedError, ValueError)):
        fit_group_universal_kriging(group=group, extent=grid.extent, group_surface_points_df=sp)


# -----------------------------------------------------------------------------
# GemPy (UCK) fit / evaluate
# -----------------------------------------------------------------------------

def _uck_inputs(grid, formations):
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
    group = next(g for g in frame.structural_groups if g.structural_elements[0].name in formations)
    group.set_interpolation_method(InterpolationMethod.UNIVERSAL_COKRIGING)
    sp = frame.surface_points[frame.surface_points["formation"].isin(formations)]
    ori = frame.orientations[frame.orientations["formation"].isin(formations)]
    return group, sp, ori


def test_uck_fit_evaluates_to_the_regular_grid_interpolation():
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    group, sp, ori = _uck_inputs(grid, ["rock1", "rock2"])

    on_grid, values_on_grid = interpolate_group_universal_cokriging(
        group=group, grid=grid, group_surface_points_df=sp, group_orientations_points_df=ori)
    field = fit_group_universal_cokriging(group=group, extent=grid.extent, group_surface_points_df=sp,
                                          group_orientations_points_df=ori)

    assert field.scalar_values == values_on_grid
    points = grid.grid_coordinates.copy()
    # GemPy computes in float32; the dense and the custom grid round differently.
    np.testing.assert_allclose(field.evaluate(points).reshape(grid.resolution), on_grid.T, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(points, grid.grid_coordinates)


def test_uck_fields_of_different_groups_evaluate_independently():
    """Each fitted field keeps its own GemPy weights, also when groups are evaluated in turn."""
    grid = RegularGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(6, 5, 4))
    fields = []
    for formations in (["rock3", "rock4"], ["rock1", "rock2"]):
        group, sp, ori = _uck_inputs(grid, formations)
        fields.append(fit_group_universal_cokriging(group=group, extent=grid.extent, group_surface_points_df=sp,
                                                    group_orientations_points_df=ori))
    points = grid.grid_coordinates
    first = [f.evaluate(points) for f in fields]
    again = [f.evaluate(points) for f in reversed(fields)][::-1]
    assert not np.allclose(first[0], first[1])
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)


def _expected_unconformity_lith(grid: RegularGrid) -> np.ndarray:
    """Analytic lithology of unconformity_input() at the grid's cell centers (ids: rock1 = 1 .. rock4 = 4)."""
    x, _, z = (grid.grid_coordinates[:, i].reshape(grid.resolution) for i in range(3))
    lith = np.zeros(grid.resolution, dtype=int)
    lith[z >= 0.20 + (x - 0.5)] = 1
    lith[z >= 0.40 + (x - 0.5)] = 2
    lith[z >= 0.62] = 3
    lith[z >= 0.82] = 4
    return lith


def _away_from_contacts(lith: np.ndarray) -> np.ndarray:
    """Cells whose 6 neighbours all have the same lithology."""
    same = np.ones_like(lith, dtype=bool)
    for axis in range(3):
        differs = np.diff(lith, axis=axis) != 0
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        same[tuple(lo)] &= ~differs
        same[tuple(hi)] &= ~differs
    return same


@pytest.mark.integration
@pytest.mark.slow
def test_geo_inr_on_an_octree_matches_the_analytic_model():
    """GeoINR is stochastic, so it is checked against the analytic solution, not the regular grid."""
    grid = octree_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
        for group in frame.structural_groups:
            group.set_interpolation_method(InterpolationMethod.GEOINR)
            # default is 4000 epochs; 1000 is what the regular-grid integration tests use
            group.configure_interpolation_params(epochs=1000)
        oct_frame = general.compute_structural_model(frame, extract_meshes=False).structural_frame

    oct_frame.check_cell_arrays()
    lith = oct_frame.to_regular(oct_frame.lith_block)
    expected = _expected_unconformity_lith(grid.equivalent_regular_grid())
    interior = _away_from_contacts(expected)
    assert np.mean((lith == expected)[interior]) >= 0.90


def test_fit_padding_matches_the_regular_grid_padding():
    assert general_octree._FIT_PADDING_CELLS == general._SURFACE_PADDING_CELLS
    grid = octree_grid()
    padded = general._build_padded_grid(grid.equivalent_regular_grid(), general._SURFACE_PADDING_CELLS)
    assert general_octree.fit_extent(grid) == padded.extent


# -----------------------------------------------------------------------------
# Refinement
# -----------------------------------------------------------------------------

def test_refinement_follows_a_plane_down_to_max_level():
    grid = octree_grid(base=(2, 2, 2), max_level=4)
    z_plane = 0.37
    cells = refine_cells(grid, lambda p: (p[:, 2] > z_plane).astype(int))

    # The cells tile the extent.
    assert (8 ** (grid.max_level - cells.level.astype(int))).sum() == np.prod(grid.finest_resolution)
    # Every cell the plane passes through is at max level; coarser cells don't touch it.
    mins, maxs = grid.cell_bounds(cells.level, cells.ijk)
    crossed = (mins[:, 2] < z_plane) & (maxs[:, 2] > z_plane)
    assert np.all(cells.level[crossed] == grid.max_level)
    assert crossed.sum() == grid.finest_resolution[0] * grid.finest_resolution[1]
    assert len(cells) < np.prod(grid.finest_resolution)


def test_refinement_without_change_keeps_the_base_blocks():
    grid = octree_grid(base=(3, 2, 1), max_level=3)
    cells = refine_cells(grid, lambda p: np.zeros(len(p), dtype=int))
    expected = base_cells(grid)
    np.testing.assert_array_equal(cells.level, expected.level)
    np.testing.assert_array_equal(cells.ijk, expected.ijk)


def _as_set(cells: OctreeCells) -> set:
    return {(int(level), *map(int, ijk)) for level, ijk in zip(cells.level, cells.ijk)}


def test_coarsening_with_constant_labels_returns_the_base_blocks():
    grid = octree_grid(base=(2, 2, 2), max_level=4)
    refined = refine_cells(grid, lambda p: (p[:, 2] > 0.37).astype(int))
    coarse = coarsen_cells(grid, refined, lambda p: np.zeros(len(p), dtype=int))
    assert _as_set(coarse) == _as_set(base_cells(grid))


def test_coarsening_with_the_refinement_labels_changes_nothing():
    grid = octree_grid(base=(2, 2, 2), max_level=4)
    label = lambda p: (p[:, 2] > 0.37).astype(int)  # noqa: E731
    refined = refine_cells(grid, label)
    assert _as_set(coarsen_cells(grid, refined, label)) == _as_set(refined)


def test_coarsening_keeps_cells_where_the_coarser_labels_change():
    """Refine for two planes, store only one: refinement survives along the stored plane only."""
    grid = octree_grid(base=(2, 2, 2), max_level=4)
    both = lambda p: np.column_stack([p[:, 2] > 0.37, p[:, 0] > 0.61]).astype(int)  # noqa: E731
    refined = refine_cells(grid, both)
    coarse = coarsen_cells(grid, refined, lambda p: (p[:, 2] > 0.37).astype(int))
    assert len(coarse) < len(refined)
    assert _as_set(coarse) == _as_set(refine_cells(grid, lambda p: (p[:, 2] > 0.37).astype(int)))
    assert (8 ** (grid.max_level - coarse.level.astype(int))).sum() == np.prod(grid.finest_resolution)


def test_refinement_continues_from_given_cells():
    grid = octree_grid(base=(1, 1, 1), max_level=3)
    start = OctreeCells(level=np.array([3], dtype=np.uint8), ijk=np.array([[0, 0, 0]], dtype=np.int32))
    # A max-level start cell is kept as it is, whatever its corners say.
    cells = refine_cells(grid, lambda p: (p[:, 0] > 0.01).astype(int), start=start)
    assert len(cells) == 1 and cells.level[0] == 3


def _level_map(grid, cells):
    """Level of every finest voxel, and whether the cells tile the extent exactly once."""
    level_map = np.full(grid.finest_resolution, -1, dtype=int)
    covered = np.zeros(grid.finest_resolution, dtype=int)
    for level, ijk in zip(cells.level, cells.ijk):
        n = 2 ** (grid.max_level - int(level))
        i, j, k = np.asarray(ijk, dtype=int) * n
        level_map[i:i + n, j:j + n, k:k + n] = level
        covered[i:i + n, j:j + n, k:k + n] += 1
    return level_map, bool(np.all(covered == 1))


def _largest_face_jump(level_map):
    return max(int(np.abs(np.diff(level_map, axis=a)).max(initial=0)) for a in range(3))


def test_balancing_grades_the_cells_next_to_a_max_level_plane():
    grid = octree_grid(base=(1, 1, 1), max_level=4)
    # The plane x = 0.52 lies in the max-level cells right of x = 0.5; left of it
    # the corners all agree, so the level-1 block [0, 0.5] stays: a jump of 3 levels.
    cells = refine_cells(grid, lambda p: (p[:, 0] > 0.52).astype(int))
    before, _ = _level_map(grid, cells)
    assert _largest_face_jump(before) > 1
    balanced = balance_2to1(grid, cells)
    after, tiles = _level_map(grid, balanced)
    assert tiles and _largest_face_jump(after) <= 1
    # Only splits: every voxel is at least as fine as before, max-level voxels stay.
    assert np.all(after >= before) and np.all(after[before == grid.max_level] == grid.max_level)
    assert balance_2to1(grid, balanced) is balanced


def test_balancing_leaves_a_balanced_octree_alone():
    grid = octree_grid(base=(2, 2, 2), max_level=3)
    cells = refine_cells(grid, lambda p: np.zeros(len(p), dtype=int))
    assert balance_2to1(grid, cells) is cells


def test_merge_labels_decide_which_siblings_merge_while_corners_keep_their_domains():
    # Two domains split at x = 0.5; constant labels, so only the merge rule matters.
    grid = octree_grid(base=(1, 1, 1), max_level=2)
    domains = lambda level, ijk: (grid.cell_centers(level, ijk)[:, 0] >= 0.5).astype(np.int64)  # noqa: E731
    refined = refine_cells(grid, lambda p: (p[:, 0] > 0.5).astype(int))
    seen = []

    def label(points, point_domains):
        seen.append(point_domains)
        return np.zeros(len(points), dtype=int)

    same = lambda level, ijk: np.zeros(len(ijk), dtype=np.int64)  # noqa: E731
    merged = coarsen_cells(grid, refined, label, cell_domains=domains, merge_labels=same)
    assert len(merged) == 1
    assert {0, 1} <= set(np.concatenate(seen).tolist())  # corners were labelled in both domains
    kept = coarsen_cells(grid, refined, label, cell_domains=domains)  # default: merge within one domain only
    assert len(kept) == 8


# -----------------------------------------------------------------------------
# Full pipeline: octree vs. regular grid at the finest resolution
# -----------------------------------------------------------------------------

def test_octree_result_matches_the_regular_grid(results):
    octree_result, regular_result = results
    oct_frame, reg_frame = octree_result.structural_frame, regular_result.structural_frame

    assert len(oct_frame.cells) < np.prod(oct_frame.grid.finest_resolution)
    oct_frame.check_cell_arrays()
    np.testing.assert_array_equal(oct_frame.to_regular(oct_frame.lith_block), reg_frame.lith_block)
    for oct_group, reg_group in zip(oct_frame.structural_groups, reg_frame.structural_groups):
        np.testing.assert_array_equal(oct_frame.to_regular(oct_group.mask), reg_group.mask)
        assert [e.scalar_value for e in oct_group.structural_elements] == \
               [e.scalar_value for e in reg_group.structural_elements]
        assert [e.id for e in oct_group.structural_elements] == [e.id for e in reg_group.structural_elements]


def test_scalar_fields_at_max_level_cells_equal_the_regular_grid(results):
    octree_result, regular_result = results
    oct_frame, reg_frame = octree_result.structural_frame, regular_result.structural_frame
    # GemPy computes in float32, and its dense and custom grids round differently.
    float32 = oct_frame.structural_groups[0].interpolation_method == InterpolationMethod.UNIVERSAL_COKRIGING
    fine = oct_frame.cells.level == oct_frame.grid.max_level
    i, j, k = oct_frame.cells.ijk[fine].T
    for oct_group, reg_group in zip(oct_frame.structural_groups, reg_frame.structural_groups):
        np.testing.assert_allclose(oct_group.scalar_field[fine], reg_group.scalar_field[i, j, k],
                                   atol=1e-4 if float32 else 1e-12)


def _eroded_rock2_cells(grid: OctreeGrid, cells: OctreeCells) -> np.ndarray:
    """Cells the (eroded) rock2 surface passes through above the unconformity at z = 0.62."""
    mins, maxs = grid.cell_bounds(cells.level, cells.ijk)
    # rock2: z = 0.40 + (x - 0.5); with a 45 degree dip, test against both x bounds.
    z_low, z_high = 0.40 + (mins[:, 0] - 0.5), 0.40 + (maxs[:, 0] - 0.5)
    crosses = (mins[:, 2] < z_high) & (maxs[:, 2] > z_low)
    return crosses & (mins[:, 2] > 0.62)


def test_eroded_surfaces_are_refined_for_meshes_then_coarsened():
    """
    The refinement for surfaces resolves the lower group's rock2 surface even above the
    unconformity, where it is eroded; the stored cells merge that refinement away again,
    since lithology and masks don't change there.
    """
    grid = octree_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
        general_octree._assign_element_ids(frame)
        fields = general_octree._fit_groups(frame)
    mesh_cells = general_octree._refine_for_surfaces(frame, fields)
    stored_cells = compute(grid).structural_frame.cells

    eroded_mesh = _eroded_rock2_cells(grid, mesh_cells)
    assert eroded_mesh.any()
    assert np.all(mesh_cells.level[eroded_mesh] == grid.max_level)

    eroded_stored = _eroded_rock2_cells(grid, stored_cells)
    assert np.any(stored_cells.level[eroded_stored] < grid.max_level)
    assert len(stored_cells) < len(mesh_cells)


def test_mixed_interpolators_per_group_match_the_regular_grid():
    """Each group keeps its own interpolator on the octree, as on the regular grid."""
    def compute_mixed(grid) -> StructuralModelResults:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=grid)
            frame["UpperSeries"].set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)
            frame["LowerSeries"].set_interpolation_method(InterpolationMethod.FINITE_DIFFERENCES)
            return general.compute_structural_model(frame, extract_meshes=False)

    grid = octree_grid()
    oct_frame = compute_mixed(grid).structural_frame
    reg_frame = compute_mixed(grid.equivalent_regular_grid()).structural_frame

    assert [g.interpolation_method for g in oct_frame.structural_groups] == [
        InterpolationMethod.RADIAL_BASIS_FUNCTION, InterpolationMethod.FINITE_DIFFERENCES]
    np.testing.assert_array_equal(oct_frame.to_regular(oct_frame.lith_block), reg_frame.lith_block)
    for oct_group, reg_group in zip(oct_frame.structural_groups, reg_frame.structural_groups):
        np.testing.assert_array_equal(oct_frame.to_regular(oct_group.mask), reg_group.mask)


def test_octree_result_survives_the_runner_round_trip(results):
    octree_result = results[0]
    restored = StructuralModelResults(**json.loads(json.dumps(to_jsonable_python(octree_result))))
    frame = restored.structural_frame
    assert isinstance(frame.grid, OctreeGrid)
    frame.check_cell_arrays()
    np.testing.assert_array_equal(frame.lith_block, octree_result.structural_frame.lith_block)


def test_no_meshes_without_extract_meshes(results):
    oct_frame = results[0].structural_frame  # computed with extract_meshes=False
    assert all(not e.vertices for g in oct_frame.structural_groups for e in g.structural_elements)
    assert all(g.extended_scalar_field is None for g in oct_frame.structural_groups)


# -----------------------------------------------------------------------------
# Not supported yet
# -----------------------------------------------------------------------------

def test_interpolators_without_a_fit_are_refused_on_an_octree(monkeypatch):
    # Every selectable interpolator has a fit now; drop one to check the guard.
    monkeypatch.delitem(general_octree.fit_dispatch, InterpolationMethod.ORDINARY_KRIGING)
    frame = general.build_structural_frame(input_data_elements=unconformity_input(), grid=octree_grid())
    frame.structural_groups[0].set_interpolation_method(InterpolationMethod.ORDINARY_KRIGING)
    with pytest.raises(NotImplementedError, match="UpperSeries"):
        general.compute_structural_model(frame)



def test_lattice_cache_evaluates_each_lattice_point_once():
    """_LatticeCachedField gives the wrapped field's values and evaluates lattice points only the first time."""
    class Counting:
        scalar_values = {"a": 1.0}

        def __init__(self):
            self.points = 0

        def evaluate(self, xyz):
            self.points += len(xyz)
            return xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2]

    grid = octree_grid()
    field = Counting()
    cached = general_octree._LatticeCachedField(field, grid)
    h = grid.finest_spacing
    corners = grid.origin + np.array([[0, 0, 0], [1, 2, 3], [-2, 0, 16], [16, 16, 18]]) * h  # incl. the shell
    off = grid.origin + np.array([[0.5, 0.5, 0.5], [0.25, 0, 0]]) * h
    first = cached.evaluate(np.concatenate([corners, off]))
    np.testing.assert_array_equal(first, field.evaluate(np.concatenate([corners, off])))
    field.points = 0
    again = cached.evaluate(np.concatenate([off, corners[::-1]]))
    np.testing.assert_array_equal(again, np.concatenate([first[4:], first[3::-1]]))
    assert field.points == len(off)  # only the off-lattice points were evaluated again
    assert cached.scalar_values == field.scalar_values
