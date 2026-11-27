"""
Fault modeling on an :class:`OctreeGrid`.

The octree counterpart of :func:`general_faults.compute_fault_domains`, which
dispatches here when the fault frame's grid is an OctreeGrid:

  1) Fit each fault's GemPy field once, over the padded extent
     (``fit_fault_universal_cokriging``; raw field, GemPy's own isovalue).
  2) Refine the octree wherever a corner lies on the other side of any fault, so
     every fault surface ends at max level across the whole extent. These are the
     stored cells: the domain id carries the same information as the sides, so
     there is nothing to coarsen. With ``grid.cell_balance`` they are 2:1
     balanced afterwards.
  3) Evaluate the faults at the cell centers; a cell's side of each fault, and
     with it its domain, is decided by its center, like a regular voxel.
  4) The cross-cut check, then domain ids, domain pairs and separated domains,
     with the regular path's own rules. Cross-cutting faults get no domains (the
     result is flagged ``cross_cutting`` and keeps its fields and meshes). The isovalue bands are tested at finest-voxel
     centers, like on the regular grid: max-level cells are voxels, coarser cells
     the band can reach are expanded into their voxels for the test only. The
     band width uses a gradient median weighted by cell volume, from central
     differences of the fitted fields.

  5) Extract every fault's EXTENDED mesh by dual contouring on the fault cells
     plus a max-level shell of ``_FIT_PADDING_CELLS`` finest cells around the
     extent (the regular path's padded grid), so it crosses the model box, and
     take its UNMASKED mesh from it: the quads whose cells all lie in the extent.

The structural model later refines these cells further; a cell coarser than max
level lies in a single domain, so its children inherit the domain id.

Current scope: of the fault meshes UNMASKED and EXTENDED (``extended_scalar_field``
stays None); MASKED and EXTENDED_MASKED come from the structural step.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from core.object_components import FaultModelResults
from core.structural_modeling_components.general_faults import (
    domain_pairs_from_band,
    raise_on_band_overlap,
    separated_domains,
    split_into_domains,
)
from core.structural_modeling_components.general_octree import _FIT_PADDING_CELLS, fit_extent
from core.structural_modeling_components.interpolator_functions.universal_cokriging import (
    GemPyFittedField,
    fit_fault_universal_cokriging,
)
from core.structural_modeling_components.structural_modeling_utility.dual_contouring import (
    contour_cells,
    dual_contour,
    padding_shell,
    restrict_to_kept_cells,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    balance_2to1,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import (
    CORNER_OFFSETS,
    CellBalance,
    OctreeCells,
    OctreeGrid,
)
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultElement,
    FaultFrame,
    FaultMeshType,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# Half-width of the isovalue bands in finest cells, as on the regular path (voxels=1.0).
_BAND_HALF_WIDTH_CELLS: float = 1.0


def compute_fault_domains_octree(fault_frame: FaultFrame) -> FaultModelResults:
    """
    Interpolate all faults on the frame's OctreeGrid and build the fault domains.

    Parameters
    ----------
    fault_frame : FaultFrame
        Frame with an OctreeGrid, fault elements and fault input data, as built by
        ``build_fault_frame``.

    Returns
    -------
    FaultModelResults
        A deep copy of the frame with ``cells`` set and, per cell, each fault's
        ``scalar_field`` and ``mask``; per fault ``scalar_value`` and the
        UNMASKED and EXTENDED meshes. If no two faults cross-cut, also
        ``domain_map``, ``domain_masks`` and per fault ``domain_pairs`` and
        ``separated_domains``; otherwise these stay at their None/empty
        defaults and ``cross_cutting`` is True, as on the regular path: the
        result can be inspected but not used for structural modeling.

    Raises
    ------
    ValueError
        If input data is missing.
    RuntimeError
        If a fault doesn't split the model or produces no domain pairs (only
        without cross-cutting).
    """
    grid = fault_frame.grid
    if not isinstance(grid, OctreeGrid):
        raise TypeError(f"compute_fault_domains_octree needs an OctreeGrid, got {type(grid).__name__}.")
    if fault_frame.fault_surface_points_df is None:
        raise ValueError("Fault surface points must be set.")

    youngest_first = list(reversed(fault_frame.fault_elements))

    logger.info("Fitting fault fields ...")
    fields = _fit_faults(fault_frame, youngest_first)

    logger.info("Refining the octree along the faults ...")
    cells = refine_cells(grid, lambda points: _fault_sides(youngest_first, fields, points))
    if grid.cell_balance == CellBalance.FACE_2TO1:
        cells = balance_2to1(grid, cells)
    logger.info("Fault octree cells: %d (equivalent regular grid: %d).",
                len(cells), int(np.prod(grid.finest_resolution)))

    centers = cells.centers(grid)
    for fault in youngest_first:
        values = fields[fault.name].evaluate(centers)
        fault.set_scalar_field(values)
        fault.set_domain_mask(values > fault.scalar_value)
        fault.extended_scalar_field = None

    domain_map = split_into_domains([fault.get_domain_mask() for fault in youngest_first])
    fault_frame.cells = cells

    faults = fault_frame.fault_elements
    tolerances = [_band_tolerance(grid, cells, fields[f.name], centers) for f in faults]
    voxel_domains, voxel_values = _band_voxels(grid, cells, domain_map, faults, fields, tolerances)

    # Cross-cut check ahead of domaining, as on the regular path: a cross-cutting
    # fault set gets no domains, but keeps its fields and meshes for inspection.
    try:
        raise_on_band_overlap(
            [(f.name, values, float(f.scalar_value)) for f, values in zip(faults, voxel_values)],
            tolerances,
        )
        cross_cutting = False
    except ValueError:
        cross_cutting = True

    if not cross_cutting:
        fault_frame.domain_map = domain_map
        fault_frame.domain_masks = {int(uid): (domain_map == uid) for uid in np.unique(domain_map)}
        for fault, tol, values in zip(faults, tolerances, voxel_values):
            pairs = domain_pairs_from_band(voxel_domains, values, float(fault.scalar_value), tol)
            if not pairs:
                raise RuntimeError(f"Fault '{fault.name}' produced no valid domain_pairs after remap.")
            fault.set_domain_pairs(frozenset((int(a), int(b)) for a, b in pairs))
            fault.set_separated_domains(separated_domains(domain_map, fault.get_domain_mask(), fault.name))

    logger.info("Extracting fault surfaces by dual contouring ...")
    dc_cells = contour_cells(grid, cells)
    padded_ijk = np.concatenate([dc_cells.ijk.astype(np.int64), padding_shell(grid, _FIT_PADDING_CELLS)])
    bounds = fit_extent(grid)
    inside = np.arange(len(padded_ijk)) < len(dc_cells)  # the cells in the extent come first
    for fault in faults:
        field = fields[fault.name]
        values = lambda p, _groups: field.evaluate(p)  # noqa: E731
        # Refined with field > isovalue (see _fault_sides), so contour with the same side rule.
        # Contoured once, for EXTENDED; UNMASKED is EXTENDED without the quads that need a shell cell.
        extended = dual_contour(grid, padded_ijk, np.zeros(len(padded_ijk), dtype=np.int64), values,
                                [float(fault.scalar_value)], strictly_above=True, padding=_FIT_PADDING_CELLS,
                                field_bounds=bounds)[0]
        mesh = restrict_to_kept_cells(extended, inside)
        if mesh.n_missing_quads:
            logger.warning("Surface of fault '%s' has %d gaps where a cell next to it is coarser than max level.",
                           fault.name, mesh.n_missing_quads)
        fault.set_mesh(FaultMeshType.UNMASKED, mesh.vertices, mesh.faces)
        fault.set_mesh(FaultMeshType.EXTENDED, extended.vertices, extended.faces)

    fault_frame.check_cell_arrays()
    return FaultModelResults(fault_frame=copy.deepcopy(fault_frame), cross_cutting=cross_cutting)


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------


def _fit_faults(fault_frame: FaultFrame, faults: List[FaultElement]) -> Dict[str, GemPyFittedField]:
    """Fit every fault over the padded extent and store its isovalue."""
    extent = fit_extent(fault_frame.grid)
    fields: Dict[str, GemPyFittedField] = {}
    for fault in faults:
        points = fault_frame.get_surface_points_for_element(fault.name)
        orientations = fault_frame.get_orientations_for_element(fault.name)
        if points.empty:
            raise ValueError(f"No surface points provided for {fault.name}")
        if orientations is None or orientations.empty:
            raise ValueError(f"No orientations provided for {fault.name}")
        field = fit_fault_universal_cokriging(
            fault_name=fault.name,
            extent=extent,
            fault_surface_points_df=points,
            fault_orientations_points_df=orientations,
        )
        fault.set_scalar_value(float(field.scalar_values[fault.name]))
        fields[fault.name] = field
    return fields


def _fault_sides(faults: List[FaultElement], fields: Dict[str, GemPyFittedField], points: FloatArray) -> np.ndarray:
    """Refinement labels: one column per fault, 1 on its positive side (``field > isovalue``)."""
    return np.column_stack([
        (fields[f.name].evaluate(points) > f.scalar_value).astype(np.int64) for f in faults
    ])


def _band_tolerance(grid: OctreeGrid, cells: OctreeCells, field: GemPyFittedField, centers: FloatArray) -> float:
    """
    Isovalue band half-width in scalar units, as on the regular path.

    ``thickness * median|∇φ|`` with ``thickness`` one finest cell (the smallest
    spacing). ``|∇φ|`` comes from central differences of the fitted field at the
    cell centers with the finest spacing as step, which is what ``np.gradient``
    computes at a regular voxel; its median is weighted by the number of finest
    voxels each cell covers.
    """
    spacing = grid.finest_spacing
    thickness = _BAND_HALF_WIDTH_CELLS * float(np.min(spacing))

    n = centers.shape[0]
    offsets = np.concatenate([np.diag(spacing), -np.diag(spacing)])  # +x, +y, +z, -x, -y, -z
    shifted = (centers[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    values = field.evaluate(shifted).reshape(6, n)
    gradient = (values[:3] - values[3:]) / (2.0 * spacing[:, None])
    grad_mag = np.sqrt(np.sum(gradient * gradient, axis=0))

    weights = 8.0 ** (grid.max_level - cells.level.astype(np.float64))
    median = _weighted_median(grad_mag, weights)
    return thickness * max(median, 1e-12)


def _band_voxels(
        grid: OctreeGrid,
        cells: OctreeCells,
        domain_map: np.ndarray,
        faults: List[FaultElement],
        fields: Dict[str, GemPyFittedField],
        tolerances: List[float],
) -> Tuple[np.ndarray, List[FloatArray]]:
    """
    The finest voxels the isovalue bands can reach, with their domain and every fault's value.

    The regular path tests the bands at voxel centers. Max-level cells are voxels;
    a coarser cell is expanded into its finest voxels (not stored, only for this
    test) when its corner values come within ``2 * tol`` of a fault's isovalue, so
    a band that ends inside a coarse cell is still seen. A coarse cell lies in one
    domain, which its voxels take.

    Returns:
        ``domains`` of shape ``(V,)`` and, per fault, its values at the ``V`` voxel centers.
    """
    fine = cells.level == grid.max_level
    domains = [domain_map[fine]]
    values: List[List[FloatArray]] = [[f.scalar_field[fine]] for f in faults]

    coarse = np.flatnonzero(~fine)
    for level in np.unique(cells.level[coarse]):
        members = coarse[cells.level[coarse] == level]
        ijk = cells.ijk[members].astype(np.int64)
        corners = grid.origin + (ijk[:, None, :] + CORNER_OFFSETS[None, :, :]) * grid.cell_size(int(level))
        near = np.zeros(len(members), dtype=bool)
        for fault, tol in zip(faults, tolerances):
            corner_values = fields[fault.name].evaluate(corners.reshape(-1, 3)).reshape(-1, 8)
            near |= (corner_values.min(axis=1) <= fault.scalar_value + 2 * tol) & \
                    (corner_values.max(axis=1) >= fault.scalar_value - 2 * tol)
        if not near.any():
            continue

        n = 2 ** (grid.max_level - int(level))
        offsets = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        voxel_ijk = (ijk[near][:, None, :] * n + offsets[None, :, :]).reshape(-1, 3)
        voxel_centers = grid.origin + (voxel_ijk + 0.5) * grid.finest_spacing
        domains.append(np.repeat(domain_map[members[near]], n ** 3))
        for fault, fault_values in zip(faults, values):
            fault_values.append(fields[fault.name].evaluate(voxel_centers))

    return np.concatenate(domains), [np.concatenate(v) for v in values]


def _weighted_median(values: FloatArray, weights: FloatArray) -> float:
    """Median of ``values`` where each value counts ``weights`` times (NaNs ignored)."""
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    values, weights = values[finite], weights[finite]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
