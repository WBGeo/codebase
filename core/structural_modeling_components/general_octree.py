"""
Structural modeling on an :class:`OctreeGrid`.

The octree counterpart of :func:`general.compute_structural_model`, which
dispatches here when the frame's grid is an OctreeGrid:

  1) Fit each group's scalar field once (``fit_group_*``), or with faults once
     per group and merged fault-domain component.
  2) Refine the octree where the lithology or any group's own surfaces change
     between cell corners. This octree resolves every surface, including the
     parts eroded by younger groups, which surface meshes need. With faults,
     refinement starts from the fault frame's cells, every cell keeps the
     domain of the fault cell it lies in, and points are evaluated with the
     field of their cell's domain.
  3) Extract the UNMASKED, MASKED and EXTENDED surface meshes of every element
     by dual contouring on the refined octree, per merged fault-domain component.
     MASKED keeps a quad only if none of its 4 cells is eroded by a younger group;
     EXTENDED also runs on a max-level shell around the extent, so it crosses the
     model box. COMBINED contours the lithology id itself.
  4) Coarsen back to the cells the stored result needs: merge cells wherever the
     lithology and the age masks don't change, so the result keeps as few cells
     as possible. Only cells of one fault domain are merged.
  5) Evaluate fields, age masks and lithology at the final cell centers and store
     them in the usual attributes, one value per cell (see ``StructuralFrame``).
  6) With faults, the MASKED and EXTENDED_MASKED meshes of every fault: its
     UNMASKED and EXTENDED triangles where the youngest group the fault does not
     affect is absent.

Current scope: interpolators listed in ``fit_dispatch``; all surface mesh types.
The domain components, their fit extents, the age mask and lithology rules are
the pointwise equivalents of ``run_interpolation_with_fault_domains``,
``set_scalar_masks_per_domain`` and ``compute_lithology_block_with_domains``.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from core.object_components import StructuralModelResults
from core.structural_modeling_components.interpolator_functions.finite_difference import (
    fit_group_finite_differences,
)
from core.structural_modeling_components.interpolator_functions.fitted_field import Extent6, FittedField
from core.structural_modeling_components.interpolator_functions.geoinr import fit_group_geo_inr
from core.structural_modeling_components.interpolator_functions.ordinary_kriging import fit_group_ordinary_kriging
from core.structural_modeling_components.interpolator_functions.piecewise_linear import (  # noqa: F401
    fit_group_piecewise_linear,
)
from core.structural_modeling_components.interpolator_functions.radial_basis_function import (
    fit_group_radial_basis_function,
)
from core.structural_modeling_components.interpolator_functions.universal_cokriging import (
    fit_group_universal_cokriging,
)
from core.structural_modeling_components.interpolator_functions.universal_kriging import (
    fit_group_universal_kriging,
)
from core.structural_modeling_components.structural_modeling_utility.dual_contouring import (
    contour_cells,
    dual_contour,
    select_faces,
    padding_shell,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    CellDomains,
    coarsen_cells,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeCells, OctreeGrid
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultMeshType,
    GeoMeshType,
    InterpolationMethod,
    StructuralFrame,
    StructuralGroup,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Per group: fault domain id -> the field of the merged component that contains
# it, or None where the component has no surface points (evaluates to zero).
GroupFields = List[Dict[int, Optional[FittedField]]]

# Interpolators that can be fitted once and evaluated anywhere.
fit_dispatch: Dict[InterpolationMethod, Callable[..., FittedField]] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: fit_group_radial_basis_function,
    InterpolationMethod.FINITE_DIFFERENCES: fit_group_finite_differences,
    # InterpolationMethod.PIECEWISE_LINEAR: fit_group_piecewise_linear,  # excluded pending parameter tuning
    InterpolationMethod.GEOINR: fit_group_geo_inr,
    InterpolationMethod.ORDINARY_KRIGING: fit_group_ordinary_kriging,
    InterpolationMethod.UNIVERSAL_KRIGING: fit_group_universal_kriging,
    InterpolationMethod.UNIVERSAL_COKRIGING: fit_group_universal_cokriging,
}

# Fits cover the extent padded by this many finest cells on every side, the same
# margin the regular-grid pipeline interpolates on (general._SURFACE_PADDING_CELLS).
# Interpolators with a support mesh (FDI, PLI) depend on the fitted extent, so this
# keeps both grid types consistent, and the margin is where EXTENDED meshes go.
_FIT_PADDING_CELLS: int = 2


def compute_structural_model_octree(
        frame: StructuralFrame,
        *,
        extract_meshes: bool = True,
) -> StructuralModelResults:
    """
    Compute a structural model on the frame's OctreeGrid.

    Parameters
    ----------
    frame : StructuralFrame
        Frame with an OctreeGrid, groups, elements and input data, as built by
        ``build_structural_frame``, optionally with a fault frame computed on the
        same grid.
    extract_meshes : bool, default True
        Extract each element's UNMASKED, MASKED, EXTENDED and COMBINED surface
        meshes by dual contouring, and each fault's MASKED and EXTENDED_MASKED
        meshes.

    Returns
    -------
    StructuralModelResults
        A deep copy of the frame with ``cells`` set and, per cell, every group's
        ``scalar_field`` and ``mask`` and the ``lith_block``.

    Raises
    ------
    NotImplementedError
        If a group uses an interpolator that has no fit/evaluate implementation yet.
    ValueError
        If an attached fault frame is not on the same OctreeGrid or has no cells,
        or if the fits of a group's domain components disagree on an element's
        scalar value.
    """
    grid = frame.grid
    if not isinstance(grid, OctreeGrid):
        raise TypeError(f"compute_structural_model_octree needs an OctreeGrid, got {type(grid).__name__}.")
    ff = frame.fault_frame
    if ff is not None and (ff.grid != grid or ff.cells is None or ff.domain_map is None):
        raise ValueError(
            "The fault model must be computed on the same OctreeGrid as the structural model. "
            "Connect the same grid to both, or recompute the fault model on this grid."
        )
    unsupported = [f"'{g.name}' ({g.interpolation_method})" for g in frame.structural_groups
                   if g.interpolation_method not in fit_dispatch]
    if unsupported:
        raise NotImplementedError(
            "On an OctreeGrid only these interpolators are available so far: "
            f"{', '.join(m.value for m in fit_dispatch)}. Groups using others: {', '.join(unsupported)}."
        )
    if not frame.structural_groups:
        raise ValueError("The structural frame contains no groups.")

    _assign_element_ids(frame)

    logger.info("Fitting scalar fields ...")
    fields = _fit_groups(frame)

    logger.info("Refining the octree ...")
    mesh_cells = _refine_for_surfaces(frame, fields)

    if extract_meshes:
        logger.info("Extracting surface meshes by dual contouring ...")
        _extract_element_meshes(frame, fields, mesh_cells)

    logger.info("Coarsening the octree for the stored result ...")
    cells = _coarsen_for_storage(frame, fields, mesh_cells)
    logger.info(
        "Octree cells: %d refined for surfaces, %d stored (equivalent regular grid: %d).",
        len(mesh_cells), len(cells), int(np.prod(grid.finest_resolution)),
    )

    logger.info("Evaluating the model at cell centers ...")
    centers = cells.centers(grid)
    domains = _point_domains(frame, centers)
    values = _evaluate_groups(frame, fields, centers, domains)
    lith, masks = _lithology_and_masks(frame, values, domains)

    frame.cells = cells
    for group_idx, group in enumerate(frame.structural_groups):
        group.scalar_field = values[group_idx]
        group.mask = masks[group_idx]
        group.extended_scalar_field = None
    frame.lith_block = lith
    frame.check_cell_arrays()

    if extract_meshes and frame.fault_frame is not None:
        _extract_fault_masked_meshes(frame, fields, masks)

    return StructuralModelResults(structural_frame=copy.deepcopy(frame))


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------


def _assign_element_ids(frame: StructuralFrame) -> None:
    """Global element ids, oldest = 1, as ``compute_lithology_block_with_domains`` assigns them."""
    current_id = 1
    for group in reversed(frame.structural_groups):
        for elem in reversed(group.structural_elements):
            if elem.id is None:
                elem.set_id(current_id)
                current_id += 1


def _fit_groups(frame: StructuralFrame) -> GroupFields:
    """
    Fit every group once per merged domain component and store the element scalar values.

    Without faults there is one component, domain 0, fitted over ``fit_extent``.
    With faults the components come from ``effective_domain_components_for_group``,
    each input point belongs to the domain of the fault cell it lies in, and each
    component is fitted over its own padded bounding box, as on the regular grid.
    Components without surface points get ``None``, which evaluates to zero, like
    the zero field a regular-grid component keeps when it is skipped.
    """
    domain_ids = _domain_ids(frame)
    sp_domains = _point_domains(frame, frame.surface_points[["X", "Y", "Z"]].to_numpy(dtype=np.float64))
    ori_domains = (
        _point_domains(frame, frame.orientations[["X", "Y", "Z"]].to_numpy(dtype=np.float64))
        if frame.orientations is not None
        else None
    )

    fields: GroupFields = []
    for group_idx, group in enumerate(frame.structural_groups):
        components = _group_components(frame, group_idx)
        formations = [e.name for e in group.structural_elements]
        by_domain: Dict[int, Optional[FittedField]] = {}
        scalar_values: Dict[str, float] = {}
        for comp_ids in components:
            in_comp = sorted(comp_ids)
            sp = frame.surface_points[
                np.isin(sp_domains, in_comp) & frame.surface_points["formation"].isin(formations).to_numpy()
            ]
            ori = (
                frame.orientations[
                    np.isin(ori_domains, in_comp) & frame.orientations["formation"].isin(formations).to_numpy()
                ]
                if frame.orientations is not None
                else None
            )
            if ori is not None and ori.empty:
                ori = None
            field: Optional[FittedField] = None
            if not sp.empty:
                field = fit_dispatch[group.interpolation_method](
                    group=group,
                    extent=_component_extent(frame, in_comp, domain_ids),
                    group_surface_points_df=sp,
                    group_orientations_points_df=ori,
                )
                _collect_scalar_values(group, field, scalar_values)
            for d in in_comp:
                by_domain[d] = field

        for elem in group.structural_elements:
            if elem.name in scalar_values:
                elem.set_scalar_value(scalar_values[elem.name])
        fields.append(by_domain)
    return fields


def _group_components(frame: StructuralFrame, group_idx: int) -> List[Set[int]]:
    """
    The merged fault-domain components of a group: domains that faults not
    affecting the group don't separate. One component, all domains, without faults.
    """
    domain_ids = _domain_ids(frame)
    ff = frame.fault_frame
    if ff is None:
        return [set(map(int, domain_ids))]
    from core.structural_modeling_components.general import effective_domain_components_for_group

    return effective_domain_components_for_group(
        domain_ids=domain_ids,
        faults=list(ff.fault_elements),
        fault_activity=frame.fault_activity,
        group_idx=group_idx,
    )


def _component_of_domains(frame: StructuralFrame, group_idx: int, domains: IntArray) -> IntArray:
    """For each domain id, the smallest domain id of the group's component that contains it."""
    label = {d: min(comp) for comp in _group_components(frame, group_idx) for d in comp}
    return np.array([label[int(d)] for d in domains], dtype=np.int64)


def _collect_scalar_values(group: StructuralGroup, field: FittedField, scalar_values: Dict[str, float]) -> None:
    """
    Add one component fit's element scalar values, raising if another component's differ.

    The interpolators map a group's element surfaces to fixed values (1..n), so
    all components agree. The regular grid keeps the last component's values;
    here a disagreement would make the lithology depend on the component order.
    """
    for elem in group.structural_elements:
        if elem.name not in field.scalar_values:
            raise ValueError(
                f"Interpolator did not return a scalar value for element '{elem.name}' in group '{group.name}'."
            )
        value = float(field.scalar_values[elem.name])
        if elem.name in scalar_values and not np.isclose(scalar_values[elem.name], value):
            raise ValueError(
                f"The fault domain components of group '{group.name}' give element '{elem.name}' "
                f"different scalar values ({scalar_values[elem.name]} and {value})."
            )
        scalar_values[elem.name] = value


def fit_extent(grid: OctreeGrid) -> Extent6:
    """The grid's extent padded by ``_FIT_PADDING_CELLS`` finest cells on every side."""
    pad = _FIT_PADDING_CELLS * grid.finest_spacing
    lower = np.array(grid.extent[0::2]) - pad
    upper = np.array(grid.extent[1::2]) + pad
    return tuple(float(v) for pair in zip(lower, upper) for v in pair)


def _component_extent(frame: StructuralFrame, comp_ids: List[int], domain_ids: IntArray) -> Extent6:
    """
    The extent a domain component is fitted over.

    With a single domain, ``fit_extent``. Otherwise the bounding box of the
    component's finest cells, padded by ``_FIT_PADDING_CELLS`` finest cells on
    the sides where it reaches the extent: the regular grid crops its edge-padded
    domain map the same way (``build_subgrid_from_bbox``).
    """
    grid = frame.grid
    if len(domain_ids) <= 1:
        return fit_extent(grid)
    ff = frame.fault_frame
    in_comp = np.isin(np.asarray(ff.domain_map), comp_ids)
    ijk = ff.cells.ijk[in_comp].astype(np.int64)
    size = 2 ** (grid.max_level - ff.cells.level[in_comp].astype(np.int64))[:, None]
    lower = (ijk * size).min(axis=0)
    upper = ((ijk + 1) * size).max(axis=0)
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)
    lower = np.where(lower == 0, -_FIT_PADDING_CELLS, lower)
    upper = np.where(upper == n_finest, n_finest + _FIT_PADDING_CELLS, upper)
    lo = grid.origin + lower * grid.finest_spacing
    hi = grid.origin + upper * grid.finest_spacing
    return tuple(float(v) for pair in zip(lo, hi) for v in pair)


def _domain_ids(frame: StructuralFrame) -> IntArray:
    """The fault domain ids, ``[0]`` without a fault frame."""
    if frame.fault_frame is None:
        return np.zeros(1, dtype=np.int64)
    return np.unique(np.asarray(frame.fault_frame.domain_map, dtype=np.int64))


def _point_domains(frame: StructuralFrame, points: FloatArray) -> IntArray:
    """
    Fault domain of each point: that of the fault cell containing it, 0 without faults.

    Points outside the extent take the nearest boundary cell's domain, like the
    edge-padded domain map of the regular grid.
    """
    ff = frame.fault_frame
    if ff is None:
        return np.zeros(points.shape[0], dtype=np.int64)
    index = ff.cells.locate(ff.grid, points)
    return np.asarray(ff.domain_map, dtype=np.int64)[index]


def _cell_domains(frame: StructuralFrame) -> Optional[CellDomains]:
    """Domain of cells: that of the fault cell their center lies in; None without faults."""
    if frame.fault_frame is None:
        return None
    grid = frame.grid
    return lambda level, ijk: _point_domains(frame, grid.cell_centers(level, ijk))


def _evaluate_groups(
        frame: StructuralFrame, fields: GroupFields, points: FloatArray, domains: IntArray
) -> FloatArray:
    """Every group's field at the points, each with its domain's field, shape ``(n_groups, M)``."""
    values = np.zeros((len(frame.structural_groups), points.shape[0]), dtype=np.float64)
    for group_idx, by_domain in enumerate(fields):
        values[group_idx] = _group_values(by_domain, points, domains)
    return values


def _group_values(by_domain: Dict[int, Optional[FittedField]], points: FloatArray, domains: IntArray) -> FloatArray:
    """One group's field at the points, each with its domain's field; zero where a domain has none."""
    values = np.zeros(points.shape[0], dtype=np.float64)
    for d in np.unique(domains):
        field = by_domain.get(int(d))
        if field is not None:
            at = domains == d
            values[at] = field.evaluate(points[at])
    return values


def _extract_element_meshes(frame: StructuralFrame, fields: GroupFields, cells: OctreeCells) -> None:
    """
    Every element's UNMASKED, MASKED and EXTENDED meshes, its group's field at the
    element's isovalue, and its COMBINED mesh.

    Dual contouring runs per merged domain component of the group, each with its
    own field, so the surfaces end half a cell before the faults that affect the
    group and run through the others. (The regular grid contours the field
    assembled from all domains instead, across the faults.) UNMASKED uses
    no masks. MASKED keeps a quad only if none of its 4 cells is eroded, i.e.
    lies where the age mask of any younger group is True at the cell center (the
    per-cell form of the erosion mask in ``general.extract_meshes``). EXTENDED
    uses no masks and adds a max-level shell of ``_FIT_PADDING_CELLS`` finest
    cells around the extent, whose cells take the domain of the nearest
    boundary cell (the regular grid's edge-padded domain map).

    COMBINED is the boundary of the region where the lithology id is at least
    the element's id: the lithology id contoured at ``id - 0.1``, as the regular
    grid does with marching cubes on the lithology block. The id already holds
    the faults' offsets and the erosion, so all cells form one group. Being
    piecewise constant, the id has no gradient, and each vertex is the centroid
    of its cell's crossings.
    """
    grid = frame.grid
    bounds = fit_extent(grid)
    dc_cells = contour_cells(grid, cells)
    centers = dc_cells.centers(grid)
    domains = _point_domains(frame, centers)
    _, masks = _lithology_and_masks(frame, _evaluate_groups(frame, fields, centers, domains), domains)
    shell = padding_shell(grid, _FIT_PADDING_CELLS)
    padded_ijk = np.concatenate([dc_cells.ijk.astype(np.int64), shell])
    padded_domains = np.concatenate([domains, _point_domains(frame, grid.cell_centers(grid.max_level, shell))])
    for group_idx, group in enumerate(frame.structural_groups):
        elements = [e for e in group.structural_elements if e.get_scalar_value() is not None]
        by_domain = fields[group_idx]
        if not elements or all(field is None for field in by_domain.values()):
            continue
        isovalues = [float(e.get_scalar_value()) for e in elements]
        # DC groups are the components, labelled by their smallest domain id, whose
        # field is the component's field.
        components = _component_of_domains(frame, group_idx, domains)
        values = lambda p, c: _group_values(by_domain, p, c)  # noqa: E731
        unmasked = dual_contour(grid, dc_cells.ijk, components, values, isovalues, field_bounds=bounds)
        not_eroded = ~masks[:group_idx].any(axis=0)
        masked = dual_contour(grid, dc_cells.ijk, components, values, isovalues, cell_kept=not_eroded,
                              field_bounds=bounds)
        extended = dual_contour(grid, padded_ijk, _component_of_domains(frame, group_idx, padded_domains),
                                values, isovalues, padding=_FIT_PADDING_CELLS, field_bounds=bounds)
        for elem, mesh, masked_mesh, extended_mesh in zip(elements, unmasked, masked, extended):
            if mesh.n_missing_quads:
                logger.warning("Surface of '%s' has %d gaps where a cell next to it is coarser than max level.",
                               elem.name, mesh.n_missing_quads)
            elem.set_mesh(GeoMeshType.UNMASKED, mesh.vertices, mesh.faces.astype(np.int32))
            elem.set_mesh(GeoMeshType.MASKED, masked_mesh.vertices, masked_mesh.faces.astype(np.int32))
            elem.set_mesh(GeoMeshType.EXTENDED, extended_mesh.vertices, extended_mesh.faces.astype(np.int32))

    def lithology(points: FloatArray, _groups: IntArray) -> FloatArray:
        point_domains = _point_domains(frame, points)
        lith, _ = _lithology_and_masks(frame, _evaluate_groups(frame, fields, points, point_domains), point_domains)
        return lith.astype(np.float64)

    elements = [e for g in frame.structural_groups for e in g.structural_elements if e.id is not None]
    combined = dual_contour(grid, dc_cells.ijk, np.zeros(len(dc_cells), dtype=np.int64), lithology,
                            [e.id - 0.1 for e in elements], field_bounds=bounds)
    for elem, mesh in zip(elements, combined):
        elem.set_mesh(GeoMeshType.COMBINED, mesh.vertices, mesh.faces.astype(np.int32))


def _extract_fault_masked_meshes(frame: StructuralFrame, fields: GroupFields, stored_masks: BoolArray) -> None:
    """
    Every fault's MASKED and EXTENDED_MASKED meshes: its UNMASKED and EXTENDED
    triangles outside the youngest group it does not affect.

    As in ``general.extract_meshes``: a mesh is kept where that group's age mask
    is False, and whole if there is no such group. MASKED is also kept whole if
    the group's mask covers the whole model (checked on the stored cells);
    EXTENDED_MASKED is not, and is then empty. A triangle is kept if the
    max-level cells of all 3 of its vertices pass, with the mask evaluated at
    those cells' centers (in the padding shell for EXTENDED vertices).
    """
    activity = frame.fault_activity_verbose or {}
    for fault in frame.fault_frame.fault_elements:
        group_idx = activity[fault.name]["youngest_group_index"] - 1 if fault.name in activity else -1
        for source, target, keep_whole in [
            (FaultMeshType.UNMASKED, FaultMeshType.MASKED, group_idx < 0 or stored_masks[group_idx].all()),
            (FaultMeshType.EXTENDED, FaultMeshType.EXTENDED_MASKED, group_idx < 0),
        ]:
            vertices = fault.vertices.get(source)
            faces = fault.edges.get(source)
            if vertices is None or faces is None:
                continue
            vertices = np.asarray(vertices, dtype=np.float64)
            faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            if keep_whole:
                kept_vertex = np.ones(vertices.shape[0], dtype=bool)
            else:
                kept_vertex = ~_age_mask_at_vertex_cells(frame, fields, vertices)[group_idx]
            kept_vertices, kept_faces, _ = select_faces(vertices, faces, kept_vertex[faces].all(axis=1))
            fault.set_mesh(target, kept_vertices, kept_faces.astype(np.int32))


def _age_mask_at_vertex_cells(frame: StructuralFrame, fields: GroupFields, vertices: FloatArray) -> BoolArray:
    """
    Every group's age mask at the centers of the max-level cells containing the
    vertices, shape ``(n_groups, V)``. Vertices in the padding shell take the
    shell cell they lie in; points on a shared face go to the upper cell.
    """
    grid = frame.grid
    ijk = np.floor((vertices - grid.origin) / grid.finest_spacing).astype(np.int64)
    n_finest = np.array(grid.finest_resolution, dtype=np.int64)
    ijk = np.clip(ijk, -_FIT_PADDING_CELLS, n_finest - 1 + _FIT_PADDING_CELLS)
    centers = grid.cell_centers(grid.max_level, ijk)
    domains = _point_domains(frame, centers)
    _, masks = _lithology_and_masks(frame, _evaluate_groups(frame, fields, centers, domains), domains)
    return masks


def _lithology_and_masks(
        frame: StructuralFrame, values: FloatArray, domains: IntArray
) -> Tuple[IntArray, BoolArray]:
    """
    Age masks and lithology ids from group values and point domains, pointwise.

    Same rules as the regular grid: a group's age mask is ``value >= value of its
    oldest element`` (all True for the oldest group), and with faults it is kept
    only in the domains ``domain_is_active_for_group`` allows; inside a group, each
    point takes the youngest element whose value it reaches; younger groups
    overwrite older ones where their mask is True; 0 is basement.

    Returns:
        ``lith`` of shape ``(M,)`` and ``masks`` of shape ``(n_groups, M)``.
    """
    groups = frame.structural_groups
    n_points = values.shape[1]
    masks = np.ones((len(groups), n_points), dtype=bool)
    for group_idx, group in enumerate(groups[:-1]):
        if not group.structural_elements:
            raise ValueError(f"Group '{group.name}' has no structural elements.")
        oldest = group.structural_elements[-1]
        sval = oldest.get_scalar_value()
        if sval is None:
            raise ValueError(f"Oldest element '{oldest.name}' in group '{group.name}' has no scalar value.")
        masks[group_idx] = values[group_idx] >= sval
    if frame.fault_frame is not None:
        from core.structural_modeling_components.general import domain_is_active_for_group

        for group_idx in range(len(groups)):
            active = [d for d in np.unique(domains)
                      if domain_is_active_for_group(frame.fault_frame, frame.fault_activity, group_idx, int(d))]
            masks[group_idx] &= np.isin(domains, active)

    lith = np.zeros(n_points, dtype=np.int64)
    for group_idx in range(len(groups) - 1, -1, -1):  # oldest -> youngest
        group_block = np.zeros(n_points, dtype=np.int64)
        for elem in groups[group_idx].structural_elements:  # youngest -> oldest
            sval = elem.get_scalar_value()
            if sval is None:
                continue
            group_block[(values[group_idx] >= sval) & (group_block == 0)] = elem.id
        group_block = np.where(masks[group_idx], group_block, 0)
        lith = np.where(group_block > 0, group_block, lith)
    return lith, masks


def _refine_for_surfaces(frame: StructuralFrame, fields: GroupFields) -> OctreeCells:
    """
    Refine until every group's surfaces and the lithology are at max level.

    Starts from the base blocks, or from the fault cells when faults are attached.
    """
    ff = frame.fault_frame
    return refine_cells(
        frame.grid,
        _labeler(frame, fields, _surface_labels),
        start=None if ff is None else ff.cells,
        cell_domains=_cell_domains(frame),
    )


def _coarsen_for_storage(frame: StructuralFrame, fields: GroupFields, cells: OctreeCells) -> OctreeCells:
    """Merge cells of one domain back wherever the lithology and the age masks don't change."""
    return coarsen_cells(
        frame.grid, cells, _labeler(frame, fields, _storage_labels), cell_domains=_cell_domains(frame)
    )


def _labeler(
        frame: StructuralFrame,
        fields: GroupFields,
        labels: Callable[[StructuralFrame, GroupFields, FloatArray, IntArray], IntArray],
) -> Callable[..., IntArray]:
    """``labels`` as the labeler refinement expects: taking point domains when faults are attached."""
    if frame.fault_frame is None:
        return lambda points: labels(frame, fields, points, np.zeros(points.shape[0], dtype=np.int64))
    return lambda points, domains: labels(frame, fields, points, domains)


def _surface_labels(
        frame: StructuralFrame, fields: GroupFields, points: FloatArray, domains: IntArray
) -> IntArray:
    """
    Labels for refining toward the surfaces.

    Column 0 is the lithology id. Then one column per group: which of the group's
    element surfaces the point lies above. The group columns keep every group's
    own surfaces at max level even where a younger group erodes them, which the
    unmasked surface meshes need.
    """
    values = _evaluate_groups(frame, fields, points, domains)
    lith, _ = _lithology_and_masks(frame, values, domains)
    columns = [lith]
    for group_idx, group in enumerate(frame.structural_groups):
        svals = sorted(e.get_scalar_value() for e in group.structural_elements if e.get_scalar_value() is not None)
        columns.append(np.searchsorted(np.asarray(svals, dtype=np.float64), values[group_idx], side="right"))
    return np.column_stack(columns)


def _storage_labels(
        frame: StructuralFrame, fields: GroupFields, points: FloatArray, domains: IntArray
) -> IntArray:
    """Labels the stored result must resolve: the lithology id and every group's age mask."""
    values = _evaluate_groups(frame, fields, points, domains)
    lith, masks = _lithology_and_masks(frame, values, domains)
    return np.column_stack([lith, masks.T.astype(np.int64)])
