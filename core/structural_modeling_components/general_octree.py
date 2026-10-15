"""
Structural modeling on an :class:`OctreeGrid`.

The octree counterpart of :func:`general.compute_structural_model`, which
dispatches here when the frame's grid is an OctreeGrid:

  1) Fit each group's scalar field once (``fit_group_*``).
  2) Refine the octree where the lithology or any group's own surfaces change
     between cell corners. This octree resolves every surface, including the
     parts eroded by younger groups, which surface meshes need.
  3) (Surface meshes will be extracted here, on the refined octree.)
  4) Coarsen back to the cells the stored result needs: merge cells wherever the
     lithology and the age masks don't change, so the result keeps as few cells
     as possible.
  5) Evaluate fields, age masks and lithology at the final cell centers and store
     them in the usual attributes, one value per cell (see ``StructuralFrame``).

Current scope: models without faults, RBF only, no surface meshes yet. The age
mask and lithology rules are the pointwise equivalents of
``set_scalar_masks_per_domain`` and ``compute_lithology_block_with_domains`` for
a single fault domain.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from core.object_components import StructuralModelResults
from core.structural_modeling_components.interpolator_functions.fitted_field import FittedField
from core.structural_modeling_components.interpolator_functions.radial_basis_function import (
    fit_group_radial_basis_function,
)
from core.structural_modeling_components.structural_modeling_utility.octree_refinement import (
    coarsen_cells,
    refine_cells,
)
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeCells, OctreeGrid
from core.structural_modeling_components.structural_objects.structural_objects import (
    InterpolationMethod,
    StructuralFrame,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Interpolators that can be fitted once and evaluated anywhere. Others follow
# behind the same interface.
fit_dispatch: Dict[InterpolationMethod, Callable[..., FittedField]] = {
    InterpolationMethod.RADIAL_BASIS_FUNCTION: fit_group_radial_basis_function,
}


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
        ``build_structural_frame``.
    extract_meshes : bool, default True
        Surface meshes are not available on an octree yet; when True this is
        logged and no meshes are stored.

    Returns
    -------
    StructuralModelResults
        A deep copy of the frame with ``cells`` set and, per cell, every group's
        ``scalar_field`` and ``mask`` and the ``lith_block``.

    Raises
    ------
    NotImplementedError
        If a fault frame is attached or a group uses an interpolator that has no
        fit/evaluate implementation yet.
    """
    grid = frame.grid
    if not isinstance(grid, OctreeGrid):
        raise TypeError(f"compute_structural_model_octree needs an OctreeGrid, got {type(grid).__name__}.")
    if frame.fault_frame is not None:
        raise NotImplementedError("Structural modeling with faults on an OctreeGrid is not implemented yet.")
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

    # Surface meshes will be extracted here, on mesh_cells.
    if extract_meshes:
        logger.info("Surface meshes on an OctreeGrid are not implemented yet; no meshes stored.")

    logger.info("Coarsening the octree for the stored result ...")
    cells = _coarsen_for_storage(frame, fields, mesh_cells)
    logger.info(
        "Octree cells: %d refined for surfaces, %d stored (equivalent regular grid: %d).",
        len(mesh_cells), len(cells), int(np.prod(grid.finest_resolution)),
    )

    logger.info("Evaluating the model at cell centers ...")
    values = _evaluate_groups(frame, fields, cells.centers(grid))
    lith, masks = _lithology_and_masks(frame, values)

    frame.cells = cells
    for group_idx, group in enumerate(frame.structural_groups):
        group.scalar_field = values[group_idx]
        group.mask = masks[group_idx]
        group.extended_scalar_field = None
    frame.lith_block = lith
    frame.check_cell_arrays()

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


def _fit_groups(frame: StructuralFrame) -> List[Optional[FittedField]]:
    """
    Fit every group with surface points and store the element scalar values.

    Groups without surface points get ``None``, which evaluates to zero, like the
    zero-initialized field a regular-grid group keeps when it is skipped.
    """
    fields: List[Optional[FittedField]] = []
    for group in frame.structural_groups:
        formations = [e.name for e in group.structural_elements]
        sp = frame.surface_points[frame.surface_points["formation"].isin(formations)]
        ori = (
            frame.orientations[frame.orientations["formation"].isin(formations)]
            if frame.orientations is not None
            else None
        )
        if ori is not None and ori.empty:
            ori = None
        if sp.empty:
            fields.append(None)
            continue

        field = fit_dispatch[group.interpolation_method](
            group=group,
            group_surface_points_df=sp,
            group_orientations_points_df=ori,
        )
        for elem in group.structural_elements:
            if elem.name not in field.scalar_values:
                raise ValueError(
                    f"Interpolator did not return a scalar value for element '{elem.name}' in group '{group.name}'."
                )
            elem.set_scalar_value(float(field.scalar_values[elem.name]))
        fields.append(field)
    return fields


def _evaluate_groups(
        frame: StructuralFrame, fields: List[Optional[FittedField]], points: FloatArray
) -> FloatArray:
    """Every group's field at the points, shape ``(n_groups, M)``."""
    values = np.zeros((len(frame.structural_groups), points.shape[0]), dtype=np.float64)
    for group_idx, field in enumerate(fields):
        if field is not None:
            values[group_idx] = field.evaluate(points)
    return values


def _lithology_and_masks(frame: StructuralFrame, values: FloatArray) -> Tuple[IntArray, BoolArray]:
    """
    Age masks and lithology ids from group values, pointwise and for one domain.

    Same rules as the regular grid: a group's age mask is ``value >= value of its
    oldest element`` (all True for the oldest group); inside a group, each point
    takes the youngest element whose value it reaches; younger groups overwrite
    older ones where their mask is True; 0 is basement.

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


def _refine_for_surfaces(frame: StructuralFrame, fields: List[Optional[FittedField]]) -> OctreeCells:
    """Refine from the base blocks until every group's surfaces and the lithology are at max level."""
    return refine_cells(frame.grid, lambda points: _surface_labels(frame, fields, points))


def _coarsen_for_storage(
        frame: StructuralFrame, fields: List[Optional[FittedField]], cells: OctreeCells
) -> OctreeCells:
    """Merge cells back wherever the lithology and the age masks don't change."""
    return coarsen_cells(frame.grid, cells, lambda points: _storage_labels(frame, fields, points))


def _surface_labels(
        frame: StructuralFrame, fields: List[Optional[FittedField]], points: FloatArray
) -> IntArray:
    """
    Labels for refining toward the surfaces.

    Column 0 is the lithology id. Then one column per group: which of the group's
    element surfaces the point lies above. The group columns keep every group's
    own surfaces at max level even where a younger group erodes them, which the
    unmasked surface meshes need.
    """
    values = _evaluate_groups(frame, fields, points)
    lith, _ = _lithology_and_masks(frame, values)
    columns = [lith]
    for group_idx, group in enumerate(frame.structural_groups):
        svals = sorted(e.get_scalar_value() for e in group.structural_elements if e.get_scalar_value() is not None)
        columns.append(np.searchsorted(np.asarray(svals, dtype=np.float64), values[group_idx], side="right"))
    return np.column_stack(columns)


def _storage_labels(
        frame: StructuralFrame, fields: List[Optional[FittedField]], points: FloatArray
) -> IntArray:
    """Labels the stored result must resolve: the lithology id and every group's age mask."""
    values = _evaluate_groups(frame, fields, points)
    lith, masks = _lithology_and_masks(frame, values)
    return np.column_stack([lith, masks.T.astype(np.int64)])
