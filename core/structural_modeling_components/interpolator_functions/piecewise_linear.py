"""
LoopStructural PLI interpolation for a single structural group.

This module provides a pure interpolator function that:
- assigns strictly increasing scalar values per element (oldest=1 .. youngest=n),
- builds a LoopStructural GeologicalModel from surface points + orientations,
- creates a PLI foliation (piecewise-linear interpolator),
- evaluates the feature value on a RegularGrid.

Notes
-----
- The function does NOT mutate the passed `group`.
- `group.structural_elements` are assumed ordered youngest -> oldest (framework convention).
- The scalar field is returned in ZYX shape (reshaped to the grid resolution, then transposed).
- Orientations only constrain the direction of the layers at their location; they
  carry no scalar value, so they may lie anywhere, not only on an interface.
  In LoopStructural 1.8.2 the gradient constraint of the piecewise-linear interpolator
  is not implemented (`P1Interpolator.add_gradient_constraints` does nothing), so each
  orientation is passed as two tangent constraints instead (gradient perpendicular to
  its strike and dip vectors, weight `tpw`). This is what FDI's gradient constraint
  does internally. Tangents do not tell up from down; that comes from the formation
  values. With only one formation (all values equal), LoopStructural additionally
  turns the gradient rows into normal-vector constraints, which define which way is up.

PLI-specific parameters
-----------------------
All LoopStructural creation/evaluation parameters are sourced from
`group.get_interpolation_params()`.

Expected attributes on params
-----------------------------
- nelements: int | float
    Discretisation size passed to `create_and_add_foliation`.
    (For PLI this controls the piecewise-linear support/mesh density.)
- regularisation: float
    Weight of the smoothness term, passed as `regularisation`.
- orientation_weight: float
    Weight of the orientation (tangent) constraints, passed as `tpw`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from LoopStructural import GeologicalModel

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


def _strike_and_dip_vectors(normals: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Two unit vectors lying in the plane of each normal (strike and dip direction)."""
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    strike = np.cross(n, [0.0, 0.0, 1.0])
    horizontal = np.linalg.norm(strike, axis=1) < 1e-8  # horizontal layer: any strike
    strike[horizontal] = [1.0, 0.0, 0.0]
    strike /= np.linalg.norm(strike, axis=1, keepdims=True)
    dip = np.cross(n, strike)
    return strike, dip


def _scale_tangent_weight_to_mesh(feature: Any, orientation_weight: float) -> None:
    """
    Make `orientation_weight` independent of the mesh size.

    LoopStructural multiplies each tangent constraint by the volume of its tetrahedron,
    so with a fixed `tpw` the orientations count less the finer the mesh (with 10 000
    instead of 2000 elements, about five times less). Dividing by the mean element
    volume removes this. The feature is built lazily (when it is first evaluated), so the
    weight can still be changed through the builder's build arguments.
    """
    mean_volume = float(np.mean(feature.builder.interpolator.support.element_size))
    feature.builder.update_build_arguments({"tpw": orientation_weight / mean_volume})


def interpolate_group_piecewise_linear(
    *,
    group: Any,  # StructuralGroup-like (name, structural_elements with .name, get_interpolation_params())
    grid: Any,  # RegularGrid-like (extent, resolution, grid_coordinates)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: pd.DataFrame,
) -> ScalarFieldAndValues:
    """
    LoopStructural PLI interpolation for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> params-like`
          The params object is expected to provide PLI-relevant attributes described in the module docstring.
    grid
        RegularGrid-like object. Expected members:
        - `extent: tuple[float, float, float, float, float, float]`
        - `resolution: tuple[int, int, int]`
        - `grid_coordinates: (N, 3) ndarray` (cell-center coordinates)
    group_surface_points_df
        DataFrame with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        DataFrame with columns ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"].

    Returns
    -------
    scalar_field : np.ndarray
        Evaluated scalar field on the grid in ZYX shape (shaped to `tuple(grid.resolution)`,
        then transposed).
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).

    Raises
    ------
    ValueError
        If surface/orientation inputs are missing/empty or required columns are absent,
        or if the DataFrames include formations not present in the group.
    """
    # --- validation ---
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")
    if group_orientations_points_df is None or group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group '{group.name}'")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")
    for col in ("X", "Y", "Z", "G_x", "G_y", "G_z", "formation"):
        if col not in group_orientations_points_df.columns:
            raise ValueError(f"Orientations for '{group.name}' missing column '{col}'")

    # Ensure all formations in the DFs belong to this group
    group_elem_names = [e.name for e in group.structural_elements]
    unknown_sp = set(group_surface_points_df["formation"].unique()) - set(group_elem_names)
    unknown_ori = set(group_orientations_points_df["formation"].unique()) - set(group_elem_names)
    if unknown_sp:
        raise ValueError(
            f"Surface points for group '{group.name}' contain formations not in the group: {sorted(unknown_sp)}"
        )
    if unknown_ori:
        raise ValueError(
            f"Orientations for group '{group.name}' contain formations not in the group: {sorted(unknown_ori)}"
        )

    # --- per-element scalar values (oldest=1 ... youngest=n) ---
    names_old_to_young = [e.name for e in reversed(group.structural_elements)]
    scalar_values_by_element: Dict[str, float] = {
        name: float(i) for i, name in enumerate(names_old_to_young, start=1)
    }

    # Map formation name -> scalar ('val') for LoopStructural
    formation_to_scalar = scalar_values_by_element

    # --- build LoopStructural input tables ---
    # Surface points table
    surface_points = group_surface_points_df.copy()
    surface_points["feature_name"] = group.name
    surface_points["val"] = surface_points["formation"].map(formation_to_scalar)

    # LoopStructural expects columns: X, Y, Z, val, feature_name, gx, gy, gz.
    surface_points = surface_points[["X", "Y", "Z", "val", "feature_name"]]
    surface_points["gx"] = np.nan
    surface_points["gy"] = np.nan
    surface_points["gz"] = np.nan

    # Orientations table. No value: orientations may lie anywhere, not only on the
    # interface of their formation (see module docstring). The gradient rows only take
    # effect when all values are equal (LoopStructural then turns them into normals).
    orientations = group_orientations_points_df.copy()
    orientations["feature_name"] = group.name
    orientations["val"] = np.nan
    orientations = orientations.rename(columns={"G_x": "gx", "G_y": "gy", "G_z": "gz"})
    orientations = orientations[["X", "Y", "Z", "val", "feature_name", "gx", "gy", "gz"]]

    # The direction of each orientation as two tangent rows: the gradient must be
    # perpendicular to the strike and to the dip vector.
    strike, dip = _strike_and_dip_vectors(orientations[["gx", "gy", "gz"]].to_numpy(dtype=float))
    tangent_rows = []
    for vectors in (strike, dip):
        rows = orientations[["X", "Y", "Z", "feature_name"]].copy()
        rows[["tx", "ty", "tz"]] = vectors
        tangent_rows.append(rows)

    # Combined table
    data_combined = pd.concat([surface_points, orientations, *tangent_rows], ignore_index=True)

    # --- build LoopStructural model ---
    # GeologicalModel(min_bounds, max_bounds) expects 3-vectors
    model = GeologicalModel(np.array(grid.extent[::2]), np.array(grid.extent[1::2]))
    if hasattr(model, "data"):
        model.data = data_combined
    else:
        # backward compatibility
        model.set_model_data(data_combined)

    # No stratigraphic column is set: LoopStructural only reads it in
    # evaluate_model()/fault topology, neither of which is used here -- we only
    # evaluate this one feature's scalar field and assign lithology ourselves.

    # --- parameters (PLI) ---
    params = group.get_interpolation_params() if hasattr(group, "get_interpolation_params") else None
    if params is None:
        raise ValueError(f"Missing interpolation params for group '{group.name}' (required for PLI)")

    # Create the foliation / feature (PLI)
    #
    # buffer=0.2 (LoopStructural's own default -- do not drop to 0.0): PLI's
    # tetrahedral support has mesh nodes right at the domain edge with little
    # to no nearby constraint when there's no margin, which solves to NaN for
    # some of them under LoopStructural >=1.8 (confirmed directly: with this
    # exact data, buffer=0.0 produces NaNs in the outermost grid slice,
    # buffer=0.2 does not). finite_difference.py uses the same buffer.
    create_kwargs: Dict[str, Any] = dict(
        interpolatortype="PLI",
        nelements=params.nelements,
        buffer=0.2,
        # Conjugate gradients with LoopStructural's default tolerance (not exposed).
        solver="cg",
        # Least-squares weights: surface points 1 (LoopStructural default), smoothness
        # `regularisation`, orientations (as tangent constraints, see module docstring)
        # `orientation_weight`.
        regularisation=params.regularisation,
        tpw=params.orientation_weight,
    )

    feature = model.create_and_add_foliation(group.name, **create_kwargs)
    _scale_tangent_weight_to_mesh(feature, params.orientation_weight)

    # Evaluate on grid
    #
    # scale=True (not False): GeologicalModel.data's setter unconditionally
    # rescales input X/Y/Z into the model's internal frame (subtract origin,
    # divide by scale_factor) before the interpolator is built/solved -- so
    # evaluation must undo that same transform on the query points, via
    # GeologicalModel.scale(), or this samples the solved field at the wrong
    # locations whenever the grid's origin isn't (0,0,0) (or scale_factor
    # isn't 1). finite_difference.py already does this correctly.
    regular_grid: FloatArray = grid.grid_coordinates  # (N, 3)
    sf_flat = model.evaluate_feature_value(
        group.name,
        regular_grid,
        scale=True,
    )

    # grid coordinates are ordered with x slowest: reshape to (nx, ny, nz), transpose to ZYX
    scalar_field = np.asarray(sf_flat).reshape(tuple(grid.resolution)).T

    return scalar_field, scalar_values_by_element
