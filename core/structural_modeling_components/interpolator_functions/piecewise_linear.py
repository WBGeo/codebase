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
- The scalar field reshape + transpose is preserved exactly as implemented.
- Orientation/gradient observations are treated as point constraints by LoopStructural
  (i.e., their XYZ location matters to the solution).

PLI-specific parameters
-----------------------
All LoopStructural creation/evaluation parameters are sourced from
`group.get_interpolation_params()`.

Expected attributes on params
-----------------------------
- nelements: int | float
    Discretisation size passed to `create_and_add_foliation`.
    (For PLI this controls the piecewise-linear support/mesh density.)
- solver: str
    Solver argument passed to `create_and_add_foliation` (if supported by your LS version).
- damp: bool
    Damp argument passed to `create_and_add_foliation` (if supported by your LS version).
- tol: float | None
    Tolerance passed to `create_and_add_foliation` as `tol` (if provided).
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
        Evaluated scalar field on the grid, shaped to `tuple(grid.resolution)` and transposed
        exactly as in the original implementation.
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

    # Orientations table
    orientations = group_orientations_points_df.copy()
    orientations["feature_name"] = group.name
    orientations["val"] = orientations["formation"].map(formation_to_scalar)
    orientations = orientations.rename(columns={"G_x": "gx", "G_y": "gy", "G_z": "gz"})
    orientations = orientations[["X", "Y", "Z", "val", "feature_name", "gx", "gy", "gz"]]

    # Combined table
    data_combined = pd.concat([surface_points, orientations], ignore_index=True)

    # --- build LoopStructural model ---
    # GeologicalModel(min_bounds, max_bounds) expects 3-vectors
    model = GeologicalModel(np.array(grid.extent[::2]), np.array(grid.extent[1::2]))
    if hasattr(model, "data"):
        model.data = data_combined
    else:
        # backward compatibility
        model.set_model_data(data_combined)

    # Stratigraphic column: preserve original order (group.structural_elements)
    stratigraphic_column: Dict[str, Dict[str, Dict[str, float]]] = {group.name: {}}
    for i, rock in enumerate(group.structural_elements):
        stratigraphic_column[group.name][rock.name] = {"min": i, "max": i + 1, "id": i}
    model.set_stratigraphic_column(stratigraphic_column)

    # --- parameters (PLI) ---
    params = group.get_interpolation_params() if hasattr(group, "get_interpolation_params") else None
    if params is None:
        raise ValueError(f"Missing interpolation params for group '{group.name}' (required for PLI)")

    # Create the foliation / feature (PLI)
    create_kwargs: Dict[str, Any] = dict(
        interpolatortype="PLI",
        nelements=params.nelements,
        buffer=0.0,
        solver=params.solver,
        damp=params.damp,
    )
    if getattr(params, "tol", None) is not None:
        create_kwargs["tol"] = params.tol

    _ = model.create_and_add_foliation(group.name, **create_kwargs)

    # Evaluate on grid
    regular_grid: FloatArray = grid.grid_coordinates  # (N, 3)
    sf_flat = model.evaluate_feature_value(
        group.name,
        regular_grid,
        scale=False,
    )

    # Match previous implementation: reshape then transpose
    scalar_field = np.asarray(sf_flat).reshape(tuple(grid.resolution)).T

    return scalar_field, scalar_values_by_element
