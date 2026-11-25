"""
Radial Basis Function (RBF) interpolation for a single structural group.

This module provides a pure interpolator function that:
- derives strictly increasing scalar values per element (oldest=1 .. youngest=n),
- fits an RBF interpolator to surface points,
- evaluates the scalar field on a regular grid.

No mutation of the passed `group` is performed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from scipy.interpolate import RBFInterpolator

from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    RBF_KERNELS_USING_EPSILON,
)

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]

PHANTOM_ELEMENT_NAME = "__phantom__"

# Polynomial degree SciPy adds to each kernel by default (`degree=None` means the
# kernel's minimum degree, at least 0). We keep SciPy's default: lowering it below
# the minimum makes the system not uniquely solvable for these kernels.
RBF_KERNEL_POLYNOMIAL_DEGREE: Dict[str, int] = {
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}

# Kernels that need only a constant polynomial term (degree 0), i.e. one point.
_RBF_KERNELS_FOR_FEW_POINTS = ("linear", "multiquadric")


def add_phantom_element(surface_points_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Add a synthetic older element below a group that has only one element.

    The surface points are copied and shifted down by 10 % of their bounding-box
    diagonal (at least 1 unit), labelled ``PHANTOM_ELEMENT_NAME``. Interpolators
    that only use surface points (RBF, Ordinary and Universal Kriging) need at
    least two different values to fit a scalar field.

    Returns
    -------
    surface_points_with_phantom, shift
    """
    coords = surface_points_df[["X", "Y", "Z"]].to_numpy()
    bbox_diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    shift = max(0.1 * bbox_diag, 1.0)
    phantom_df = surface_points_df.assign(Z=surface_points_df["Z"] - shift, formation=PHANTOM_ELEMENT_NAME)
    return pd.concat([surface_points_df, phantom_df], ignore_index=True), shift


def _polynomial_matrix(points: FloatArray, degree: int) -> FloatArray:
    """Monomials up to `degree` evaluated at `points` (centred and scaled, as SciPy does)."""
    centred = points - points.mean(axis=0)
    scale = float(np.abs(centred).max())
    scaled = centred / scale if scale > 0 else centred
    x, y, z = scaled.T
    columns = [np.ones(len(points))]
    if degree >= 1:
        columns += [x, y, z]
    if degree >= 2:
        columns += [x * x, y * y, z * z, x * y, x * z, y * z]
    return np.column_stack(columns)


def rbf_input_problem(points: FloatArray, kernel: str, neighbors: Optional[int] = None) -> Optional[str]:
    """
    Check whether RBF interpolation with `kernel` can be fitted to `points`.

    Kernels with a polynomial term (``thin_plate_spline`` and ``cubic``: a plane,
    ``quintic``: a curved surface) need at least as many points as the polynomial
    has terms (4 or 10), and the points must not all lie in one plane (or, for
    ``quintic``, on one simple curved surface). Otherwise SciPy fails with
    "At least N data points are required" or "Singular matrix".

    Parameters
    ----------
    points
        (N, 3) coordinates that go into the fit, including phantom points.
    kernel
        RBF kernel name.
    neighbors
        Number of nearest neighbours used per evaluation point, or None for all.

    Returns
    -------
    A description of the problem for the user, or None if the points are sufficient.
    """
    degree = RBF_KERNEL_POLYNOMIAL_DEGREE.get(kernel, 0)
    n_terms = {0: 1, 1: 4, 2: 10}[degree]
    alternatives = " or ".join(f"'{k}'" for k in _RBF_KERNELS_FOR_FEW_POINTS)
    n_points = len(points)

    if n_points < n_terms:
        return (
            f"the kernel '{kernel}' needs at least {n_terms} surface points, "
            f"but only {n_points} are available. Add surface points, or choose the kernel "
            f"{alternatives}, which also work with fewer points."
        )
    if neighbors is not None and min(neighbors, n_points) < n_terms:
        return (
            f"the kernel '{kernel}' needs at least {n_terms} neighbours, "
            f"but 'neighbors' is set to {neighbors}. Increase 'neighbors' or leave it empty."
        )
    if degree >= 1 and np.linalg.matrix_rank(_polynomial_matrix(points, degree)) < n_terms:
        shape = "in one plane" if degree == 1 else "on one plane or one simple curved surface"
        return (
            f"all {n_points} surface points lie {shape}, so the kernel '{kernel}' cannot be "
            f"fitted. Add surface points outside this plane, or choose the kernel {alternatives}."
        )
    return None


def interpolate_group_radial_basis_function(
    *,
    group: Any,  # expected: StructuralGroup-like (group.name, group.structural_elements, group.get_interpolation_params())
    grid: Any,  # expected: RegularGrid-like (grid.resolution and either grid.grid_coordinates or grid.gridx/y/z)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,  # unused here
) -> ScalarFieldAndValues:
    """
    Radial Basis Function interpolation for a single structural group (pure).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]`
        - `get_interpolation_params() -> RBFParams-like` (kernel/smoothing/epsilon/neighbors)
    grid
        RegularGrid-like object providing:
        - `resolution: tuple[int, int, int]`
        - optionally `grid_coordinates: (N, 3) ndarray`
        - otherwise `grid.gridx`, `grid.gridy`, `grid.gridz` 1D coordinate arrays
    group_surface_points_df
        DataFrame of constraint points with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        Optional orientations DataFrame. Accepted for API consistency but not used.

    Returns
    -------
    scalar_field : np.ndarray
        Interpolated scalar field reshaped to `grid.resolution` and then transposed
        exactly as in the original implementation.
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value. Values are strictly increasing from
        oldest=1 to youngest=n (group.structural_elements is assumed youngest->oldest,
        so reversed order is used here).

    Notes
    -----
    - Assigns strictly increasing scalar values: oldest = 1, youngest = n
    - Uses params from `group.get_interpolation_params()` (expected to be your RBFParams)
    - Does NOT mutate `group`
    """
    # Check if any surface points are provided
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")

    # Check if orientations are provided and warn that they will not be used
    if group_orientations_points_df is not None and not group_orientations_points_df.empty:
        warnings.warn(
            f"Orientations provided for group '{group.name}' will not be used in RBF interpolation",
            UserWarning,
        )

    # Check if there are at least two distinct elements in the group
    _phantom_name: Optional[str] = None
    if len(group.structural_elements) < 2:
        if len(group.structural_elements) == 1:
            _phantom_name = PHANTOM_ELEMENT_NAME
            group_surface_points_df, _shift = add_phantom_element(group_surface_points_df)
            warnings.warn(
                f"Group '{group.name}' has only one structural element. RBF interpolation "
                f"requires at least two. A phantom 'older' element has been synthesized by "
                f"shifting all points downward by {_shift:.4g} units (10%% of the point "
                f"cloud bounding-box diagonal). This approximation assumes sub-horizontal "
                f"layering and WILL produce incorrect results for steeply dipping or "
                f"overturned structures. Consider using FDI, PLI, UCK, or GeoINR instead.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Group '{group.name}' must contain at least two structural elements for RBF interpolation"
            )

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")

    # 1) scalar values oldest->youngest = 1..n (groups list is youngest->oldest, so reverse)
    scalar_values_by_element: Dict[str, float] = {
        elem.name: float(i)
        for i, elem in enumerate(reversed(group.structural_elements), start=1)
    }
    if _phantom_name is not None:
        scalar_values_by_element[_phantom_name] = 0.0

    # 2) map formations -> scalar values
    vals: FloatArray = (
        group_surface_points_df["formation"]
        .map(scalar_values_by_element)
        .astype(float)
        .to_numpy()
    )
    if _phantom_name is not None:
        scalar_values_by_element.pop(_phantom_name)

    # 3) coordinates of control points
    coords: FloatArray = group_surface_points_df[["X", "Y", "Z"]].to_numpy(dtype=float)

    # 4) parameters (expected: your RBFParams-like object)
    params = group.get_interpolation_params()

    # Vertical anisotropy: stretch z of points and grid alike (1.0 = isotropic)
    z_scaling = float(getattr(params, "anisotropy_scaling_z", 1.0))
    axis_scaling = np.array([1.0, 1.0, z_scaling])
    coords = coords * axis_scaling

    # 5) fit RBF
    rbfi = RBFInterpolator(
        coords,
        vals,
        kernel=params.kernel,
        smoothing=params.smoothing,
        # scale-invariant kernels are evaluated without the shape parameter
        epsilon=params.epsilon if params.kernel in RBF_KERNELS_USING_EPSILON else None,
        neighbors=params.neighbors,  # None => all points
    )

    # 6) evaluate on grid
    # Prefer a precomputed (N,3) array if your grid exposes it; otherwise build from axes.
    if hasattr(grid, "grid_coordinates") and grid.grid_coordinates is not None:
        grid_points: FloatArray = grid.grid_coordinates
    else:
        gx, gy, gz = np.meshgrid(grid.gridx, grid.gridy, grid.gridz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    scalar_flat: FloatArray = rbfi(grid_points * axis_scaling)
    scalar_field = scalar_flat.reshape(tuple(grid.resolution)).T  # (nx, ny, nz)

    return scalar_field, scalar_values_by_element
