"""
Shared helpers for the minimum input data of the interpolators that use only
surface points (RBF, Ordinary Kriging, Universal Kriging).

- the phantom element added below a group with a single element,
- the test whether a polynomial trend (constant, plane, quadric) can be fitted
  through a set of points. RBF kernels with a trend and Universal Kriging's
  linear drift fail with a singular matrix otherwise.
"""

from __future__ import annotations

from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.floating]

PHANTOM_ELEMENT_NAME = "__phantom__"

# Number of polynomial terms in 3D per degree: 1; 1, x, y, z; plus x², y², z², xy, xz, yz
POLYNOMIAL_TERMS_3D = {0: 1, 1: 4, 2: 10}


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


def points_support_polynomial(points: FloatArray, degree: int) -> bool:
    """
    True if a polynomial trend of `degree` can be fitted through `points`.

    Needs at least ``POLYNOMIAL_TERMS_3D[degree]`` points, and for degree 1 the
    points must not all lie in one plane (for degree 2 not on one quadric surface).
    """
    if len(points) < POLYNOMIAL_TERMS_3D[degree]:
        return False
    if degree == 0:
        return True
    return int(np.linalg.matrix_rank(_polynomial_matrix(points, degree))) == POLYNOMIAL_TERMS_3D[degree]
