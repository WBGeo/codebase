"""
Fitted scalar fields: an interpolator fitted once and evaluated at any points.

The regular-grid pipeline fits and evaluates in one call (``interpolate_group_*``).
Adaptive grids evaluate the same field many times at points that are only known
during refinement, so they use ``fit_group_*`` functions that return a
:class:`FittedField` instead.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
from pykrige.core import _adjust_for_anisotropy
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from scipy.spatial.distance import cdist

FloatArray = npt.NDArray[np.floating]
Extent6 = Tuple[float, float, float, float, float, float]

# Relative tolerance within which a field value counts as its isovalue (snap_to_isovalues).
# Group fields have their isovalues at 1..n, about 0.1 apart per finest cell, and
# rounding errors near 1e-15, so this is far above rounding and far below a cell.
ISOVALUE_SNAP: float = 1e-9


def snap_to_isovalues(values: FloatArray, isovalues: Iterable[float]) -> npt.NDArray[np.float64]:
    """
    The values, with those within ``ISOVALUE_SNAP * max(1, |isovalue|)`` of an isovalue set to it.

    A surface that runs exactly through sample points (e.g. a flat horizon at a
    round depth on a plane of octree cell corners or regular cell centers) has
    the isovalue there only up to rounding, and the sign of that rounding error
    decides point by point which side of the surface they are on: the dual
    contouring of the octree scrambled such surfaces, and lithology ids along
    them were noise. Snapped values equal the isovalue, so ``>=`` puts all of
    them on one side. Both grid types snap, so they decide such ties alike.

    Args:
        values: Field values, any shape; not changed.
        isovalues: The isovalues to snap to.

    Returns:
        A float64 copy of ``values`` with the snapped entries.
    """
    snapped = np.array(values, dtype=np.float64)
    for iso in isovalues:
        close = np.abs(snapped - iso) <= ISOVALUE_SNAP * max(1.0, abs(float(iso)))
        snapped[close] = iso
    return snapped


class FittedField(Protocol):
    """
    A scalar field fitted to the data of one structural group.

    Attributes:
        scalar_values: Isovalue of each structural element of the group,
            ``element name -> value``, as the matching ``interpolate_group_*``
            function returns them.
    """

    scalar_values: Dict[str, float]

    def evaluate(self, xyz: FloatArray) -> FloatArray:
        """
        Field values at points.

        Args:
            xyz: Point coordinates, shape ``(M, 3)``.

        Returns:
            Values of shape ``(M,)``.
        """
        ...


class LoopStructuralFittedField:
    """
    A foliation of a fitted LoopStructural ``GeologicalModel`` (FDI, PLI), evaluable anywhere.

    Attributes:
        scalar_values: Element name -> isovalue, oldest = 1 .. youngest = n.
    """

    def __init__(self, model: Any, feature_name: str, scale: bool, scalar_values: Dict[str, float]) -> None:
        self._model = model
        self._feature_name = feature_name
        self._scale = scale
        self.scalar_values = scalar_values

    def evaluate(self, xyz: FloatArray) -> FloatArray:
        """Feature values at points of shape ``(M, 3)``; returns shape ``(M,)``."""
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self._model.evaluate_feature_value(self._feature_name, points, scale=self._scale))


class KrigingFittedField:
    """
    A fitted pykrige ``OrdinaryKriging3D`` or ``UniversalKriging3D``, evaluable anywhere.

    The kriging estimate at a point x is ``b(x)ᵀ A⁻¹ r``, with ``A`` the kriging
    matrix, ``r`` the data values padded with zeros for the drift and unbiasedness
    rows, and ``b(x)`` the variogram to the data plus the drift terms at x. The
    weights ``A⁻¹ r`` don't depend on x, so they are solved once here ("dual
    kriging") and each point costs one dot product over the data, instead of the
    full per-point solve pykrige's ``execute`` does to also return the variance.
    The estimate equals ``execute``'s up to rounding.

    With a moving neighbourhood (``n_closest_points``) the weights differ per
    point, so those fields are evaluated with pykrige's own loop backend.

    Attributes:
        scalar_values: Element name -> isovalue, oldest = 1 .. youngest = n.
    """

    # Evaluation points per chunk, bounded so the (points x data) distance matrix stays small.
    _CHUNK_ENTRIES: int = 10_000_000

    def __init__(
            self,
            kriging: Union[OrdinaryKriging3D, UniversalKriging3D],
            scalar_values: Dict[str, float],
            n_closest_points: Union[int, None] = None,
    ) -> None:
        if isinstance(kriging, UniversalKriging3D) and (kriging.specified_drift or kriging.functional_drift):
            raise NotImplementedError("Only the 'regional_linear' drift is supported for evaluation at points.")
        self._kriging = kriging
        self._n_closest_points = n_closest_points
        self.scalar_values = scalar_values
        self._regional_linear = isinstance(kriging, UniversalKriging3D) and kriging.regional_linear_drift
        self._data = np.column_stack((kriging.X_ADJUSTED, kriging.Y_ADJUSTED, kriging.Z_ADJUSTED))
        self._weights = None if n_closest_points is not None else self._dual_weights()

    def _dual_weights(self) -> FloatArray:
        """Solve ``A w = r`` once for the data, drift and unbiasedness weights."""
        k = self._kriging
        n = self._data.shape[0]
        if isinstance(k, UniversalKriging3D):
            n_withdrifts = n + (3 if self._regional_linear else 0)
            a = k._get_kriging_matrix(n, n_withdrifts)
        else:
            a = k._get_kriging_matrix(n)
        rhs = np.zeros(a.shape[0])
        rhs[:n] = k.VALUES
        if k.pseudo_inv:
            from pykrige.core import P_INV
            return P_INV[k.pseudo_inv_type](a) @ rhs
        return scipy.linalg.solve(a, rhs)

    def evaluate(self, xyz: FloatArray) -> FloatArray:
        """Kriging estimate at points of shape ``(M, 3)``; returns shape ``(M,)``."""
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        k = self._kriging
        if self._weights is None:
            values, _ = k.execute(
                "points", points[:, 0], points[:, 1], points[:, 2],
                backend="loop", n_closest_points=self._n_closest_points,
            )
            return np.asarray(values, dtype=np.float64)

        adjusted = _adjust_for_anisotropy(
            points.copy(),  # shifts its input in place
            [k.XCENTER, k.YCENTER, k.ZCENTER],
            [k.anisotropy_scaling_y, k.anisotropy_scaling_z],
            [k.anisotropy_angle_x, k.anisotropy_angle_y, k.anisotropy_angle_z],
        )
        n = self._data.shape[0]
        w_data, w_rest = self._weights[:n], self._weights[n:]
        out = np.empty(points.shape[0], dtype=np.float64)
        chunk = max(1, self._CHUNK_ENTRIES // max(n, 1))
        for start in range(0, points.shape[0], chunk):
            p = adjusted[start:start + chunk]
            d = cdist(p, self._data, "euclidean")
            b = -k.variogram_function(k.variogram_model_parameters, d)
            if k.exact_values:
                b[np.absolute(d) <= k.eps] = 0.0  # a point on a datum takes its value, as in execute
            value = b @ w_data
            i = 0
            if self._regional_linear:
                value += p @ w_rest[:3]
                i = 3
            value += w_rest[i]  # unbiasedness (Lagrange) term
            out[start:start + chunk] = value
        return out
