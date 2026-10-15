"""
Fitted scalar fields: an interpolator fitted once and evaluated at any points.

The regular-grid pipeline fits and evaluates in one call (``interpolate_group_*``).
Adaptive grids evaluate the same field many times at points that are only known
during refinement, so they use ``fit_group_*`` functions that return a
:class:`FittedField` instead.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
Extent6 = Tuple[float, float, float, float, float, float]


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
