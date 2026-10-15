"""
Fitted scalar fields: an interpolator fitted once and evaluated at any points.

The regular-grid pipeline fits and evaluates in one call (``interpolate_group_*``).
Adaptive grids evaluate the same field many times at points that are only known
during refinement, so they use ``fit_group_*`` functions that return a
:class:`FittedField` instead.
"""

from __future__ import annotations

from typing import Dict, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


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
