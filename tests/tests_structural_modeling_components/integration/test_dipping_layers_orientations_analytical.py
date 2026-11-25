"""
Dipping layers whose dip is known only from orientations.

All surface points lie on one vertical line of x (x = 1), so they fix the height of the
two interfaces there but not their dip. The dip comes from four orientations near the
model edges, placed inside a layer rather than on an interface. Methods that use the
direction of orientations must reproduce the analytical lithology; RBF, which ignores
orientations, serves as a control that the scenario really needs them.

Guards against:
- orientations being used as interface points (they would pull the interface towards
  their location; the test also shifts them away from the layer centre),
- PLI ignoring orientation directions (LoopStructural's PLI has no gradient constraint;
  orientations are passed as tangent constraints instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general  # type: ignore
from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import InterpolationMethod

_SLOPE = 0.25                 # dz/dx of both interfaces (about 14 degrees)
_Z_TOP, _Z_BOTTOM = 0.45, 0.2  # interface heights at x = 1


def _top(x):
    return _Z_TOP + _SLOPE * (x - 1.0)


def _bottom(x):
    return _Z_BOTTOM + _SLOPE * (x - 1.0)


def _input(orientation_offset: float) -> InputData_StructuralElements:
    sp_rows = []
    for y in (0.2, 0.5, 0.8):
        sp_rows.append({"X": 1.0, "Y": y, "Z": _top(1.0), "formation": "rock2"})
        sp_rows.append({"X": 1.0, "Y": y, "Z": _bottom(1.0), "formation": "rock1"})

    normal = np.array([-_SLOPE, 0.0, 1.0]) / np.hypot(_SLOPE, 1.0)
    ori_rows = []
    for x in (0.2, 1.8):
        for y in (0.3, 0.7):
            z = 0.5 * (_top(x) + _bottom(x)) + orientation_offset  # inside rock1, not on an interface
            ori_rows.append({"X": x, "Y": y, "Z": z, "G_x": normal[0], "G_y": normal[1], "G_z": normal[2],
                             "formation": "rock1"})

    return InputData_StructuralElements(
        name="dipping_layers",
        mapping_object={"Strat_Series1": ("rock2", "rock1")},
        surface_points=pd.DataFrame(sp_rows),
        orientations=pd.DataFrame(ori_rows),
    )


def _expected_lith_block(grid: RegularGrid) -> np.ndarray:
    """XYZ lithology: basement 0 below the lower interface, rock1 1, rock2 2 above the upper one."""
    x = np.broadcast_to(grid.gridx.reshape(-1, 1, 1), tuple(grid.resolution))
    z = np.broadcast_to(grid.gridz.reshape(1, 1, -1), tuple(grid.resolution))
    expected = np.zeros(tuple(grid.resolution), dtype=int)
    expected[z >= _bottom(x)] = 1
    expected[z >= _top(x)] = 2
    return expected


def _match(method: InterpolationMethod, params: dict, orientation_offset: float) -> float:
    grid = RegularGrid(extent=(0.0, 2.0, 0.0, 1.0, 0.0, 1.0), resolution=(20, 10, 10))
    frame = general.build_structural_frame(input_data_elements=_input(orientation_offset), grid=grid)
    for g in frame.structural_groups:
        g.set_interpolation_method(method)
        if params:
            g.configure_interpolation_params(**params)
    res = general.compute_structural_model(frame, extract_meshes=False, verbose=False)
    return float(np.mean(res.structural_frame.lith_block == _expected_lith_block(grid)))


# PLI's default mesh (2000 elements for this little data) is too coarse here (83 %).
_CASES = [
    (InterpolationMethod.FINITE_DIFFERENCES, {}),
    (InterpolationMethod.PIECEWISE_LINEAR, {"nelements": 10_000}),
    (InterpolationMethod.UNIVERSAL_COKRIGING, {}),
]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("orientation_offset", [0.0, 0.1], ids=["mid_layer", "off_centre"])
@pytest.mark.parametrize("method,params", _CASES, ids=[m.value for m, _ in _CASES])
def test_dip_from_orientations(method, params, orientation_offset):
    if method == InterpolationMethod.UNIVERSAL_COKRIGING:
        pytest.importorskip("gempy")
    match = _match(method, params, orientation_offset)
    assert match >= 0.95, f"[{method.name}] {match:.1%} of cells match the dipping layers"


@pytest.mark.integration
@pytest.mark.slow
def test_dip_scenario_needs_orientations():
    """Control: without orientations (RBF) the dip is unknown and the result is far off."""
    match = _match(InterpolationMethod.RADIAL_BASIS_FUNCTION, {"kernel": "linear"}, 0.0)
    assert match < 0.7
