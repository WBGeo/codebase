# tests/test_interpolate_group_universal_kriging.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# TODO: change to your real import path if different:
from core.structural_modeling_components.interpolator_functions.universal_kriging_per_group import (
    interpolate_group_universal_kriging,
)


# --- Minimal fakes ------------------------------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeUKParams:
    def __init__(
        self,
        variogram_model="spherical",
        sill=1.0,
        range=2.0,
        nugget=0.1,
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        neighbors=None,
        # UK-specific
        drift_terms=None,
        specified_drift_arrays=None,
        specified_drift=None,
        external_drift=None,
        external_drift_grid=None,
    ):
        self.variogram_model = variogram_model
        self.sill = sill
        self.range = range
        self.nugget = nugget
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.neighbors = neighbors

        self.drift_terms = drift_terms
        self.specified_drift_arrays = specified_drift_arrays
        self.specified_drift = specified_drift
        self.external_drift = external_drift
        self.external_drift_grid = external_drift_grid


class FakeGroup:
    def __init__(self, name: str, element_names: list[str], params: FakeUKParams | None = None):
        self.name = name
        # function assumes youngest->oldest
        self.structural_elements = [FakeElem(n) for n in element_names]
        self._params = params or FakeUKParams()

    def get_interpolation_params(self):
        return self._params


class FakeGrid:
    def __init__(self, resolution=(2, 2, 2)):
        nx, ny, nz = resolution
        self.resolution = resolution
        self.gridx = np.arange(nx, dtype=float)
        self.gridy = np.arange(ny, dtype=float)
        self.gridz = np.arange(nz, dtype=float)


def surface_df(formations: list[str]) -> pd.DataFrame:
    n = len(formations)
    return pd.DataFrame(
        {
            "X": np.linspace(0.0, 1.0, n),
            "Y": np.linspace(0.0, 1.0, n),
            "Z": np.linspace(0.0, 1.0, n),
            "formation": formations,
        }
    )


# --- Core behavior tests ------------------------------------------------------

def test_raises_when_surface_points_missing_or_empty():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=None)

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=pd.DataFrame())


def test_warns_when_orientations_provided_but_unused(monkeypatch):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"])
    odf = pd.DataFrame({"dip": [1]})

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    class DummyUK3D:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    with pytest.warns(UserWarning, match="will not be used"):
        interpolate_group_universal_kriging(
            group=group,
            grid=grid,
            group_surface_points_df=sdf,
            group_orientations_points_df=odf,
        )


def test_requires_at_least_two_elements():
    group = FakeGroup("G", ["only_one"])
    grid = FakeGrid()
    sdf = surface_df(["only_one"])

    with pytest.raises(ValueError, match="at least two structural elements"):
        interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "formation"])
def test_raises_on_missing_required_columns(missing_col: str):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"]).drop(columns=[missing_col])

    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)


def test_raises_when_surface_points_contain_unknown_formations():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "NOT_IN_GROUP"])

    with pytest.raises(ValueError, match="contain formations not in the group"):
        interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)


def test_scalar_values_mapping_is_oldest_1_youngest_n(monkeypatch):
    group = FakeGroup("G", ["youngest", "middle", "oldest"])
    grid = FakeGrid()
    sdf = surface_df(["oldest", "middle", "youngest"])

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    class DummyUK3D:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    field, mapping = interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    assert mapping == {"oldest": 1.0, "middle": 2.0, "youngest": 3.0}
    assert np.isfinite(field).all()


# --- UK-specific parameter behavior tests ------------------------------------

def test_default_drift_terms_is_regional_linear(monkeypatch):
    params = FakeUKParams(drift_terms=None)
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid()
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.2, 0.8],
            "Y": [0.0, 1.0, 0.1, 0.9],
            "Z": [0.0, 1.0, 0.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    captured = {}

    class DummyUK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["kwargs"] = kwargs

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    assert captured["kwargs"]["drift_terms"] == "regional_linear"


def test_drift_terms_tuple_or_set_is_converted_to_list(monkeypatch):
    params = FakeUKParams(drift_terms=("regional_linear", "point_log"))
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid()
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.2, 0.8],
            "Y": [0.0, 1.0, 0.1, 0.9],
            "Z": [0.0, 1.0, 0.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    captured = {}

    class DummyUK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["drift_terms"] = kwargs["drift_terms"]

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    assert isinstance(captured["drift_terms"], list)
    assert captured["drift_terms"] == ["regional_linear", "point_log"]


def test_specified_drift_dict_is_converted_to_specified_drift_arrays(monkeypatch):
    # If specified_drift_arrays is None and specified_drift is a dict, values() become list
    specified = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    params = FakeUKParams(specified_drift_arrays=None, specified_drift=specified)
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid()
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.2, 0.8],
            "Y": [0.0, 1.0, 0.1, 0.9],
            "Z": [0.0, 1.0, 0.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    captured = {}

    class DummyUK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["specified_drift"] = kwargs.get("specified_drift")

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    sd = captured["specified_drift"]
    assert isinstance(sd, list)
    assert len(sd) == 2
    assert np.array_equal(sd[0], specified["a"])
    assert np.array_equal(sd[1], specified["b"])


def test_specified_drift_single_array_becomes_list(monkeypatch):
    specified = np.array([1, 2, 3])
    params = FakeUKParams(specified_drift_arrays=None, specified_drift=specified)
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid()
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.2, 0.8],
            "Y": [0.0, 1.0, 0.1, 0.9],
            "Z": [0.0, 1.0, 0.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    captured = {}

    class DummyUK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["specified_drift"] = kwargs.get("specified_drift")

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "UniversalKriging3D", DummyUK3D)

    interpolate_group_universal_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    sd = captured["specified_drift"]
    assert isinstance(sd, list)
    assert len(sd) == 1
    assert np.array_equal(sd[0], specified)


def test_external_drift_is_passed_to_constructor_and_external_drift_grid_to_execute(monkeypatch):
    params = FakeUKParams(
        external_drift=np.array([10.0, 11.0, 12.0, 13.0]),
        external_drift_grid=np.ones((2, 2, 2)),
    )
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid(resolution=(2, 2, 2))
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.2, 0.8],
            "Y": [0.0, 1.0, 0.1, 0.9],
            "Z": [0.0, 1.0, 0.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structural_modeling_components.interpolator_functions.universal_kriging_per_group as mod

    captured = {}

    class DummyUK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["constructor_external_drift"] = kwargs.get("external_drift")
