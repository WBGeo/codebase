# tests/test_interpolate_group_ordinary_kriging.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group import (
    interpolate_group_ordinary_kriging,
)


# --- Minimal fakes ------------------------------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeOKParams:
    def __init__(
        self,
        variogram_model="spherical",
        sill=1.0,
        range=2.0,
        nugget=0.1,
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        neighbors=None,
    ):
        self.variogram_model = variogram_model
        self.sill = sill
        self.range = range
        self.nugget = nugget
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.neighbors = neighbors


class FakeGroup:
    def __init__(self, name: str, element_names: list[str], params: FakeOKParams | None = None):
        self.name = name
        # Function assumes structural_elements are youngest->oldest
        self.structural_elements = [FakeElem(n) for n in element_names]
        self._params = params or FakeOKParams()

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


# --- Tests -------------------------------------------------------------------

def test_raises_when_surface_points_missing_or_empty():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=None)

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=pd.DataFrame())


def test_warns_when_orientations_provided_but_unused(monkeypatch):
    # This is a pure warning test -> mock kriging so it doesn't try to compute.
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"])
    odf = pd.DataFrame({"dip": [1]})

    import core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group as mod

    class DummyOK3D:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            # Return a scalar_field with same shape pykrige would give for grid (len(y), len(x), len(z)) often,
            # but we don't care here. Return something finite.
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "OrdinaryKriging3D", DummyOK3D)

    with pytest.warns(UserWarning, match="will not be used"):
        interpolate_group_ordinary_kriging(
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
        interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "formation"])
def test_raises_on_missing_required_columns(missing_col: str):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"]).drop(columns=[missing_col])

    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)


def test_raises_when_surface_points_contain_unknown_formations():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "NOT_IN_GROUP"])

    with pytest.raises(ValueError, match="contain formations not in the group"):
        interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)


def test_scalar_values_mapping_is_oldest_1_youngest_n(monkeypatch):
    # youngest->oldest order in group; function reverses so oldest=1
    group = FakeGroup("G", ["youngest", "middle", "oldest"])
    grid = FakeGrid()
    sdf = surface_df(["oldest", "middle", "youngest"])

    import core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group as mod

    class DummyOK3D:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            return np.zeros((len(grid.gridy), len(grid.gridx), len(grid.gridz))), None

    monkeypatch.setattr(mod, "OrdinaryKriging3D", DummyOK3D)

    field, mapping = interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    assert mapping == {"oldest": 1.0, "middle": 2.0, "youngest": 3.0}
    assert np.isfinite(field).all()


def test_constructs_ordinary_kriging3d_with_expected_parameters_and_calls_execute(monkeypatch):
    params = FakeOKParams(
        variogram_model="linear",
        sill=11.0,
        range=22.0,
        nugget=0.33,
        anisotropy_scaling_y=1.5,
        anisotropy_scaling_z=2.5,
        neighbors=None,
    )
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid(resolution=(3, 2, 4))
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.5, 0.2],
            "Y": [0.0, 1.0, 0.1, 0.8],
            "Z": [0.0, 1.0, 0.4, 0.9],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group as mod

    captured = {}

    class DummyOK3D:
        def __init__(self, x, y, z, vals, **kwargs):
            captured["x"] = np.asarray(x)
            captured["y"] = np.asarray(y)
            captured["z"] = np.asarray(z)
            captured["vals"] = np.asarray(vals)
            captured["kwargs"] = kwargs

        def execute(self, mode, gridx, gridy, gridz, **kwargs):
            captured["execute"] = {
                "mode": mode,
                "gridx": np.asarray(gridx),
                "gridy": np.asarray(gridy),
                "gridz": np.asarray(gridz),
                "kwargs": kwargs,
            }
            # return deterministic array
            out = np.arange(len(gridx) * len(gridy) * len(gridz), dtype=float).reshape(len(gridy), len(gridx), len(gridz))
            return out, None

    monkeypatch.setattr(mod, "OrdinaryKriging3D", DummyOK3D)

    field, mapping = interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    # Mapping should be old=1, young=2
    assert mapping == {"old": 1.0, "young": 2.0}

    # Constructor wiring
    assert captured["kwargs"]["variogram_model"] == "linear"
    assert captured["kwargs"]["variogram_parameters"] == [11.0, 22.0, 0.33]
    assert captured["kwargs"]["anisotropy_scaling_y"] == 1.5
    assert captured["kwargs"]["anisotropy_scaling_z"] == 2.5

    # Execute wiring: neighbors=None -> backend="vectorized", n_closest_points=None
    ex = captured["execute"]
    assert ex["mode"] == "grid"
    assert np.array_equal(ex["gridx"], grid.gridx)
    assert np.array_equal(ex["gridy"], grid.gridy)
    assert np.array_equal(ex["gridz"], grid.gridz)
    assert ex["kwargs"]["backend"] == "vectorized"
    assert ex["kwargs"]["n_closest_points"] is None

    assert field.shape == (len(grid.gridy), len(grid.gridx), len(grid.gridz))


def test_backend_is_loop_and_n_closest_points_is_set_when_neighbors_provided(monkeypatch):
    params = FakeOKParams(neighbors=7)
    group = FakeGroup("G", ["young", "old"], params=params)
    grid = FakeGrid(resolution=(2, 2, 2))
    sdf = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.5, 0.2],
            "Y": [0.0, 1.0, 0.1, 0.8],
            "Z": [0.0, 1.0, 0.4, 0.9],
            "formation": ["old", "old", "young", "young"],
        }
    )

    import core.structuralmodeling_components.interpolators_per_group.ordinary_kriging_per_group as mod

    captured = {}

    class DummyOK3D:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, mode, gridx, gridy, gridz, **kwargs):
            captured["backend"] = kwargs["backend"]
            captured["n_closest_points"] = kwargs["n_closest_points"]
            return np.zeros((len(gridy), len(gridx), len(gridz))), None

    monkeypatch.setattr(mod, "OrdinaryKriging3D", DummyOK3D)

    field, _ = interpolate_group_ordinary_kriging(group=group, grid=grid, group_surface_points_df=sdf)

    assert captured["backend"] == "loop"
    assert captured["n_closest_points"] == 7
    assert np.isfinite(field).all()
