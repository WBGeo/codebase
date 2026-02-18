# tests/test_interpolate_group_pli.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.structural_modeling_components.interpolator_functions.piecewise_linear import (
    interpolate_group_piecewise_linear,
)

# --- Minimal fakes used by the unit tests ------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeParams:
    def __init__(self, nelements=20_000, solver="cg", damp=True, tol=None):
        self.nelements = nelements
        self.solver = solver
        self.damp = damp
        self.tol = tol


class FakeGroup:
    def __init__(self, name: str, element_names: list[str], params: FakeParams | None = None):
        self.name = name
        # IMPORTANT: interpolator assumes group.structural_elements is youngest->oldest
        self.structural_elements = [FakeElem(n) for n in element_names]
        self._params = params or FakeParams()

    def get_interpolation_params(self):
        return self._params


class FakeGrid:
    def __init__(self, resolution=(2, 2, 2), extent=(0, 1, 0, 1, 0, 1)):
        self.resolution = resolution
        self.extent = extent
        nx, ny, nz = resolution
        gx, gy, gz = np.meshgrid(
            np.arange(nx, dtype=float),
            np.arange(ny, dtype=float),
            np.arange(nz, dtype=float),
            indexing="ij",
        )
        self.grid_coordinates = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


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


def orientations_df(formations: list[str]) -> pd.DataFrame:
    n = len(formations)
    return pd.DataFrame(
        {
            "X": np.linspace(0.0, 1.0, n),
            "Y": np.linspace(0.0, 1.0, n),
            "Z": np.linspace(0.0, 1.0, n),
            "G_x": np.ones(n, dtype=float),
            "G_y": np.zeros(n, dtype=float),
            "G_z": np.zeros(n, dtype=float),
            "formation": formations,
        }
    )


class DummyGeologicalModel:
    def __init__(self, min_bounds, max_bounds):
        self.min_bounds = np.asarray(min_bounds)
        self.max_bounds = np.asarray(max_bounds)
        self.data = None
        self.stratigraphic_column = None
        self.created = []

    def set_model_data(self, df):
        self.data = df

    def set_stratigraphic_column(self, col):
        self.stratigraphic_column = col

    def create_and_add_foliation(self, name, **kwargs):
        self.created.append((name, kwargs))
        return object()

    def evaluate_feature_value(self, feature_name, points, scale=True, **kwargs):
        return np.arange(points.shape[0], dtype=float)


# --- Validation / behavior tests ---------------------------------------------

def test_raises_when_surface_points_missing_or_empty():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=None, group_orientations_points_df=orientations_df(["old"])
        )

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=pd.DataFrame(), group_orientations_points_df=orientations_df(["old"])
        )


def test_raises_when_orientations_missing_or_empty():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"])

    with pytest.raises(ValueError, match="No orientations provided"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=None
        )

    with pytest.raises(ValueError, match="No orientations provided"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=pd.DataFrame()
        )


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "formation"])
def test_raises_on_missing_required_surface_columns(missing_col: str):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"]).drop(columns=[missing_col])
    odf = orientations_df(["young", "old"])

    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=odf
        )


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"])
def test_raises_on_missing_required_orientation_columns(missing_col: str):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()
    sdf = surface_df(["young", "old"])
    odf = orientations_df(["young", "old"]).drop(columns=[missing_col])

    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=odf
        )


def test_raises_when_df_contains_unknown_formations():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid()

    sdf = surface_df(["young", "NOT_IN_GROUP"])
    odf = orientations_df(["young", "old"])

    with pytest.raises(ValueError, match="contain formations not in the group"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=odf
        )


# --- Core tests ---------------------------------------------------------------

def test_scalar_values_mapping_is_oldest_1_youngest_n_and_transpose_behavior(monkeypatch):
    group = FakeGroup("G", ["youngest", "middle", "oldest"], params=FakeParams(nelements=33333, solver="cg", damp=True))
    grid = FakeGrid(resolution=(2, 1, 3))
    sdf = surface_df(["oldest", "middle", "youngest"])
    odf = orientations_df(["oldest", "middle", "youngest"])

    import core.structural_modeling_components.interpolator_functions.piecewise_linear as mod
    monkeypatch.setattr(mod, "GeologicalModel", DummyGeologicalModel)

    field, mapping = interpolate_group_piecewise_linear(
        group=group,
        grid=grid,
        group_surface_points_df=sdf,
        group_orientations_points_df=odf,
    )

    assert mapping["oldest"] == 1.0
    assert mapping["middle"] == 2.0
    assert mapping["youngest"] == 3.0

    assert field.shape == tuple(grid.resolution)[::-1]
    flat = np.arange(grid.grid_coordinates.shape[0], dtype=float)
    expected = flat.reshape(grid.resolution).T
    assert np.array_equal(field, expected)


def test_passes_pli_params_into_create_and_add_foliation(monkeypatch):
    group = FakeGroup("G", ["young", "old"], params=FakeParams(nelements=44444, solver="cg", damp=False, tol=1e-5))
    grid = FakeGrid(resolution=(2, 2, 1))
    sdf = surface_df(["young", "old"])
    odf = orientations_df(["young", "old"])

    import core.structural_modeling_components.interpolator_functions.piecewise_linear as mod

    captured = {}

    def factory(minb, maxb):
        m = DummyGeologicalModel(minb, maxb)
        captured["model"] = m
        return m

    monkeypatch.setattr(mod, "GeologicalModel", factory)

    _field, _mapping = interpolate_group_piecewise_linear(
        group=group,
        grid=grid,
        group_surface_points_df=sdf,
        group_orientations_points_df=odf,
    )

    model: DummyGeologicalModel = captured["model"]
    assert len(model.created) == 1
    name, kwargs = model.created[0]

    assert name == group.name
    assert kwargs["interpolatortype"] == "PLI"
    assert kwargs["nelements"] == 44444
    assert kwargs["solver"] == "cg"
    assert kwargs["damp"] is False
    assert kwargs["tol"] == 1e-5
