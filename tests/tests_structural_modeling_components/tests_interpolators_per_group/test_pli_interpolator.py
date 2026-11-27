# tests/test_interpolate_group_pli.py

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.structural_modeling_components.interpolator_functions.piecewise_linear import (
    _strike_and_dip_vectors,
    interpolate_group_piecewise_linear,
)

# --- Minimal fakes used by the unit tests ------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeParams:
    def __init__(self, nelements=20_000, regularisation=1.0, orientation_weight=1.0):
        self.nelements = nelements
        self.regularisation = regularisation
        self.orientation_weight = orientation_weight


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


class DummyBuilder:
    """Stand-in for the LoopStructural feature builder (mean element volume 0.5)."""

    def __init__(self):
        self.interpolator = SimpleNamespace(support=SimpleNamespace(element_size=np.array([0.25, 0.75])))
        self.build_arguments = {}

    def update_build_arguments(self, arguments):
        self.build_arguments.update(arguments)


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
        self.feature = SimpleNamespace(builder=DummyBuilder())
        return self.feature

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


def test_raises_when_interpolation_params_missing(monkeypatch):
    group = FakeGroup("G", ["young", "old"])
    group._params = None
    grid = FakeGrid()

    sdf = surface_df(["young", "old"])
    odf = orientations_df(["young", "old"])

    import core.structural_modeling_components.interpolator_functions.piecewise_linear as mod
    monkeypatch.setattr(mod, "GeologicalModel", DummyGeologicalModel)

    with pytest.raises(ValueError, match="Missing interpolation params"):
        interpolate_group_piecewise_linear(
            group=group, grid=grid, group_surface_points_df=sdf, group_orientations_points_df=odf
        )


# --- Core tests ---------------------------------------------------------------

def test_scalar_values_mapping_is_oldest_1_youngest_n_and_transpose_behavior(monkeypatch):
    group = FakeGroup("G", ["youngest", "middle", "oldest"], params=FakeParams(nelements=33333, regularisation=1.0))
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
    group = FakeGroup("G", ["young", "old"], params=FakeParams(nelements=44444, regularisation=0.5, orientation_weight=3.0))
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
    assert kwargs["regularisation"] == 0.5
    assert "gpw" not in kwargs  # PLI has no gradient constraint: orientations go in as tangents
    assert "damp" not in kwargs and "tol" not in kwargs
    # orientation weight divided by the mean element volume (0.5)
    assert model.feature.builder.build_arguments["tpw"] == pytest.approx(6.0)


def test_orientations_become_tangents_without_value(monkeypatch):
    """Each orientation: one gradient row and two tangent rows (strike, dip), none with a value."""
    group = FakeGroup("G", ["young", "old"])
    sdf = surface_df(["young", "old"])
    odf = orientations_df(["young", "old"])

    import core.structural_modeling_components.interpolator_functions.piecewise_linear as mod

    captured = {}

    def factory(minb, maxb):
        m = DummyGeologicalModel(minb, maxb)
        captured["model"] = m
        return m

    monkeypatch.setattr(mod, "GeologicalModel", factory)
    interpolate_group_piecewise_linear(
        group=group, grid=FakeGrid(), group_surface_points_df=sdf, group_orientations_points_df=odf,
    )

    data = captured["model"].data
    is_gradient = data["gx"].notna()
    is_tangent = data["tx"].notna()
    assert is_gradient.sum() == 2 and is_tangent.sum() == 4
    assert data.loc[is_gradient | is_tangent, "val"].isna().all()
    assert data.loc[~(is_gradient | is_tangent), "val"].tolist() == [2.0, 1.0]  # young=2, old=1
    # orientations in the test data have normal (1, 0, 0): tangents must be perpendicular
    assert np.allclose(data.loc[is_tangent, "tx"], 0.0)


def test_strike_and_dip_vectors_are_perpendicular_unit_vectors():
    normals = np.array([[0.0, 0.0, 1.0], [-0.25, 0.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, -3.0]])
    strike, dip = _strike_and_dip_vectors(normals)
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    for v in (strike, dip):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
        assert np.allclose(np.sum(v * unit, axis=1), 0.0)
    assert np.allclose(np.sum(strike * dip, axis=1), 0.0)


class _FaceNaNModel:
    """Like LoopStructural's PLI: NaN exactly on the mesh face x = 0.5 and outside [0, 1]^3, else x + y + z."""

    def __init__(self):
        self.calls = 0

    def evaluate_feature_value(self, name, points, scale=True):
        self.calls += 1
        p = np.asarray(points, dtype=float)
        values = p.sum(axis=1)
        values[(p[:, 0] == 0.5) | np.any((p < 0.0) | (p > 1.0), axis=1)] = np.nan
        return values


def test_points_on_a_support_mesh_face_are_evaluated_minimally_shifted():
    from core.structural_modeling_components.interpolator_functions.fitted_field import (
        LoopStructuralFittedField,
        evaluate_loopstructural_feature,
    )

    points = np.array([[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 0.25], [2.0, 0.5, 0.5]])
    model = _FaceNaNModel()
    values = evaluate_loopstructural_feature(model, "G", points, scale=True)

    np.testing.assert_allclose(values[:3], points[:3].sum(axis=1), atol=1e-8)  # face points filled
    assert np.isnan(values[3])                                                 # outside stays NaN
    assert model.calls == 3  # once, then shifted one way and the other for the point outside
    # the octree's fitted field uses the same guard
    field = LoopStructuralFittedField(_FaceNaNModel(), "G", True, {"a": 1.0})
    np.testing.assert_allclose(field.evaluate(points[:3]), points[:3].sum(axis=1), atol=1e-8)


def test_grid_evaluation_has_no_nan_on_a_support_mesh_face(monkeypatch):
    """interpolate_group_piecewise_linear fills cell centers that lie on a support mesh face."""
    import core.structural_modeling_components.interpolator_functions.piecewise_linear as pli

    monkeypatch.setattr(pli, "_fit_pli_model", lambda *args: (_FaceNaNModel(), {"a": 1.0}))
    grid = FakeGrid(resolution=(3, 2, 2))
    grid.grid_coordinates = grid.grid_coordinates * 0.25  # x = 0, 0.25, 0.5: one plane on the face
    field, _ = pli.interpolate_group_piecewise_linear(
        group=FakeGroup("G", ["a"]), grid=grid,
        group_surface_points_df=surface_df(["a"]), group_orientations_points_df=orientations_df(["a"]))
    assert not np.isnan(field).any()
