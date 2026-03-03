# tests/test_interpolate_group_rbf.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.structural_modeling_components.interpolator_functions.radial_basis_function import (
    interpolate_group_radial_basis_function)


# --- Minimal fakes used by the unit tests ------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeParams:
    def __init__(self, kernel="thin_plate_spline", smoothing=0.0, epsilon=None, neighbors=None):
        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.neighbors = neighbors


class FakeGroup:
    def __init__(self, name: str, element_names: list[str], params: FakeParams | None = None):
        self.name = name
        # IMPORTANT: function assumes group.structural_elements is youngest->oldest
        self.structural_elements = [FakeElem(n) for n in element_names]
        self._params = params or FakeParams()

    def get_interpolation_params(self):
        return self._params


class FakeGrid:
    def __init__(self, resolution=(2, 2, 2), *, use_grid_coordinates: bool):
        self.resolution = resolution
        nx, ny, nz = resolution
        self.gridx = np.arange(nx, dtype=float)
        self.gridy = np.arange(ny, dtype=float)
        self.gridz = np.arange(nz, dtype=float)

        if use_grid_coordinates:
            gx, gy, gz = np.meshgrid(self.gridx, self.gridy, self.gridz, indexing="ij")
            self.grid_coordinates = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        else:
            self.grid_coordinates = None


def surface_df(formations: list[str]) -> pd.DataFrame:
    # simple coordinates; only formations matter for scalar mapping
    n = len(formations)
    return pd.DataFrame(
        {
            "X": np.linspace(0.0, 1.0, n),
            "Y": np.linspace(0.0, 1.0, n),
            "Z": np.linspace(0.0, 1.0, n),
            "formation": formations,
        }
    )


# --- Validation / behavior tests ---------------------------------------------

def test_raises_when_surface_points_missing_or_empty():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid(use_grid_coordinates=True)

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=None)

    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=pd.DataFrame())


def test_warns_when_orientations_provided_but_unused(monkeypatch):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["young", "old"])
    odf = pd.DataFrame({"dip": [1]})

    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod

    class DummyRBF:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, points):
            return np.zeros(points.shape[0], dtype=float)

    monkeypatch.setattr(mod, "RBFInterpolator", DummyRBF)

    with pytest.warns(UserWarning, match="will not be used"):
        interpolate_group_radial_basis_function(
            group=group,
            grid=grid,
            group_surface_points_df=sdf,
            group_orientations_points_df=odf,
        )



def test_zero_elements_raises():
    """Zero structural elements must still raise ValueError (phantom workaround only handles exactly 1)."""
    group = FakeGroup("G", [])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = pd.DataFrame({"X": [0.5], "Y": [0.5], "Z": [0.5], "formation": ["orphan"]})

    with pytest.raises(ValueError):
        interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)


# --- Single-element phantom workaround tests ---------------------------------

def _dummy_rbf_class():
    """Return a DummyRBF class that accepts the RBFInterpolator interface."""
    class DummyRBF:
        def __init__(self, coords, vals, **kwargs):
            pass
        def __call__(self, points):
            return np.zeros(points.shape[0])
    return DummyRBF


def test_single_element_issues_warning(monkeypatch):
    """Single-element group warns instead of raising."""
    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod
    monkeypatch.setattr(mod, "RBFInterpolator", _dummy_rbf_class())

    group = FakeGroup("G", ["only_one"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["only_one"])

    with pytest.warns(UserWarning, match="phantom"):
        interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)


def test_single_element_phantom_not_in_returned_scalar_values(monkeypatch):
    """The synthetic phantom formation must not leak into the returned scalar_values_by_element."""
    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod
    monkeypatch.setattr(mod, "RBFInterpolator", _dummy_rbf_class())

    group = FakeGroup("G", ["only_one"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["only_one"])

    with pytest.warns(UserWarning):
        _, mapping = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)

    assert list(mapping.keys()) == ["only_one"]
    assert "__phantom__" not in mapping


def test_single_element_real_element_scalar_is_1(monkeypatch):
    """Real element keeps its normal scalar value of 1.0 (same as if it were the oldest of two)."""
    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod
    monkeypatch.setattr(mod, "RBFInterpolator", _dummy_rbf_class())

    group = FakeGroup("G", ["only_one"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["only_one"])

    with pytest.warns(UserWarning):
        _, mapping = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)

    assert mapping["only_one"] == 1.0


def test_single_element_smoke_runs_and_returns_finite_field():
    """End-to-end: single element with real SciPy RBF produces a finite scalar field."""
    group = FakeGroup("G", ["only_one"])
    grid = FakeGrid(resolution=(3, 3, 3), use_grid_coordinates=True)
    sdf = pd.DataFrame({
        "X": [0.5, 1.5, 2.5],
        "Y": [0.5, 1.5, 0.5],
        "Z": [1.0, 1.0, 1.0],
        "formation": ["only_one", "only_one", "only_one"],
    })

    with pytest.warns(UserWarning, match="phantom"):
        field, mapping = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)

    assert list(mapping.keys()) == ["only_one"]
    assert field.shape == grid.resolution
    assert np.isfinite(field).all()


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "formation"])
def test_raises_on_missing_required_columns(missing_col: str):
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["young", "old"]).drop(columns=[missing_col])

    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)


def test_scalar_values_mapping_is_oldest_1_youngest_n():
    # structural_elements assumed youngest->oldest
    group = FakeGroup("G", ["youngest", "middle", "oldest"])
    grid = FakeGrid(use_grid_coordinates=True)
    sdf = surface_df(["oldest", "middle", "youngest"])

    # Mock RBFInterpolator so we don't depend on SciPy internals for this tests.
    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod

    class DummyRBF:
        def __init__(self, coords, vals, kernel, smoothing, epsilon, neighbors):
            # capture for assertions
            self.coords = coords
            self.vals = vals
            self.kernel = kernel
            self.smoothing = smoothing
            self.epsilon = epsilon
            self.neighbors = neighbors

        def __call__(self, points):
            return np.zeros(points.shape[0], dtype=float)

    orig = mod.RBFInterpolator
    mod.RBFInterpolator = DummyRBF
    try:
        field, mapping = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)
    finally:
        mod.RBFInterpolator = orig

    assert mapping["oldest"] == 1.0
    assert mapping["middle"] == 2.0
    assert mapping["youngest"] == 3.0

    # shape should be resolution, with transpose applied
    assert field.shape == grid.resolution


def test_uses_grid_coordinates_when_available_and_transposes_output():
    group = FakeGroup("G", ["young", "old"],
                      params=FakeParams(kernel="linear", smoothing=0.25, epsilon=2.0, neighbors=7))
    grid = FakeGrid(resolution=(2, 1, 3), use_grid_coordinates=True)
    sdf = surface_df(["young", "old"])

    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod

    captured = {}

    class DummyRBF:
        def __init__(self, coords, vals, kernel, smoothing, epsilon, neighbors):
            captured["coords"] = coords
            captured["vals"] = vals
            captured["kernel"] = kernel
            captured["smoothing"] = smoothing
            captured["epsilon"] = epsilon
            captured["neighbors"] = neighbors

        def __call__(self, points):
            captured["points"] = points
            # Return predictable values 0..N-1 so we can check reshape+transpose.
            return np.arange(points.shape[0], dtype=float)

    orig = mod.RBFInterpolator
    mod.RBFInterpolator = DummyRBF
    try:
        field, _ = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)
    finally:
        mod.RBFInterpolator = orig

    # Asserts constructor wiring
    assert captured["kernel"] == "linear"
    assert captured["smoothing"] == 0.25
    assert captured["epsilon"] == 2.0
    assert captured["neighbors"] == 7

    # Must use provided grid_coordinates
    assert np.shares_memory(captured["points"], grid.grid_coordinates) or np.array_equal(
        captured["points"], grid.grid_coordinates
    )

    # Check reshape(...).T behavior:
    # scalar_flat = arange(N)
    # scalar_field = scalar_flat.reshape(resolution).T
    flat = np.arange(grid.grid_coordinates.shape[0], dtype=float)
    expected = flat.reshape(grid.resolution).T
    assert np.array_equal(field, expected)


def test_builds_grid_points_from_axes_when_grid_coordinates_missing():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid(resolution=(2, 2, 1), use_grid_coordinates=False)
    sdf = surface_df(["young", "old"])

    import core.structural_modeling_components.interpolator_functions.radial_basis_function as mod

    captured = {}

    class DummyRBF:
        def __init__(self, coords, vals, kernel, smoothing, epsilon, neighbors):
            pass

        def __call__(self, points):
            captured["points"] = points
            return np.zeros(points.shape[0], dtype=float)

    orig = mod.RBFInterpolator
    mod.RBFInterpolator = DummyRBF
    try:
        field, _ = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)
    finally:
        mod.RBFInterpolator = orig

    assert captured["points"].shape == (np.prod(grid.resolution), 3)
    assert field.shape == grid.resolution[::-1]


# --- Optional: tiny real SciPy smoke tests ------------------------------------
# Mark as slow if you want; usually it's still very fast for tiny grids.

def test_real_rbfi_smoke_runs_and_returns_correct_shape():
    group = FakeGroup("G", ["young", "old"])
    grid = FakeGrid(resolution=(2, 2, 2), use_grid_coordinates=True)

    # Two formations, multiple points each (RBF usually happier with more points)
    sdf = pd.DataFrame(
        {
            "X": [0.0, 0.2, 1.0, 0.8],
            "Y": [0.0, 0.1, 1.0, 0.9],
            "Z": [0.0, 0.0, 1.0, 1.0],
            "formation": ["old", "old", "young", "young"],
        }
    )

    field, mapping = interpolate_group_radial_basis_function(group=group, grid=grid, group_surface_points_df=sdf)

    assert mapping == {"old": 1.0, "young": 2.0}
    assert field.shape == grid.resolution
    assert np.isfinite(field).all()
