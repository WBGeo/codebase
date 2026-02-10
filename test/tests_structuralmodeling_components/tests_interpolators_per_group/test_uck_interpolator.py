# tests/test_interpolate_group_universal_cokriging.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# TODO: adjust if your module path differs
from core.structuralmodeling_components.interpolator_functions.universal_cokriging_per_group import (
    interpolate_group_universal_cokriging,
    rescale_scalar_field_safe,
)


# --- Minimal fakes ------------------------------------------------------------

class FakeElem:
    def __init__(self, name: str):
        self.name = name


class FakeGroup:
    def __init__(self, name: str, element_names: list[str]):
        self.name = name
        self.structural_elements = [FakeElem(n) for n in element_names]


class FakeGrid:
    def __init__(self, extent=(0, 1, 0, 1, 0, 1), resolution=(2, 2, 1)):
        self.extent = extent
        self.resolution = resolution


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


def ori_df(formations: list[str]) -> pd.DataFrame:
    n = len(formations)
    return pd.DataFrame(
        {
            "X": np.linspace(0.0, 1.0, n),
            "Y": np.linspace(0.0, 1.0, n),
            "Z": np.linspace(0.0, 1.0, n),
            "G_x": np.ones(n),
            "G_y": np.zeros(n),
            "G_z": np.zeros(n),
            "formation": formations,
        }
    )


# --- Interpolator validation tests -------------------------------------------

def test_raises_when_surface_points_missing_or_empty():
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_universal_cokriging(
            group=group, grid=grid, group_surface_points_df=None, group_orientations_points_df=ori_df(["a"])
        )
    with pytest.raises(ValueError, match="No surface points provided"):
        interpolate_group_universal_cokriging(
            group=group, grid=grid, group_surface_points_df=pd.DataFrame(), group_orientations_points_df=ori_df(["a"])
        )


def test_raises_when_orientations_missing_or_empty():
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    with pytest.raises(ValueError, match="No orientations provided"):
        interpolate_group_universal_cokriging(
            group=group, grid=grid, group_surface_points_df=surface_df(["a"]), group_orientations_points_df=None
        )
    with pytest.raises(ValueError, match="No orientations provided"):
        interpolate_group_universal_cokriging(
            group=group, grid=grid, group_surface_points_df=surface_df(["a"]), group_orientations_points_df=pd.DataFrame()
        )


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "formation"])
def test_surface_points_missing_required_columns(missing_col: str):
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    sp = surface_df(["a"]).drop(columns=[missing_col])
    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_universal_cokriging(group=group, grid=grid, group_surface_points_df=sp, group_orientations_points_df=ori_df(["a"]))


@pytest.mark.parametrize("missing_col", ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"])
def test_orientations_missing_required_columns(missing_col: str):
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    od = ori_df(["a"]).drop(columns=[missing_col])
    with pytest.raises(ValueError, match=f"missing column '{missing_col}'"):
        interpolate_group_universal_cokriging(group=group, grid=grid, group_surface_points_df=surface_df(["a"]), group_orientations_points_df=od)


def test_raises_on_unknown_formations_in_surface_points():
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    sp = surface_df(["a", "NOT_IN_GROUP"])
    with pytest.raises(ValueError, match="contain formations not in the group"):
        interpolate_group_universal_cokriging(group=group, grid=grid, group_surface_points_df=sp, group_orientations_points_df=ori_df(["a"]))


def test_raises_on_unknown_formations_in_orientations():
    group = FakeGroup("G", ["a", "b"])
    grid = FakeGrid()
    od = ori_df(["a", "NOT_IN_GROUP"])
    with pytest.raises(ValueError, match="contain formations not in the group"):
        interpolate_group_universal_cokriging(group=group, grid=grid, group_surface_points_df=surface_df(["a"]), group_orientations_points_df=od)


# --- GemPy integration mocked (true unit tests) -------------------------------

def test_happy_path_calls_gempy_and_rescales(monkeypatch):
    """
    We mock GemPy to avoid running a real model. We also mock rescale_scalar_field_safe
    so we can assert the scalar_field passed in has the expected transpose/reshape.
    """
    group = FakeGroup("G", ["e1", "e2", "e3"])
    grid = FakeGrid(extent=(0, 10, 0, 10, 0, 10), resolution=(2, 2, 1))

    sp = surface_df(["e1", "e2", "e3"])
    od = ori_df(["e1", "e2", "e3"])

    import core.structuralmodeling_components.interpolator_functions.universal_cokriging_per_group as mod

    # ---- capture calls ----
    captured = {}

    # ---- fake GemPy tables ----
    class FakeSurfaceTable:
        def __init__(self):
            self.name_id_map = {"dummy": 1}

    class FakeOrientationsTable:
        pass

    def fake_surface_from_arrays(**kwargs):
        captured["surface_from_arrays"] = kwargs
        return FakeSurfaceTable()

    def fake_ori_from_arrays(**kwargs):
        captured["ori_from_arrays"] = kwargs
        return FakeOrientationsTable()

    def fake_structural_frame_from_tables(surface_data, orientation_data):
        captured["structural_frame"] = (surface_data, orientation_data)
        return "FAKE_STRUCTURAL_FRAME"

    # ---- fake geo model / results ----
    class FakeRawArrays:
        def __init__(self):
            # must match len(group_elem_names)==3
            self.scalar_field_at_surface_points = [np.array([10.0, 20.0, 30.0])]
            # scalar_field_matrix[0] must have prod(resolution)=4 values
            self.scalar_field_matrix = [np.array([0.0, 1.0, 2.0, 3.0])]

    class FakeSolutions:
        def __init__(self):
            self.raw_arrays = FakeRawArrays()

    class FakeGeoModel:
        def __init__(self):
            self.solutions = FakeSolutions()

    def fake_create_geomodel(project_name, extent, resolution, structural_frame):
        captured["create_geomodel"] = {
            "project_name": project_name,
            "extent": extent,
            "resolution": resolution,
            "structural_frame": structural_frame,
        }
        return FakeGeoModel()

    def fake_map_stack_to_surfaces(gempy_model, mapping_object):
        captured["map_stack_to_surfaces"] = {"model": gempy_model, "mapping": mapping_object}

    def fake_compute_model(model):
        captured["compute_model_called"] = True

    # ---- mock rescale to assert inputs ----
    def fake_rescale_scalar_field_safe(scalar_field, scalar_values_by_element):
        captured["rescale_in"] = {
            "scalar_field": np.array(scalar_field, copy=True),
            "scalar_values": dict(scalar_values_by_element),
        }
        # return something distinguishable
        return scalar_field + 100.0, {k: float(i + 1) for i, k in enumerate(sorted(scalar_values_by_element.keys()))}

    # Patch the gempy entrypoints used by the function
    monkeypatch.setattr(mod.gp.data.surface_points.SurfacePointsTable, "from_arrays", staticmethod(fake_surface_from_arrays))
    monkeypatch.setattr(mod.gp.data.orientations.OrientationsTable, "from_arrays", staticmethod(fake_ori_from_arrays))
    monkeypatch.setattr(mod.gp.data.structural_frame.StructuralFrame, "from_data_tables", staticmethod(fake_structural_frame_from_tables))
    monkeypatch.setattr(mod.gp, "create_geomodel", fake_create_geomodel)
    monkeypatch.setattr(mod.gp, "map_stack_to_surfaces", fake_map_stack_to_surfaces)
    monkeypatch.setattr(mod.gp, "compute_model", fake_compute_model)
    monkeypatch.setattr(mod, "rescale_scalar_field_safe", fake_rescale_scalar_field_safe)

    field, values = interpolate_group_universal_cokriging(
        group=group,
        grid=grid,
        group_surface_points_df=sp,
        group_orientations_points_df=od,
    )

    # GemPy create_geomodel should be called with list(...) conversions
    assert captured["create_geomodel"]["extent"] == list(grid.extent)
    assert captured["create_geomodel"]["resolution"] == list(grid.resolution)

    # Mapping must map group.name -> list of all element names
    assert captured["map_stack_to_surfaces"]["mapping"] == {group.name: ["e1", "e2", "e3"]}

    assert captured["compute_model_called"] is True

    # Check the raw scalar values were zipped to elements before rescale
    assert captured["rescale_in"]["scalar_values"] == {"e1": 10.0, "e2": 20.0, "e3": 30.0}

    # Check reshape + transpose before rescale:
    # scalar_field_matrix = [0,1,2,3], resolution=(2,2,1) -> reshape(2,2,1).T -> (1,2,2)
    expected_pre_rescale = np.array([0.0, 1.0, 2.0, 3.0]).reshape(grid.resolution).T
    assert np.array_equal(captured["rescale_in"]["scalar_field"], expected_pre_rescale)

    # Our fake rescale adds +100
    assert np.array_equal(field, expected_pre_rescale + 100.0)
    assert isinstance(values, dict)


def test_raises_if_gempy_returns_wrong_number_of_scalar_values(monkeypatch):
    group = FakeGroup("G", ["e1", "e2", "e3"])
    grid = FakeGrid(resolution=(2, 2, 1))
    sp = surface_df(["e1", "e2", "e3"])
    od = ori_df(["e1", "e2", "e3"])

    import core.structuralmodeling_components.interpolator_functions.universal_cokriging_per_group as mod

    class FakeSurfaceTable:
        def __init__(self):
            self.name_id_map = {"dummy": 1}

    class FakeOrientationsTable:
        pass

    monkeypatch.setattr(mod.gp.data.surface_points.SurfacePointsTable, "from_arrays", staticmethod(lambda **kw: FakeSurfaceTable()))
    monkeypatch.setattr(mod.gp.data.orientations.OrientationsTable, "from_arrays", staticmethod(lambda **kw: FakeOrientationsTable()))
    monkeypatch.setattr(mod.gp.data.structural_frame.StructuralFrame, "from_data_tables", staticmethod(lambda a, b: "FRAME"))

    class FakeRawArrays:
        def __init__(self):
            # WRONG length: 2 vs 3 elements
            self.scalar_field_at_surface_points = [np.array([10.0, 20.0])]
            self.scalar_field_matrix = [np.array([0.0, 1.0, 2.0, 3.0])]

    class FakeGeoModel:
        def __init__(self):
            self.solutions = type("S", (), {"raw_arrays": FakeRawArrays()})()

    monkeypatch.setattr(mod.gp, "create_geomodel", lambda **kw: FakeGeoModel())
    monkeypatch.setattr(mod.gp, "map_stack_to_surfaces", lambda **kw: None)
    monkeypatch.setattr(mod.gp, "compute_model", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="GemPy returned 2 scalar values"):
        interpolate_group_universal_cokriging(
            group=group, grid=grid, group_surface_points_df=sp, group_orientations_points_df=od
        )


# --- Pure function tests for rescale_scalar_field_safe ------------------------

def test_rescale_scalar_field_safe_n1_is_shift_to_one():
    S = np.array([[10.0, 11.0]])
    vals = {"only": 10.0}
    S_new, vals_new = rescale_scalar_field_safe(S, vals)
    assert np.allclose(S_new, S + (1.0 - 10.0))
    assert vals_new == {"only": 1.0}


def test_rescale_scalar_field_safe_maps_surfaces_to_exact_integers():
    # Two surfaces at 10 and 30 should map to 1 and 2; midpoint 20 should map to 1.5
    S = np.array([10.0, 20.0, 30.0])
    vals = {"a": 10.0, "b": 30.0}
    S_new, vals_new = rescale_scalar_field_safe(S, vals)

    assert vals_new == {"a": 1.0, "b": 2.0}
    assert np.allclose(S_new, np.array([1.0, 1.5, 2.0]))
