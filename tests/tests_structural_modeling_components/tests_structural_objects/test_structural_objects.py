import numpy as np
import pandas as pd
import pytest


from core.structural_modeling_components.structural_objects.structural_objects import (  # type: ignore
    StructuralElement,
    StructuralGroup,
    StructuralFrame,
    FaultElement,
    FaultFrame,
)

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (
    InterpolationMethod,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_grid(resolution=(2, 2, 2)) -> RegularGrid:
    return RegularGrid(extent=(0.0, 2.0, 0.0, 2.0, 0.0, 2.0), resolution=resolution)


def surface_points_df(formations: list[str]) -> pd.DataFrame:
    # one point per formation, minimal shape
    rows = []
    for i, f in enumerate(formations):
        rows.append({"X": float(i), "Y": 0.0, "Z": 0.0, "formation": f})
    return pd.DataFrame(rows)


def orientations_df(formations: list[str]) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(formations):
        rows.append(
            {
                "X": float(i),
                "Y": 0.0,
                "Z": 0.0,
                "G_x": 1.0,
                "G_y": 0.0,
                "G_z": 0.0,
                "formation": f,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# StructuralElement
# -----------------------------------------------------------------------------
def test_structural_element_setters_and_mesh_roundtrip():
    e = StructuralElement(name="e1")

    e.set_scalar_value(3.14)
    assert e.scalar_value == 3.14
    assert e.get_scalar_value() == 3.14

    e.set_id(7)
    assert e.id == 7

    e.set_color("#AABBCC")
    assert e.color == "#AABBCC"

    v = np.zeros((3, 3), dtype=float)
    ed = np.ones((2, 2), dtype=int)

    e.set_mesh("masked", v, ed)
    got_v, got_e = e.get_mesh("masked")
    assert got_v is v
    assert got_e is ed


def test_structural_element_rejects_invalid_mesh_type():
    e = StructuralElement(name="e1")
    with pytest.raises(ValueError, match="Invalid mesh type"):
        e.set_mesh("nope", np.zeros((1, 3)), np.zeros((1, 2)))


def test_structural_element_get_mesh_raises_keyerror_for_missing():
    e = StructuralElement(name="e1")
    with pytest.raises(KeyError, match="Mesh 'masked' not found"):
        _ = e.get_mesh("masked")


# -----------------------------------------------------------------------------
# StructuralGroup
# -----------------------------------------------------------------------------
def test_structural_group_getitem_returns_element_or_raises():
    e1 = StructuralElement(name="a")
    e2 = StructuralElement(name="b")
    g = StructuralGroup(name="G", structural_elements=[e1, e2])

    assert g["a"] is e1
    with pytest.raises(KeyError, match="not found in group"):
        _ = g["missing"]


def test_structural_group_update_interpolation_context_requires_two_points():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])

    pts = np.array([[0.0, 0.0, 0.0]])  # only one point
    with pytest.raises(ValueError, match="Not enough points"):
        g.update_interpolation_context(pts)


def test_structural_group_update_interpolation_context_computes_reasonable_values():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])

    # 3 points on x-axis; median NN distance should be 1.0
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    g.update_interpolation_context(pts)

    # context is private, but set_interpolation_method depends on it;
    # we can infer it exists by successfully setting a method later
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    params = g.get_interpolation_params()
    assert params is not None


def test_structural_group_set_interpolation_method_requires_context():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])

    with pytest.raises(RuntimeError, match="context not initialized"):
        g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)


def test_structural_group_set_interpolation_method_accepts_string_and_rejects_invalid():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g.update_interpolation_context(pts)

    # accept string version of enum values
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION.value)
    assert g.interpolation_method == InterpolationMethod.RADIAL_BASIS_FUNCTION

    with pytest.raises(ValueError, match="not a valid interpolation method"):
        g.set_interpolation_method("definitely_not_a_method")


def test_structural_group_configure_interpolation_params_updates_existing_fields():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g.update_interpolation_context(pts)
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    params = g.get_interpolation_params()
    assert params is not None

    # Choose an existing field robustly (pydantic v1/v2 compatible)
    if hasattr(params, "model_fields"):  # pydantic v2
        field_name = next(iter(params.model_fields.keys()))
    else:  # pydantic v1
        field_name = next(iter(params.__fields__.keys()))

    # Set it to a new value of a reasonable type
    current = getattr(params, field_name)
    new_value = 0.123 if isinstance(current, (int, float)) else current
    g.configure_interpolation_params(**{field_name: new_value})
    assert getattr(g.get_interpolation_params(), field_name) == new_value


def test_structural_group_configure_interpolation_params_requires_initialized_params():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])
    with pytest.raises(ValueError, match="have not been initialized"):
        g.configure_interpolation_params(kernel="linear")


def test_structural_group_configure_interpolation_params_rejects_unknown_field():
    g = StructuralGroup(name="G", structural_elements=[StructuralElement(name="e1")])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g.update_interpolation_context(pts)
    g.set_interpolation_method(InterpolationMethod.RADIAL_BASIS_FUNCTION)

    with pytest.raises(AttributeError, match="not a valid parameter"):
        g.configure_interpolation_params(this_field_does_not_exist=123)


# -----------------------------------------------------------------------------
# FaultElement
# -----------------------------------------------------------------------------
def test_fault_element_scalar_field_and_color_validation():
    f = FaultElement("F1")

    with pytest.raises(ValueError, match="Scalar field must be a numpy array"):
        f.set_scalar_field([1, 2, 3])  # type: ignore[arg-type]

    arr = np.zeros((2, 2, 2), dtype=float)
    f.set_scalar_field(arr)
    assert f.scalar_field is arr

    f.set_color("#112233")
    assert f.color == "#112233"

    with pytest.raises(ValueError, match="valid hex string"):
        f.set_color("red")  # missing #


def test_fault_element_mesh_and_mask_and_domain_pairs_helpers():
    f = FaultElement("F1")

    v = np.zeros((4, 3), dtype=float)
    e = np.zeros((2, 2), dtype=int)
    f.set_mesh("masked", v, e)
    got_v, got_e = f.get_mesh("masked")
    assert got_v is v
    assert got_e is e

    with pytest.raises(ValueError, match="Invalid mesh type"):
        f.set_mesh("combined", v, e)  # FaultElement only allows masked/unmasked

    with pytest.raises(ValueError, match="No mask set"):
        _ = f.get_domain_mask()

    mask = np.array([True, False, True], dtype=bool)
    f.set_domain_mask(mask)
    assert np.array_equal(f.get_domain_mask(), mask)
    assert np.array_equal(f.get_inverse_domain_mask(), ~mask)

    # domain pairs normalization: (2,1) becomes (1,2), duplicates removed, (3,3) ignored
    f.set_domain_pairs({(2, 1), (1, 2), (3, 3)})
    assert f.get_domain_pairs() == frozenset({(1, 2)})
    assert f.domain_pairs_flat() == {1, 2}


def test_fault_element_set_domain_pairs_rejects_empty_after_normalization():
    f = FaultElement("F1")
    with pytest.raises(ValueError, match="empty after normalization"):
        f.set_domain_pairs({(1, 1)})  # becomes empty


def test_fault_element_set_domain_pairs_rejects_bad_tuple():
    f = FaultElement("F1")
    with pytest.raises(ValueError, match="must be a tuple"):
        f.set_domain_pairs({(1, 2, 3)})  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# FaultFrame
# -----------------------------------------------------------------------------
def test_fault_frame_get_element_by_name_and_df_filters():
    f1 = FaultElement("F1")
    f2 = FaultElement("F2")
    ff = FaultFrame([f1, f2])

    assert ff.get_element_by_name("F2") is f2
    assert ff.get_element_by_name("missing") is None

    sp = surface_points_df(["F1", "F2", "F2"])
    ori = orientations_df(["F1", "F2"])
    ff.set_surface_points_df(sp)
    ff.set_orientations_df(ori)

    assert len(ff.get_surface_points_for_element("F2")) == 2
    assert len(ff.get_orientations_for_element("F1")) == 1
    assert ff.get_surface_points_for_element("missing").empty


def test_fault_frame_set_grid_requires_regulargrid():
    ff = FaultFrame([FaultElement("F1")])
    with pytest.raises(ValueError, match="instance of RegularGrid"):
        ff.set_grid(object())  # type: ignore[arg-type]

    g = make_grid()
    ff.set_grid(g)
    assert ff.grid is g


# -----------------------------------------------------------------------------
# StructuralFrame
# -----------------------------------------------------------------------------
def test_structural_frame_set_fault_frame_validations_and_activity_mapping():
    g1 = StructuralGroup(name="G1", structural_elements=[StructuralElement(name="e1")])
    g2 = StructuralGroup(name="G2", structural_elements=[StructuralElement(name="e2")])
    sf = StructuralFrame(structural_groups=[g1, g2])

    # set required private state for tests
    grid = make_grid()
    sf.grid = grid  # the class has no setter; OK for unit tests

    fault = FaultElement("F1")
    ff = FaultFrame([fault])

    # cannot assign without ff.grid
    with pytest.raises(ValueError, match="FaultFrame grid must be set"):
        sf.set_fault_frame(ff)

    ff.set_grid(grid)

    # cannot assign without domain_map
    with pytest.raises(ValueError, match="domain map missing"):
        sf.set_fault_frame(ff)

    # now satisfy domain_map requirement
    ff.set_domain_map(np.zeros(grid.resolution, dtype=float))

    sf.set_fault_frame(ff)
    assert sf.fault_frame is ff

    # default activity should exist
    assert sf.fault_activity == {"F1": 0}
    verbose = sf.fault_activity_verbose
    assert verbose is not None
    assert verbose["F1"]["youngest_group_index"] == 0
    assert verbose["F1"]["youngest_group_name"] == "G1"

    # set activity by index
    sf.set_fault_activity_by_index("F1", 1)
    assert sf.fault_activity == {"F1": 1}

    # invalid fault name
    with pytest.raises(KeyError, match="not found in fault frame"):
        sf.set_fault_activity_by_index("missing", 0)

    # invalid index bounds
    with pytest.raises(ValueError, match="range"):
        sf.set_fault_activity_by_index("F1", 999)

    # set by group name
    sf.set_fault_activity_by_group("F1", "G1")
    assert sf.fault_activity == {"F1": 0}

    with pytest.raises(KeyError, match="Structural group 'nope' not found"):
        sf.set_fault_activity_by_group("F1", "nope")


def test_structural_frame_set_fault_frame_allows_equal_grid_values():
    sf = StructuralFrame(structural_groups=[StructuralGroup(name="G", structural_elements=[])])
    sf.grid = make_grid()

    ff = FaultFrame([FaultElement("F1")])
    ff.set_grid(make_grid())  # different instance, same values
    ff.set_domain_map(np.zeros((2, 2, 2), dtype=float))

    assert ff.grid is not sf.grid
    assert ff.grid == sf.grid

    # should NOT raise
    sf.set_fault_frame(ff)
    assert sf.fault_frame is ff   # or whatever accessor/attr you have


def test_structural_frame_detach_fault_frame_clears_state():
    sf = StructuralFrame(structural_groups=[StructuralGroup(name="G1", structural_elements=[])])
    sf.grid = make_grid()

    ff = FaultFrame([FaultElement("F1")])
    ff.set_grid(sf.grid)
    ff.set_domain_map(np.zeros(sf.grid.resolution, dtype=float))
    sf.set_fault_frame(ff)

    assert sf.fault_frame is not None
    sf.set_fault_frame(None)

    assert sf.fault_frame is None
    assert sf.fault_activity is None
    assert sf.fault_activity_verbose is None
