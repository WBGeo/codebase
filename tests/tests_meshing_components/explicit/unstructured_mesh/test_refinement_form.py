"""
The Workbench smart form behind create_unstructured_mesh_data's refinement
input: its declared inputs and the conversion of form data to Refinement.
Form data arrives as a JSON object of flattened keys ("root.<kind>....").
"""
import inspect
import json
import typing

import pytest

import core.meshing_components.explicit.unstructured.mesh_data as mesh_data
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    FunctionShaftRefinement,
    LinearShaftRefinement,
    LinearWellRefinement,
    TriangulationRefinement,
)


def _unwrap(func):
    return getattr(func, "__wrapped__", func)


to_data = _unwrap(getattr(mesh_data, "__internal__refined_options_to_data"))
to_form = _unwrap(getattr(mesh_data, "__internal__refined_form"))

_LINEAR = {"SizeMin": 5.0, "SizeMax": 50.0, "DistMin": 10.0, "DistMax": 100.0}


def _form(**flat):
    return json.dumps({f"root.{key}": value for key, value in flat.items()})


def test_smart_input_lists_every_form_parameter():
    """Unlisted form parameters receive unrelated mesher inputs in the Workbench
    (e.g. the tolerance arriving as `ellipses`), so the form fails to open."""
    smart_input = typing.get_args(mesh_data.SmartRefinement)[1]
    form_parameters = list(inspect.signature(to_form).parameters)
    assert list(smart_input.inputs) == form_parameters
    mesher_parameters = inspect.signature(_unwrap(mesh_data.create_unstructured_mesh_data)).parameters
    assert all(name in mesher_parameters for name in smart_input.inputs)


def test_nothing_selected_gives_none():
    assert to_data(_form(**{"well.refwell": "None", "shafts.refshafts": "None",
                            "triangulations.with_triangulations": False})) is None


def test_shaft_only_refinement_is_kept():
    data = {"shafts.refshafts": "Linear", **{f"shafts.linear.{k}": v for k, v in _LINEAR.items()}}
    result = to_data(_form(**data))
    assert isinstance(result.shafts, LinearShaftRefinement)
    assert result.shafts.SizeMin == 5.0
    assert result.wells is None and result.sources is None


def test_function_shaft_refinement_uses_the_function_type():
    result = to_data(_form(**{"shafts.refshafts": "Function", "shafts.function.expression": "5 + DIST"}))
    assert isinstance(result.shafts, FunctionShaftRefinement)
    assert result.shafts.expression == "5 + DIST"


@pytest.mark.parametrize("checked, expected", [(True, True), ("true", True), (False, False), ("false", False)])
def test_optional_groups_accept_boolean_or_string_checkbox(checked, expected):
    data = {
        "well.refwell": "Linear", **{f"well.linear.{k}": v for k, v in _LINEAR.items()},
        "triangulations.with_triangulations": checked,
        **{f"triangulations.present.{k}": v for k, v in _LINEAR.items()},
    }
    result = to_data(_form(**data))
    assert isinstance(result.wells, LinearWellRefinement)
    assert isinstance(result.triangulations, TriangulationRefinement) is expected
