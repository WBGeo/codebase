import json

import pytest
from pydantic_core import to_jsonable_python

from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    CustomSfepyBuilder,
    map_mat_id_to_lithology,
)


def _valid_input_text(mesh_results):
    """A minimal, syntactically-plausible SfePy-input-file-shaped text that
    references every real group on mesh_results -- enough for
    CustomSfepyBuilder's static validation to accept."""
    valid = map_mat_id_to_lithology(mesh_results)
    lines = "\n".join(f"    'Omega{m}': 'cells of group {m}'," for m in valid)
    return f"regions = {{\n{lines}\n}}\n"


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

def test_construction_basic(model1_implicit_mesh, model1_structural_result):
    builder = CustomSfepyBuilder(
        input_file_contents=_valid_input_text(model1_implicit_mesh),
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    assert builder.mesh_type == model1_implicit_mesh.mesh_type
    assert builder.mat_id_to_lithology == map_mat_id_to_lithology(model1_implicit_mesh)


def test_construction_without_geomodel_result_is_allowed(model1_implicit_mesh):
    builder = CustomSfepyBuilder(
        input_file_contents=_valid_input_text(model1_implicit_mesh),
        mesh_results=model1_implicit_mesh,
    )
    assert builder.geomodel_result is None


def test_rejects_mesh_with_no_type(model1_implicit_mesh):
    bad_mesh = model1_implicit_mesh.model_copy(update={"mesh_type": None})
    with pytest.raises(ValueError, match="mesh_type is None"):
        CustomSfepyBuilder(input_file_contents=_valid_input_text(model1_implicit_mesh), mesh_results=bad_mesh)


def test_rejects_mesh_with_no_lithology_mapping(model1_implicit_mesh):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})
    with pytest.raises(ValueError, match="no lithology mapping"):
        CustomSfepyBuilder(input_file_contents=_valid_input_text(model1_implicit_mesh), mesh_results=bad_mesh)


def test_rejects_file_referencing_unknown_group(model1_implicit_mesh):
    with pytest.raises(ValueError, match="do not exist"):
        CustomSfepyBuilder(
            input_file_contents="'Omega99': 'cells of group 99',",
            mesh_results=model1_implicit_mesh,
        )


def test_geomodel_soft_check_does_not_raise_for_valid_case(model1_implicit_mesh, model1_structural_result):
    # Doesn't raise even though this checks only logs a warning when the
    # counts look off -- construction must always succeed here.
    builder = CustomSfepyBuilder(
        input_file_contents=_valid_input_text(model1_implicit_mesh),
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    assert builder is not None


def test_custom_builder_survives_json_roundtrip(model1_implicit_mesh, model1_structural_result):
    """Mirrors test_builder_survives_json_roundtrip for HydrothermalProblemBuilder:
    input_file_contents is stored as text specifically so it survives the
    real Workbench backend's serialize/reconstruct cycle between component
    executions (a temp file path would not)."""
    builder = CustomSfepyBuilder(
        input_file_contents=_valid_input_text(model1_implicit_mesh),
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    as_json = json.dumps(to_jsonable_python(builder))
    rebuilt = CustomSfepyBuilder(**json.loads(as_json))

    assert rebuilt.input_file_contents == builder.input_file_contents
    assert rebuilt.mesh_type == builder.mesh_type
    assert rebuilt.mat_id_to_lithology == builder.mat_id_to_lithology
