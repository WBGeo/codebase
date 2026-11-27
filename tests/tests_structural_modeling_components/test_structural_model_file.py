"""Save and load structural models in the native .wbgeo_sm format (structural_model_file.py)."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, TypeAdapter

from core.object_components import StructuralModelResults
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.structural_modeling_components import structural_model_file as smf
from core.structural_modeling_components.structural_objects.structural_objects import (
    PARAMS_CLASS_BY_METHOD,
    StructuralGroup,
)


def _leaves(obj, path="model", out=None):
    """All arrays, DataFrames and plain values of a result, keyed by their path."""
    out = {} if out is None else out
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        out[path] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _leaves(value, f"{path}[{key}]", out)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _leaves(value, f"{path}[{i}]", out)
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            _leaves(getattr(obj, name), f"{path}.{name}", out)
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            _leaves(getattr(obj, name), f"{path}.{name}", out)
    elif obj is not None:
        out[path] = obj
    return out


@pytest.mark.parametrize("model_fixture", ["small_model1", "small_model2", "small_model2_octree"])
def test_round_trip_keeps_every_value(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    loaded = smf.structural_model_from_bytes(smf.structural_model_to_bytes(model))

    assert isinstance(loaded, StructuralModelResults)
    before, after = _leaves(model), _leaves(loaded)
    assert before.keys() == after.keys()
    for path, value in before.items():
        other = after[path]
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other), path
            # dtypes follow the declared field types; integer data stays integer
            assert value.dtype.kind == other.dtype.kind, path
        elif isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, other, obj=path)
        else:
            assert value == other, path


def test_round_trip_keeps_lithology_faults_and_order(small_model2):
    loaded = smf.structural_model_from_bytes(smf.structural_model_to_bytes(small_model2))
    frame, frame_loaded = small_model2.structural_frame, loaded.structural_frame

    assert frame_loaded.lith_block.dtype == np.int32
    assert frame_loaded.fault_activity == frame.fault_activity == {"fault": 1}
    assert [g.name for g in frame_loaded.structural_groups] == ["Strat_Series2", "Strat_Series1"]
    assert [e.name for g in frame_loaded.structural_groups for e in g.structural_elements] == \
        ["rock4", "rock3", "rock2", "rock1"]
    fault, fault_loaded = frame.fault_frame.fault_elements[0], frame_loaded.fault_frame.fault_elements[0]
    assert fault_loaded.separated_domains == fault.separated_domains
    assert fault_loaded.domain_pairs == fault.domain_pairs
    assert set(frame_loaded.fault_frame.domain_masks) == set(frame.fault_frame.domain_masks)


def test_loaded_model_gives_the_same_implicit_mesh(small_model2):
    loaded = smf.structural_model_from_bytes(smf.structural_model_to_bytes(small_model2))
    mesh, mesh_loaded = create_implicit_structured_mesh(small_model2), create_implicit_structured_mesh(loaded)

    assert np.array_equal(mesh.nodes, mesh_loaded.nodes)
    assert len(mesh.elements) == len(mesh_loaded.elements)
    for block, block_loaded in zip(mesh.elements, mesh_loaded.elements):
        assert block.type == block_loaded.type
        assert np.array_equal(block.data, block_loaded.data)


@pytest.mark.parametrize("method, params_class", list(PARAMS_CLASS_BY_METHOD.items()))
def test_interpolation_params_keep_their_class(method, params_class):
    # several parameter classes share the same fields (e.g. PLI/FDI, UCK/OK)
    group = StructuralGroup(name="g", interpolation_method=method, interpolation_params=params_class())
    adapter = TypeAdapter(StructuralGroup)
    for loaded in (adapter.validate_json(adapter.dump_json(group)), adapter.validate_python(group.model_dump())):
        assert type(loaded.interpolation_params) is params_class
        assert loaded.interpolation_params == group.interpolation_params


def test_save_and_load_files(small_model1, tmp_path):
    path = smf.save_structural_model(small_model1, tmp_path / "small_model1")
    assert path.name == "small_model1.wbgeo_sm"
    loaded = smf.load_structural_model(path)
    assert np.array_equal(loaded.structural_frame.lith_block, small_model1.structural_frame.lith_block)
    # an existing extension is not doubled
    assert smf.save_structural_model(small_model1, tmp_path / "again.wbgeo_sm").name == "again.wbgeo_sm"


def test_file_layout_and_manifest(small_model1):
    with zipfile.ZipFile(io.BytesIO(smf.structural_model_to_bytes(small_model1))) as archive:
        assert set(archive.namelist()) == {"manifest.json", "model.json", "arrays.npz"}
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format"] == "wbgeo_structural_model"
    assert manifest["format_version"] == smf.FORMAT_VERSION
    assert manifest["result_type"] == "StructuralModelResults"


def _rewrite(data, member, new_content):
    """Return the file with one member replaced (None removes it)."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == member:
                if new_content is not None:
                    dst.writestr(name, new_content)
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def _manifest_with(data, **changes):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    return _rewrite(data, "manifest.json", json.dumps({**manifest, **changes}))


def test_load_rejects_invalid_files(small_model1):
    data = smf.structural_model_to_bytes(small_model1)
    cases = [
        (b"not a zip", "not a ZIP archive"),
        (_rewrite(data, "arrays.npz", None), "missing arrays.npz"),
        (_manifest_with(data, format="something_else"), "format 'something_else'"),
        (_manifest_with(data, result_type="MeshResults"), "contains a MeshResults, not a structural model"),
        (_manifest_with(data, format_version=smf.FORMAT_VERSION + 1), "written by a newer WBGeo"),
        (_manifest_with(data, format_version="1"), "Invalid format version"),
    ]
    for content, message in cases:
        with pytest.raises(ValueError, match=message):
            smf.structural_model_from_bytes(content)


def test_save_rejects_other_types():
    with pytest.raises(TypeError, match="Expected a StructuralModelResults"):
        smf.structural_model_to_bytes("not a model")


@pytest.mark.integration
@pytest.mark.slow
def test_loaded_model_gives_the_same_unstructured_mesh(small_model2):
    from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
    loaded = smf.structural_model_from_bytes(smf.structural_model_to_bytes(small_model2))
    kwargs = dict(tolerance=50, mesh_size=150, curve_mesh_size=20)
    mesh = create_unstructured_mesh_data(geomodel_result=small_model2, **kwargs)
    mesh_loaded = create_unstructured_mesh_data(geomodel_result=loaded, **kwargs)

    assert np.allclose(mesh.nodes, mesh_loaded.nodes)
    assert [(b.type, len(b.data)) for b in mesh.elements] == [(b.type, len(b.data)) for b in mesh_loaded.elements]
    for block, block_loaded in zip(mesh.elements, mesh_loaded.elements):
        assert np.array_equal(block.data, block_loaded.data)
