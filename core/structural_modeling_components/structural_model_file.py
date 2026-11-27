"""
Save and load computed structural models in WBGeo's own file format (``.wbgeo_sm``).

A file is a ZIP archive with three members:

- ``manifest.json``: file format name and version, result type, creation time;
- ``model.json``: the result serialized with its Pydantic schema, with every NumPy array replaced by a
  reference ``{"__array__": "<key>"}``;
- ``arrays.npz``: the referenced arrays, compressed, in their original dtypes.

Because the file follows the Pydantic schema of the result types, fields that are added later are saved
without changes here. Loading returns the same type that was saved, so every downstream component works
unchanged. Incompatible changes of the result types increase ``FORMAT_VERSION`` and add a migration step to
``_MIGRATIONS``, so older files stay readable.
"""
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Union

import numpy as np
from pydantic import TypeAdapter

from core.object_components import StructuralModelResults

logger = logging.getLogger(__name__)

FILE_EXTENSION = ".wbgeo_sm"
FORMAT_NAME = "wbgeo_structural_model"
FORMAT_VERSION = 1
RESULT_TYPE = "StructuralModelResults"

_ARRAY_KEY = "__array__"
_MANIFEST, _MODEL, _ARRAYS = "manifest.json", "model.json", "arrays.npz"

#: format version -> function that upgrades the ``model.json`` content of that version to the next one
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def _extract_arrays(obj: Any, arrays: Dict[str, np.ndarray]) -> Any:
    """Replace every NumPy array in a dumped result by a reference and collect the arrays."""
    if isinstance(obj, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = obj
        return {_ARRAY_KEY: key}
    if isinstance(obj, dict):
        return {_json_key(k): _extract_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_extract_arrays(v, arrays) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _json_key(key: Any) -> Any:
    if isinstance(key, Enum):
        return key.value
    if isinstance(key, np.generic):
        return key.item()
    return key


def _insert_arrays(obj: Any, arrays: Dict[str, np.ndarray]) -> Any:
    """Put the arrays back in place of their references."""
    if isinstance(obj, dict):
        if set(obj) == {_ARRAY_KEY}:
            return arrays[obj[_ARRAY_KEY]]
        return {k: _insert_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_insert_arrays(v, arrays) for v in obj]
    return obj


def structural_model_to_bytes(model: StructuralModelResults) -> bytes:
    """Serialize a structural model into the content of a ``.wbgeo_sm`` file."""
    if not isinstance(model, StructuralModelResults):
        raise TypeError(f"Expected a StructuralModelResults, got {type(model).__name__}.")
    arrays: Dict[str, np.ndarray] = {}
    content = _extract_arrays(TypeAdapter(StructuralModelResults).dump_python(model), arrays)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "result_type": RESULT_TYPE,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST, json.dumps(manifest, indent=1))
        archive.writestr(_MODEL, json.dumps(content))
        # the npz is already compressed
        archive.writestr(zipfile.ZipInfo(_ARRAYS), array_buffer.getvalue(), compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def structural_model_from_bytes(data: bytes) -> StructuralModelResults:
    """Read a structural model from the content of a ``.wbgeo_sm`` file."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError("This is not a WBGeo structural model file (not a ZIP archive).") from e
    with archive:
        names = set(archive.namelist())
        missing = {_MANIFEST, _MODEL, _ARRAYS} - names
        if missing:
            raise ValueError(f"This is not a WBGeo structural model file (missing {', '.join(sorted(missing))}).")
        manifest = json.loads(archive.read(_MANIFEST))
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(f"This is not a WBGeo structural model file (format {manifest.get('format')!r}).")
        if manifest.get("result_type") != RESULT_TYPE:
            raise ValueError(f"The file contains a {manifest.get('result_type')}, not a structural model.")
        version = manifest.get("format_version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"Invalid format version {version!r} in the file.")
        if version > FORMAT_VERSION:
            raise ValueError(f"The file was written by a newer WBGeo (format version {version}); this version "
                             f"reads up to format version {FORMAT_VERSION}. Please update WBGeo.")
        content = json.loads(archive.read(_MODEL))
        with np.load(io.BytesIO(archive.read(_ARRAYS)), allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    while version < FORMAT_VERSION:
        content = _MIGRATIONS[version](content)
        version += 1
    return TypeAdapter(StructuralModelResults).validate_python(_insert_arrays(content, arrays))


def save_structural_model(model: StructuralModelResults, path: Union[str, Path]) -> Path:
    """Save a structural model to a ``.wbgeo_sm`` file; the extension is added if missing.

    Returns:
        The path of the written file.
    """
    path = Path(path)
    if path.suffix != FILE_EXTENSION:
        path = path.with_name(path.name + FILE_EXTENSION)
    path.write_bytes(structural_model_to_bytes(model))
    logger.info("Saved structural model to %s", path)
    return path


def load_structural_model(path: Union[str, Path]) -> StructuralModelResults:
    """Load a structural model from a ``.wbgeo_sm`` file."""
    return structural_model_from_bytes(Path(path).read_bytes())
