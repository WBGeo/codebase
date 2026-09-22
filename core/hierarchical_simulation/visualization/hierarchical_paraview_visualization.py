"""Launch ParaView for WBGeo hierarchical Exodus meshes.

This module is separate from hierarchical_meshing_visualization.py.
It launches the ParaView desktop GUI from the normal WBGeo Python environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

from py_api_wbgeo.nodesapi import wbgeo_component

PathLike = Union[str, os.PathLike]


def _find_paraview_executable(
    paraview_executable: Optional[PathLike] = None,
) -> str:
    """Return the ParaView desktop executable."""
    if paraview_executable is not None:
        candidate = Path(paraview_executable).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

        resolved = shutil.which(str(paraview_executable))
        if resolved:
            return resolved

        raise FileNotFoundError(
            f"ParaView executable was not found: {paraview_executable}"
        )

    resolved = shutil.which("paraview")
    if resolved:
        return resolved

    candidates = []

    if sys.platform == "darwin":
        candidates = [
            Path("/Applications/ParaView.app/Contents/MacOS/paraview"),
            Path.home() / "Applications/ParaView.app/Contents/MacOS/paraview",
        ]
    elif not sys.platform.startswith("win"):
        candidates = [
            Path("/usr/bin/paraview"),
            Path("/usr/local/bin/paraview"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "ParaView could not be found. Install ParaView, add 'paraview' "
        "to PATH, or pass paraview_executable='/path/to/paraview'."
    )


def _validate_exodus_file(filename: PathLike) -> str:
    """Validate and normalize an Exodus filename."""
    path = Path(filename).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Exodus file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Expected an Exodus file, got: {path}")

    return str(path)


def open_exodus_in_paraview(
    exodus_file: PathLike,
    *,
    paraview_executable: Optional[PathLike] = None,
    wait: bool = False,
) -> subprocess.Popen:
    """Open one Exodus file in the ParaView desktop GUI."""
    executable = _find_paraview_executable(paraview_executable)
    filename = _validate_exodus_file(exodus_file)

    process = subprocess.Popen([executable, filename])

    if wait:
        process.wait()

    return process


@wbgeo_component(
    identifier="wbgeo::open_hierarchical_exodus_in_paraview",
    title="Open Hierarchical Exodus in ParaView",
    description=(
        "Opens the two exported hierarchical Exodus meshes in the ParaView "
        "desktop application for native block visibility and inspection."
    ),
    color="#A52A2A",
    border_color="#000000",
    group="Hierarchical Simulation",
)
def open_hierarchical_exodus_in_paraview(
    mesh1_exodus: PathLike,
    mesh2_exodus: PathLike,
    paraview_executable: Optional[PathLike] = None,
    wait: bool = False,
) -> subprocess.Popen:
    """Open both hierarchical Exodus meshes in one ParaView GUI session.

    mesh1_exodus is the bigger/outer mesh and mesh2_exodus is the
    smaller/inner mesh. Both appear as separate sources in ParaView's
    Pipeline Browser, where native block visibility controls can be used.
    """
    executable = _find_paraview_executable(paraview_executable)
    mesh1_file = _validate_exodus_file(mesh1_exodus)
    mesh2_file = _validate_exodus_file(mesh2_exodus)

    process = subprocess.Popen([executable, mesh1_file, mesh2_file])

    if wait:
        process.wait()

    return process


def open_exodus_files_in_paraview(
    exodus_files: Sequence[PathLike],
    *,
    paraview_executable: Optional[PathLike] = None,
    wait: bool = False,
) -> subprocess.Popen:
    """Open any number of Exodus files in one ParaView GUI session."""
    if not exodus_files:
        raise ValueError("At least one Exodus file is required.")

    executable = _find_paraview_executable(paraview_executable)
    filenames = [_validate_exodus_file(f) for f in exodus_files]

    process = subprocess.Popen([executable, *filenames])

    if wait:
        process.wait()

    return process
