"""Generic export of WBGeo SimulationResults, independent of solver package."""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from core.object_components import SimulationResults
from core.simulation_components.simulation_visualization.simulation_visualization import build_grid_from_class
from py_api_wbgeo.nodesapi import BasicallyABufferedFile


def export_simulation_results(
    sim: SimulationResults, format: str = "auto", golem_output_dir: str = ""
) -> BasicallyABufferedFile:
    """Export all time steps from SFePy or GOLEM SimulationResults.

    ``format="vtk"`` returns a ZIP containing one legacy VTK file per time
    step plus a ParaView PVD collection. ``format="exodus"`` returns GOLEM's
    original native Exodus result file unchanged. ``format="golem"`` (or
    ``"native"``) returns a ZIP of the complete GOLEM output directory,
    including the Exodus result and any CSV/log/generated files.
    """
    fmt = (format or "auto").strip().lower()
    solver = str(getattr(sim, "source_solver", "") or "").strip().lower()
    native_dir = _find_golem_output_dir(sim, golem_output_dir)
    if fmt in {"auto", "default"}:
        if solver == "golem" or native_dir is not None:
            return _export_golem_output_directory(sim, native_dir)
        if solver == "sfepy" or getattr(sim, "sfepy_stdout", None):
            return _export_simulation_results_to_vtk(sim)
        raise ValueError(
            "Cannot identify the solver or find GOLEM's native output directory. "
            "Specify golem_output_dir in the DSL export component, or choose "
            "format='vtk' for SfePy. The exporter will not silently convert "
            "unidentified GOLEM results into VTK."
        )
    if fmt == "vtk":
        return _export_simulation_results_to_vtk(sim)
    if fmt in {"exodus", "exo", "e", "ex2"}:
        return _export_simulation_results_to_exodus(sim, native_dir)
    if fmt in {"golem", "native", "golem_zip", "output_dir"}:
        return _export_golem_output_directory(sim, native_dir)
    raise NotImplementedError(
        f"Unsupported simulation export format {format!r}. "
        "Supported formats are 'auto', 'vtk', 'exodus', and 'golem'."
    )


def _native_files(directory: Path):
    return sorted(
        (p for p in directory.rglob("*")
         if p.is_file() and p.suffix.lower() in {".e", ".exo", ".ex2"}
         and not any(part.endswith("_cp") for part in p.relative_to(directory).parts)),
        key=lambda p: str(p),
    )


def _find_golem_output_dir(sim: SimulationResults, override: str = ""):
    """Resolve native outputs, even when Workbench discarded dynamic metadata.

    Never choose arbitrarily among multiple GOLEM runs.
    """
    if override.strip():
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"GOLEM output directory does not exist: {candidate}")
        if not _native_files(candidate):
            raise FileNotFoundError(f"No Exodus file in GOLEM output directory: {candidate}")
        return candidate

    for attr in ("source_output_dir", "source_exodus_file"):
        value = getattr(sim, attr, None)
        if value:
            candidate = Path(value).expanduser().resolve()
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.is_dir() and _native_files(candidate):
                return candidate

    # The Workbench GOLEM run component saves outputs to
    # Path.cwd()/golem_output/<unique run id>.
    base = Path.cwd() / "golem_output"
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and _native_files(p)]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        raise ValueError(
            f"Found {len(candidates)} GOLEM output directories under {base}. "
            "Cannot safely associate a simulation result with one run after "
            "serialization. Set golem_output_dir explicitly to the correct "
            "directory in the Export Simulation Results component."
        )
    return None


def _times(sim: SimulationResults):
    times = sorted(sim.nodes_by_time.keys())
    if not times:
        raise ValueError("SimulationResults has no time steps to export.")
    return times


def _export_simulation_results_to_vtk(
    sim: SimulationResults,
) -> BasicallyABufferedFile:
    times = _times(sim)
    tmp_dir = tempfile.mkdtemp(prefix="sim_export_vtk_")
    try:
        vtk_names = [f"result_{i:04d}.vtk" for i in range(len(times))]
        for name, t in zip(vtk_names, times):
            grid = build_grid_from_class(sim, t)
            grid.save(os.path.join(tmp_dir, name))

        dataset_lines = "\n".join(
            f'    <DataSet timestep="{t}" group="" part="0" file="{name}"/>'
            for name, t in zip(vtk_names, times)
        )
        pvd_content = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            '  <Collection>\n'
            f'{dataset_lines}\n'
            '  </Collection>\n'
            '</VTKFile>\n'
        )
        pvd_path = os.path.join(tmp_dir, "results.pvd")
        with open(pvd_path, "w", encoding="utf-8") as stream:
            stream.write(pvd_content)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pvd_path, "results.pvd")
            for name in vtk_names:
                zf.write(os.path.join(tmp_dir, name), name)
        buf.seek(0)
        buf.filename = "simulation_results.zip"
        return buf
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _export_simulation_results_to_exodus(
    sim: SimulationResults, source_dir: Path | None = None
) -> BasicallyABufferedFile:
    """Return a native GOLEM Exodus file without VTK conversion."""
    source = getattr(sim, "source_exodus_file", None)
    candidate = Path(source).resolve() if source else None
    if candidate is None or not candidate.is_file():
        directory = source_dir or _find_golem_output_dir(sim)
        if directory is None:
            raise FileNotFoundError("No accessible GOLEM output directory found.")
        files = _native_files(directory)
        if not files:
            raise FileNotFoundError(f"No Exodus file in {directory}")
        candidate = max(files, key=lambda p: p.stat().st_mtime)
    buf = io.BytesIO(candidate.read_bytes())
    buf.filename = candidate.name
    return buf


def _export_golem_output_directory(
    sim: SimulationResults, source_dir: Path | None = None
) -> BasicallyABufferedFile:
    """ZIP preserved native GOLEM outputs, including Exodus and CSV files."""
    directory = source_dir or _find_golem_output_dir(sim)
    if directory is None:
        raise FileNotFoundError(
            "Cannot locate GOLEM output files. Set golem_output_dir to the "
            "directory containing the .e/.exo and .csv files."
        )
    directory = directory.resolve()
    if not _native_files(directory):
        raise FileNotFoundError(f"No native GOLEM Exodus file in {directory}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if not d.endswith("_cp"))
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink() or not path.is_file():
                    continue
                archive.write(path, path.relative_to(directory).as_posix())
    buf.seek(0)
    buf.filename = "golem_results.zip"
    return buf
