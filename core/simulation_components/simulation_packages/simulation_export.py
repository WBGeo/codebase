"""Generic export of WBGeo SimulationResults, independent of solver package."""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile

from vtkmodules.vtkIOExodus import vtkExodusIIWriter

from core.object_components import SimulationResults
from core.simulation_components.simulation_visualization.simulation_visualization import build_grid_from_class
from py_api_wbgeo.nodesapi import BasicallyABufferedFile


def export_simulation_results(
    sim: SimulationResults, format: str = "vtk"
) -> BasicallyABufferedFile:
    """Export all time steps from SFePy or GOLEM SimulationResults.

    ``format="vtk"`` returns a ZIP containing one legacy VTK file per time
    step plus a ParaView PVD collection. ``format="exodus"`` returns GOLEM's
    original native Exodus result file unchanged. ``format="golem"`` (or
    ``"native"``) returns a ZIP of the complete GOLEM output directory,
    including the Exodus result and any CSV/log/generated files.
    """
    fmt = format.strip().lower()
    if fmt in {"vtk"}:
        return _export_simulation_results_to_vtk(sim)
    if fmt in {"exodus", "exo", "e", "ex2"}:
        return _export_simulation_results_to_exodus(sim)
    if fmt in {"golem", "native", "golem_zip", "output_dir"}:
        return _export_golem_output_directory(sim)
    raise NotImplementedError(
        f"Unsupported simulation export format {format!r}. "
        "Supported formats are 'vtk', 'exodus', and 'golem'."
    )


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
    sim: SimulationResults,
) -> BasicallyABufferedFile:
    """Return GOLEM's original multi-time-step Exodus result unchanged.

    GOLEM already writes one native Exodus database (normally ``wbgeo.e``)
    containing all time steps.  Re-writing the unified ``SimulationResults``
    would lose that native one-file representation, so Exodus export is only
    available when the result object came from GOLEM and retains its source
    Exodus path.
    """
    source = getattr(sim, "source_exodus_file", None)
    if not source or not os.path.isfile(source):
        raise ValueError(
            "Exodus export requires a GOLEM SimulationResults object with its "
            "native source Exodus file. SFePy results should be exported as VTK."
        )

    with open(source, "rb") as stream:
        buf = io.BytesIO(stream.read())
    buf.seek(0)
    buf.filename = os.path.basename(source) or "wbgeo.e"
    return buf


def _export_golem_output_directory(
    sim: SimulationResults,
) -> BasicallyABufferedFile:
    """ZIP the complete preserved GOLEM output directory.

    This is the recommended GOLEM export when the run may create additional
    files such as CSV tables. The archive contains exactly the user-facing
    files preserved by ``run_simulation_golem()`` in its output directory.
    Internal ``*_cp`` checkpoint directories have already been filtered by
    the runner.
    """
    source_dir = getattr(sim, "source_output_dir", None)
    if not source_dir or not os.path.isdir(source_dir):
        raise ValueError(
            "GOLEM output-directory export requires a GOLEM SimulationResults "
            "object that retains its source output directory."
        )

    source_dir = os.path.abspath(source_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = sorted(d for d in dirs if not d.endswith("_cp"))
            for filename in sorted(files):
                path = os.path.join(root, filename)
                arcname = os.path.relpath(path, source_dir)
                zf.write(path, arcname)

    buf.seek(0)
    buf.filename = "golem_results.zip"
    return buf
