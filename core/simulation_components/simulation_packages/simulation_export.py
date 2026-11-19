"""Export of WBGeo SimulationResults, the same way for every simulator.

Everything comes from the result itself (geometry, fields, solver tables,
input files, solver output); no solver output folder needs to exist, so an
export works the same after the Workbench has passed the result between
nodes as JSON.
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from enum import StrEnum

from core.object_components import SimulationResults
from core.simulation_components.output_format.csv_tables import numeric_csv_text
from core.simulation_components.output_format.exodus.exodus_results_writer import write_exodus_results
from core.simulation_components.simulation_visualization.simulation_visualization import build_grid_from_class
from py_api_wbgeo.nodesapi import BasicallyABufferedFile


class SimulationExportFormat(StrEnum):
    """Export formats for SimulationResults."""
    AUTO = "auto"      # the simulator's usual format: GOLEM -> Exodus, otherwise VTK
    VTK = "vtk"        # one legacy VTK file per saved time step + ParaView PVD
    EXODUS = "exodus"  # one multi-time-step Exodus II file


def export_simulation_results(
    sim: SimulationResults,
    format: SimulationExportFormat = SimulationExportFormat.AUTO,
) -> BasicallyABufferedFile:
    """Export a SimulationResults from either simulator as one ZIP.

    The ZIP contains the fields in the chosen format -- ``vtk``: one legacy
    VTK file per time step plus ``results.pvd``; ``exodus``: ``results.e``
    with all time steps and named blocks -- plus whatever else the result
    holds: solver tables as CSV under ``tables/``, the input file(s) under
    ``input/`` and the solver console output under ``solver_output/``.
    ``auto`` uses Exodus for GOLEM results (its native format) and VTK
    otherwise. ``format`` accepts a SimulationExportFormat or its value.
    """
    try:
        fmt = SimulationExportFormat(str(format or "auto").strip().lower())
    except ValueError:
        raise NotImplementedError(
            f"Unsupported simulation export format {format!r}. "
            f"Supported formats are {[f.value for f in SimulationExportFormat]}."
        ) from None
    if fmt == SimulationExportFormat.AUTO:
        fmt = SimulationExportFormat.EXODUS if sim.solver == "golem" else SimulationExportFormat.VTK
    if not sim.nodes_by_time:
        raise ValueError("SimulationResults has no time steps to export.")

    tmp_dir = tempfile.mkdtemp(prefix="sim_export_")
    try:
        files = (_write_vtk(sim, tmp_dir) if fmt == SimulationExportFormat.VTK
                 else _write_exodus(sim, tmp_dir))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(os.path.join(tmp_dir, name), name)
            for name, table in sim.solver_tables.items():
                archive.writestr(f"tables/{name}", numeric_csv_text(table))
            for name, text in sim.input_files.items():
                archive.writestr(f"input/{name}", text)
            for stage, text in sim.solver_output.items():
                archive.writestr(f"solver_output/{stage}.txt", text)
        buf.seek(0)
        buf.filename = "simulation_results.zip"
        return buf
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_vtk(sim: SimulationResults, directory: str) -> list[str]:
    times = sorted(sim.nodes_by_time)
    names = [f"result_{i:04d}.vtk" for i in range(len(times))]
    for name, t in zip(names, times):
        build_grid_from_class(sim, t).save(os.path.join(directory, name))
    datasets = "\n".join(
        f'    <DataSet timestep="{t}" group="" part="0" file="{name}"/>' for name, t in zip(names, times)
    )
    with open(os.path.join(directory, "results.pvd"), "w", encoding="utf-8") as stream:
        stream.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            '  <Collection>\n'
            f'{datasets}\n'
            '  </Collection>\n'
            '</VTKFile>\n'
        )
    return ["results.pvd"] + names


def _write_exodus(sim: SimulationResults, directory: str) -> list[str]:
    write_exodus_results(sim, os.path.join(directory, "results.e"))
    return ["results.e"]
