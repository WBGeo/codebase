"""
Export computed structural models: WBGeo's own ``.wbgeo_sm`` file (lossless, can be loaded again) or the
surfaces and lithology block for other software (GOCAD TSurf, OBJ, VTK).

The external formats contain the surfaces of all structural elements (``masked`` meshes, i.e. where the
element exists in the model) and of all faults (``unmasked`` meshes). Each export also contains
``surfaces.csv``, which lists the surfaces youngest to oldest with their group, kind (horizon or fault),
colour and file, because these formats do not store the stratigraphic order. The VTK export also contains
the lithology block, the scalar fields and the fault domains as a grid: VTK ImageData (``.vti``) for a
model on a regular grid, an unstructured grid of hexahedra (``.vtu``) for a model on an octree grid.
"""
import csv
import io
import os
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List, Optional

import numpy as np

from core.object_components import StructuralModelResults
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid
from core.structural_modeling_components.structural_model_file import FILE_EXTENSION, structural_model_to_bytes
from core.structural_modeling_components.structural_objects.structural_objects import FaultMeshType, GeoMeshType


class StructuralModelExportFormat(StrEnum):
    """Export formats for structural models."""
    WBGEO = "wbgeo_sm"      # WBGeo's own file, lossless, can be loaded again
    GOCAD_TS = "gocad_ts"   # one GOCAD TSurf (.ts) file per surface
    OBJ = "obj"             # one Wavefront OBJ file with all surfaces, colours in an MTL file
    VTK = "vtk"             # surfaces as VTK PolyData (.vtp) + lithology block and fields as a VTK grid (.vti/.vtu)


EXPORT_FILE_NAME = "structural_model"


@dataclass
class _Surface:
    name: str
    kind: str              # "horizon" or "fault"
    group: str             # structural group of a horizon, "" for a fault
    color: Optional[str]   # hex colour, e.g. "#5fa040"
    vertices: np.ndarray   # (n, 3) float
    triangles: np.ndarray  # (m, 3) int, 0-based


def _surfaces(model: StructuralModelResults) -> List[_Surface]:
    """All non-empty surfaces, horizons youngest to oldest, then faults."""
    frame = model.structural_frame
    surfaces = []
    for group in frame.structural_groups:
        for element in group.structural_elements:
            vertices, triangles = element.vertices.get(GeoMeshType.MASKED), element.edges.get(GeoMeshType.MASKED)
            if vertices is not None and triangles is not None and len(vertices) and len(triangles):
                surfaces.append(_Surface(element.name, "horizon", group.name, element.color,
                                         np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)))
    if frame.fault_frame is not None:
        for fault in frame.fault_frame.fault_elements:
            vertices, triangles = fault.vertices.get(FaultMeshType.UNMASKED), fault.edges.get(FaultMeshType.UNMASKED)
            if vertices is not None and triangles is not None and len(vertices) and len(triangles):
                surfaces.append(_Surface(fault.name, "fault", "", fault.color,
                                         np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)))
    if not surfaces:
        raise ValueError("The structural model has no surfaces to export. Compute it with surface extraction "
                         "(extract_meshes=True).")
    return surfaces


def _safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("._")
    return keep or "surface"


def _file_names(surfaces: List[_Surface], extension: str) -> List[str]:
    """One unique file name per surface."""
    names, used = [], set()
    for surface in surfaces:
        base = _safe_name(surface.name)
        candidate, i = f"{base}{extension}", 2
        while candidate.lower() in used:
            candidate, i = f"{base}_{i}{extension}", i + 1
        used.add(candidate.lower())
        names.append(candidate)
    return names


def _rgb(color: Optional[str]) -> tuple:
    """Hex colour -> RGB in [0, 1]; grey if unset or invalid."""
    try:
        value = color.lstrip("#")
        return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except (AttributeError, ValueError):
        return (0.6, 0.6, 0.6)


def _surface_table(surfaces: List[_Surface], files: List[str]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["order", "name", "kind", "group", "color", "file"])
    for i, (surface, file) in enumerate(zip(surfaces, files), 1):
        writer.writerow([i, surface.name, surface.kind, surface.group, surface.color or "", file])
    return out.getvalue()


def gocad_tsurf(surface: _Surface) -> str:
    """One surface as a GOCAD TSurf (.ts) file: vertices (VRTX) and triangles (TRGL), 1-based."""
    r, g, b = _rgb(surface.color)
    lines = [
        "GOCAD TSurf 1",
        "HEADER {",
        f"name:{surface.name}",
        f"*solid*color:{r:.6f} {g:.6f} {b:.6f} 1",
        "}",
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM",
        "NAME Default",
        'AXIS_NAME "X" "Y" "Z"',
        'AXIS_UNIT "m" "m" "m"',
        "ZPOSITIVE Elevation",
        "END_ORIGINAL_COORDINATE_SYSTEM",
    ]
    if surface.kind == "fault":
        lines += [f"GEOLOGICAL_FEATURE {surface.name}", "GEOLOGICAL_TYPE fault"]
    lines.append("TFACE")
    lines += [f"VRTX {i} {x:.6f} {y:.6f} {z:.6f}" for i, (x, y, z) in enumerate(surface.vertices, 1)]
    lines += [f"TRGL {a} {b} {c}" for a, b, c in surface.triangles + 1]
    lines.append("END")
    return "\n".join(lines) + "\n"


def _obj_and_mtl(surfaces: List[_Surface]) -> tuple:
    """All surfaces in one OBJ file (one object per surface) plus its MTL file with the colours."""
    obj = [f"mtllib {EXPORT_FILE_NAME}.mtl"]
    mtl = []
    materials = _file_names(surfaces, "")
    offset = 1
    for surface, material in zip(surfaces, materials):
        r, g, b = _rgb(surface.color)
        mtl += [f"newmtl {material}", f"Kd {r:.6f} {g:.6f} {b:.6f}", ""]
        obj += [f"o {material}", f"usemtl {material}"]
        obj += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in surface.vertices]
        obj += [f"f {a} {b} {c}" for a, b, c in surface.triangles + offset]
        offset += len(surface.vertices)
    return "\n".join(obj) + "\n", "\n".join(mtl)


def _cell_arrays(model: StructuralModelResults) -> Dict[str, np.ndarray]:
    """The per-cell arrays of the model, flat in the frame's cell order (``array.ravel()``).

    Lithology, scalar field and age mask of each group, and the fault domain and scalar field of each
    fault. On an octree the fault results belong to the fault model's own cells; they are taken at the
    centers of the structural model's cells.
    """
    frame = model.structural_frame
    shape = tuple(frame.cell_array_shape())
    arrays: Dict[str, np.ndarray] = {}

    def add(name: str, values, on_fault_cells: bool = False) -> None:
        if values is None:
            return
        values = np.asarray(values)
        if on_fault_cells and isinstance(frame.grid, OctreeGrid):
            ff = frame.fault_frame
            if ff.cells is None or values.shape != (len(ff.cells),):
                return
            values = values[ff.cells.locate(ff.grid, frame.cell_centers())]
        if values.shape != shape:
            return
        if values.dtype == bool:
            values = values.astype(np.uint8)
        arrays[name] = values.ravel()

    add("lithology", frame.lith_block)
    for group in frame.structural_groups:
        add(f"scalar_field_{group.name}", group.scalar_field)
        add(f"mask_{group.name}", group.mask)
    if frame.fault_frame is not None:
        add("fault_domain", frame.fault_frame.domain_map, on_fault_cells=True)
        for fault in frame.fault_frame.fault_elements:
            add(f"scalar_field_{fault.name}", fault.scalar_field, on_fault_cells=True)
    return arrays


def _vtk_grid(model: StructuralModelResults):
    """Lithology block, scalar fields, masks and fault domains as a VTK grid with cell data, or None.

    Regular grid: VTK ImageData. Octree grid: an unstructured grid with one hexahedron per octree cell.
    """
    import pyvista as pv
    frame = model.structural_frame
    grid = frame.grid
    if grid is None or (isinstance(grid, OctreeGrid) and frame.cells is None):
        return None
    arrays = _cell_arrays(model)
    if not arrays:
        return None

    if isinstance(grid, OctreeGrid):
        mins, maxs = grid.cell_bounds(frame.cells.level, frame.cells.ijk)
        # VTK hexahedron corner order: bottom face counter-clockwise, then top face
        pick = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=bool)
        corners = np.where(pick[None, :, :], maxs[:, None, :], mins[:, None, :])  # (n, 8, 3)
        n = len(corners)
        cells = np.hstack([np.full((n, 1), 8), np.arange(8 * n).reshape(n, 8)]).ravel()
        vtk_grid = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8),
                                       corners.reshape(-1, 3))
        for name, values in arrays.items():
            vtk_grid.cell_data[name] = values
        # merge the corner points that neighbouring cells share (cell data stays as it is)
        return vtk_grid.clean()

    extent = np.asarray(grid.extent, dtype=float)
    resolution = np.asarray(grid.resolution, dtype=int)
    spacing = (extent[1::2] - extent[0::2]) / resolution
    image = pv.ImageData(dimensions=tuple(resolution + 1), spacing=tuple(spacing), origin=tuple(extent[0::2]))
    for name, values in arrays.items():
        # arrays are stored (nx, ny, nz); VTK cell data runs x fastest
        image.cell_data[name] = values.reshape(tuple(resolution)).ravel(order="F")
    return image


def _vtk_files(model: StructuralModelResults, surfaces: List[_Surface], files: List[str]) -> Dict[str, bytes]:
    import pyvista as pv
    out: Dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for surface, file in zip(surfaces, files):
            faces = np.hstack([np.full((len(surface.triangles), 1), 3), surface.triangles]).ravel()
            poly = pv.PolyData(surface.vertices, faces=faces)
            poly.field_data["name"] = [surface.name]
            poly.field_data["kind"] = [surface.kind]
            path = os.path.join(tmp, file)
            poly.save(path)
            with open(path, "rb") as f:
                out[f"surfaces/{file}"] = f.read()
        vtk_grid = _vtk_grid(model)
        if vtk_grid is not None:
            name = "lithology_block.vtu" if isinstance(model.structural_frame.grid, OctreeGrid) else "lithology_block.vti"
            path = os.path.join(tmp, name)
            vtk_grid.save(path)
            with open(path, "rb") as f:
                out[name] = f.read()
    return out


def export_structural_model(model: StructuralModelResults,
                            format: StructuralModelExportFormat = StructuralModelExportFormat.WBGEO) -> io.BytesIO:
    """Export a structural model as a file to download.

    Args:
        model: The computed structural model.
        format: ``wbgeo_sm`` (WBGeo's own file, can be loaded again), ``gocad_ts``, ``obj`` or ``vtk``.

    Returns:
        The file content, with ``filename`` set: ``structural_model.wbgeo_sm`` or a ZIP for the other formats.
    """
    format = StructuralModelExportFormat(format)
    if format == StructuralModelExportFormat.WBGEO:
        buffer = io.BytesIO(structural_model_to_bytes(model))
        buffer.filename = EXPORT_FILE_NAME + FILE_EXTENSION
        return buffer

    surfaces = _surfaces(model)
    members: Dict[str, bytes] = {}
    if format == StructuralModelExportFormat.GOCAD_TS:
        files = _file_names(surfaces, ".ts")
        for surface, file in zip(surfaces, files):
            members[file] = gocad_tsurf(surface).encode("utf-8")
    elif format == StructuralModelExportFormat.OBJ:
        files = [f"{EXPORT_FILE_NAME}.obj"] * len(surfaces)
        obj, mtl = _obj_and_mtl(surfaces)
        members[f"{EXPORT_FILE_NAME}.obj"] = obj.encode("utf-8")
        members[f"{EXPORT_FILE_NAME}.mtl"] = mtl.encode("utf-8")
    else:
        files = [f"surfaces/{name}" for name in _file_names(surfaces, ".vtp")]
        members.update(_vtk_files(model, surfaces, [f.split("/", 1)[1] for f in files]))
    members["surfaces.csv"] = _surface_table(surfaces, files).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    buffer.seek(0)
    buffer.filename = f"{EXPORT_FILE_NAME}_{format.value}.zip"
    return buffer
