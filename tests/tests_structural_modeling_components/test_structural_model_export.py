"""Export Structural Model (GOCAD TSurf, OBJ, VTK, .wbgeo_sm) and Load Structural Model."""
import csv
import io
import zipfile

import numpy as np
import pytest

from core.structural_modeling_components import structural_workbench_components as swc
from core.structural_modeling_components.structural_model_export import (
    StructuralModelExportFormat,
    export_structural_model,
)
from core.structural_modeling_components.structural_model_file import structural_model_from_bytes
from core.structural_modeling_components.structural_objects.structural_objects import FaultMeshType, GeoMeshType


def _members(buffer):
    with zipfile.ZipFile(buffer) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _expected_surfaces(model):
    """(name, kind, n_vertices, n_triangles) of all surfaces, horizons youngest to oldest, then faults."""
    frame = model.structural_frame
    out = [(e.name, "horizon", len(e.vertices[GeoMeshType.MASKED]), len(e.edges[GeoMeshType.MASKED]))
           for g in frame.structural_groups for e in g.structural_elements]
    if frame.fault_frame is not None:
        out += [(f.name, "fault", len(f.vertices[FaultMeshType.UNMASKED]), len(f.edges[FaultMeshType.UNMASKED]))
                for f in frame.fault_frame.fault_elements]
    return out


def _surface_table(members):
    return list(csv.DictReader(io.StringIO(members["surfaces.csv"].decode())))


def _read_tsurf(text):
    """Minimal GOCAD TSurf reader: name, vertex ids and coordinates, triangles."""
    name, ids, vertices, triangles = None, [], [], []
    for line in text.splitlines():
        if line.startswith("name:"):
            name = line[5:]
        elif line.startswith("VRTX "):
            _, i, x, y, z = line.split()
            ids.append(int(i))
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("TRGL "):
            triangles.append(tuple(int(v) for v in line.split()[1:]))
    return name, ids, np.array(vertices), np.array(triangles)


def test_wbgeo_export_can_be_loaded_again(small_model2):
    buffer = export_structural_model(small_model2, "wbgeo_sm")
    assert buffer.filename == "structural_model.wbgeo_sm"
    loaded = structural_model_from_bytes(buffer.getvalue())
    assert np.array_equal(loaded.structural_frame.lith_block, small_model2.structural_frame.lith_block)


def test_surface_table_lists_the_stratigraphic_order(small_model2):
    rows = _surface_table(_members(export_structural_model(small_model2, "obj")))
    assert [(r["order"], r["name"], r["kind"], r["group"]) for r in rows] == [
        ("1", "rock4", "horizon", "Strat_Series2"), ("2", "rock3", "horizon", "Strat_Series2"),
        ("3", "rock2", "horizon", "Strat_Series1"), ("4", "rock1", "horizon", "Strat_Series1"),
        ("5", "fault", "fault", "")]


def test_gocad_tsurf_export(small_model2):
    buffer = export_structural_model(small_model2, StructuralModelExportFormat.GOCAD_TS)
    assert buffer.filename == "structural_model_gocad_ts.zip"
    members = _members(buffer)
    rows = _surface_table(members)
    expected = _expected_surfaces(small_model2)
    assert [r["file"] for r in rows] == [f"{name}.ts" for name, *_ in expected]

    frame = small_model2.structural_frame
    rock1 = frame.get_element_by_name("rock1")
    for (name, kind, n_vertices, n_triangles), row in zip(expected, rows):
        text = members[row["file"]].decode()
        assert text.startswith("GOCAD TSurf 1\nHEADER {\n")
        assert text.rstrip().endswith("END")
        assert ("GEOLOGICAL_TYPE fault" in text) == (kind == "fault")
        ts_name, ids, vertices, triangles = _read_tsurf(text)
        assert ts_name == name
        assert ids == list(range(1, n_vertices + 1))  # 1-based, consecutive
        assert len(triangles) == n_triangles
        assert triangles.min() >= 1 and triangles.max() <= n_vertices
    _, _, vertices, triangles = _read_tsurf(members["rock1.ts"].decode())
    assert np.allclose(vertices, rock1.vertices[GeoMeshType.MASKED], atol=1e-6)
    assert np.array_equal(triangles - 1, rock1.edges[GeoMeshType.MASKED])


def test_obj_export(small_model2):
    members = _members(export_structural_model(small_model2, "obj"))
    assert set(members) == {"structural_model.obj", "structural_model.mtl", "surfaces.csv"}
    obj = members["structural_model.obj"].decode().splitlines()
    expected = _expected_surfaces(small_model2)
    assert [line[2:] for line in obj if line.startswith("o ")] == [name for name, *_ in expected]
    assert sum(line.startswith("v ") for line in obj) == sum(e[2] for e in expected)
    faces = np.array([[int(v) for v in line.split()[1:]] for line in obj if line.startswith("f ")])
    assert len(faces) == sum(e[3] for e in expected)
    assert faces.min() == 1 and faces.max() == sum(e[2] for e in expected)
    mtl = members["structural_model.mtl"].decode()
    assert mtl.count("newmtl ") == len(expected) and "Kd " in mtl


def test_vtk_export(small_model2, tmp_path):
    import pyvista as pv
    members = _members(export_structural_model(small_model2, "vtk"))
    expected = _expected_surfaces(small_model2)
    assert set(members) == {"surfaces.csv", "lithology_block.vti"} | {f"surfaces/{e[0]}.vtp" for e in expected}
    for name, _, n_vertices, n_triangles in expected:
        path = tmp_path / f"{name}.vtp"
        path.write_bytes(members[f"surfaces/{name}.vtp"])
        poly = pv.read(path)
        assert (poly.n_points, poly.n_cells) == (n_vertices, n_triangles)

    path = tmp_path / "lithology_block.vti"
    path.write_bytes(members["lithology_block.vti"])
    image = pv.read(path)
    frame = small_model2.structural_frame
    assert image.n_cells == np.prod(frame.grid.resolution)
    assert np.allclose(image.bounds, frame.grid.extent)
    # cell order: x fastest; the lithology block is stored (nx, ny, nz)
    lith = image.cell_data["lithology"].reshape(frame.grid.resolution, order="F")
    assert np.array_equal(lith, frame.lith_block)
    assert {"scalar_field_Strat_Series1", "scalar_field_Strat_Series2", "fault_domain",
            "scalar_field_fault"} <= set(image.cell_data.keys())


def test_vtk_export_of_an_octree_model(small_model2_octree, tmp_path):
    """On an octree grid the lithology block is an unstructured grid with one hexahedron per octree cell."""
    import pyvista as pv
    model = small_model2_octree
    members = _members(export_structural_model(model, "vtk"))
    assert "lithology_block.vtu" in members and "lithology_block.vti" not in members
    expected = _expected_surfaces(model)
    assert {f"surfaces/{e[0]}.vtp" for e in expected} <= set(members)

    path = tmp_path / "lithology_block.vtu"
    path.write_bytes(members["lithology_block.vtu"])
    block = pv.read(path)
    frame = model.structural_frame
    assert block.n_cells == len(frame.cells)
    assert set(block.celltypes) == {pv.CellType.HEXAHEDRON}
    assert np.allclose(block.bounds, frame.grid.extent)
    assert np.array_equal(block.cell_data["lithology"], frame.lith_block)
    # cell order is the frame's: cell centers coincide
    assert np.allclose(block.cell_centers().points, frame.cell_centers())
    # cells of different sizes, and no gaps: their volumes add up to the model volume
    sizes = block.compute_cell_sizes()["Volume"]
    assert len(np.unique(np.round(sizes, 6))) > 1
    assert np.isclose(sizes.sum(), np.prod(np.diff(np.reshape(frame.grid.extent, (3, 2)), axis=1)))
    # fault results come from the fault model's own cells, taken at the structural cells' centers
    ff = frame.fault_frame
    domains = np.asarray(ff.domain_map)[ff.cells.locate(ff.grid, frame.cell_centers())]
    assert np.array_equal(block.cell_data["fault_domain"], domains)
    assert len(np.unique(domains)) > 1
    assert {"scalar_field_Strat_Series1", "mask_Strat_Series1", "scalar_field_fault"} <= set(block.cell_data.keys())


def test_file_names_stay_unique_and_safe():
    from core.structural_modeling_components.structural_model_export import _Surface, _file_names
    surfaces = [_Surface(n, "horizon", "g", None, np.zeros((3, 3)), np.zeros((1, 3), int))
                for n in ("Rock 1", "rock_1", "a/b", "")]
    assert _file_names(surfaces, ".ts") == ["Rock_1.ts", "rock_1_2.ts", "a_b.ts", "surface.ts"]


def test_export_component_forwards_the_format(small_model1):
    buffer = swc.export_structural_model(small_model1, "gocad_ts")
    assert buffer.filename == "structural_model_gocad_ts.zip"
    assert swc.export_structural_model(small_model1).filename == "structural_model.wbgeo_sm"


def test_load_component_reads_an_exported_file(small_model1, tmp_path):
    path = tmp_path / "model.wbgeo_sm"
    path.write_bytes(swc.export_structural_model(small_model1).getvalue())
    loaded = swc.load_structural_model_from_file(str(path))  # an absolute path stays as it is
    assert np.array_equal(loaded.structural_frame.lith_block, small_model1.structural_frame.lith_block)
    with pytest.raises(ValueError, match="Structural model file not found"):
        swc.load_structural_model_from_file(str(tmp_path / "missing.wbgeo_sm"))
