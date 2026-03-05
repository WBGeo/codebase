# tests/test_marching_cubes_utils.py

from __future__ import annotations

import numpy as np
import pytest

from core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction import (
    marching_cubes, marching_cubes_per_element)


def test_marching_cubes_calls_skimage_for_each_level_and_offsets_vertices(monkeypatch):
    # Arrange
    block = np.zeros((2, 2, 2), dtype=float)
    elements = [10, 20]  # important: non-0/1 IDs
    spacing = (1.0, 2.0, 3.0)
    extent = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)

    calls: list[dict] = []

    def fake_marching_cubes(block_arg, level, spacing=None, **kwargs):
        calls.append({"block": block_arg, "level": level, "spacing": spacing, "kwargs": kwargs})
        # verts are simple and recognizable
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)
        faces = np.array([[0, 1, 0]], dtype=int)
        normals = np.zeros_like(verts)
        values = np.zeros((verts.shape[0],), dtype=float)
        return verts, faces, normals, values

    # Patch the function where it is looked up: in your module under test
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod

    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    # Act
    mc_vertices, mc_edges = marching_cubes(block, elements, spacing, extent)

    # Assert: called once per element with float(level) and spacing forwarded
    assert [c["level"] for c in calls] == [float(e) for e in elements]
    assert all(c["spacing"] == spacing for c in calls)

    # Assert: vertices are offset by extent[0], extent[2], extent[4]
    offset = np.array([extent[0], extent[2], extent[4]], dtype=float)
    expected_verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float) + offset

    assert len(mc_vertices) == 2
    assert np.allclose(mc_vertices[0], expected_verts)
    assert np.allclose(mc_vertices[1], expected_verts)

    # Assert: faces passed through
    assert len(mc_edges) == 2
    assert np.array_equal(mc_edges[0], np.array([[0, 1, 0]], dtype=int))


def test_marching_cubes_accepts_float_levels(monkeypatch):
    block = np.zeros((2, 2, 2), dtype=float)
    elements = [0.5, 1.25]
    spacing = (1.0, 1.0, 1.0)
    extent = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    seen_levels = []

    def fake_marching_cubes(block_arg, level, spacing=None, **kwargs):
        seen_levels.append(level)
        verts = np.zeros((1, 3), dtype=float)
        faces = np.zeros((1, 3), dtype=int)
        return verts, faces, None, None

    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod
    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    marching_cubes(block, elements, spacing, extent)

    # ensure cast to float(level) is preserved
    assert seen_levels == [float(x) for x in elements]


def test_marching_cubes_per_element_passes_mask_and_offsets_vertices(monkeypatch):
    block = np.zeros((3, 3, 3), dtype=float)
    element = 7.0
    spacing = (2.0, 2.0, 2.0)
    extent = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    mask = np.zeros(block.shape, dtype=bool)
    mask[1, 1, 1] = True

    captured = {}

    def fake_marching_cubes(block_arg, level, spacing=None, mask=None, **kwargs):
        captured["block"] = block_arg
        captured["level"] = level
        captured["spacing"] = spacing
        captured["mask"] = mask
        verts = np.array([[5.0, 6.0, 7.0]], dtype=float)
        faces = np.array([[0, 0, 0]], dtype=int)
        return verts, faces, None, None

    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_extraction as mod
    monkeypatch.setattr(mod.measure, "marching_cubes", fake_marching_cubes)

    vertices, edges = marching_cubes_per_element(block, element, spacing, extent, mask)

    assert captured["level"] == element
    assert captured["spacing"] == spacing
    assert captured["mask"] is mask

    offset = np.array([extent[0], extent[2], extent[4]], dtype=float)
    assert np.allclose(vertices, np.array([[5.0, 6.0, 7.0]]) + offset)
    assert np.array_equal(edges, np.array([[0, 0, 0]], dtype=int))
