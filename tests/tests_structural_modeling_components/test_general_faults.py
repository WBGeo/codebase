from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.object_components import FaultModelResults
from core.structural_modeling_components.general_faults import (  # type: ignore
    compute_domain_pairs_from_fault_band,
    check_fault_crosscuts_via_isovalue_bands,
    check_fault_model_not_crosscutting,
    build_fault_frame,
    compute_fault_domains,
)

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import (
    FaultElement, FaultFrame, FaultMeshType,
)


# -----------------------------------------------------------------------------
# compute_domain_pairs_from_fault_band (pure)
# -----------------------------------------------------------------------------

def test_compute_domain_pairs_from_fault_band_returns_expected_pairs_no_gradient():
    # Domain map (XYZ) has two domains: 0 on left half, 1 on right half
    domain_map_xyz = np.zeros((4, 1, 1), dtype=int)
    domain_map_xyz[2:, :, :] = 1

    # Scalar field crosses 0 at the boundary; choose scalar_value=0
    scalar_field_xyz = np.array([-2.0, -1.0, 1.0, 2.0], dtype=float).reshape(4, 1, 1)

    pairs = compute_domain_pairs_from_fault_band(
        domain_map_xyz=domain_map_xyz,
        scalar_field_xyz=scalar_field_xyz,
        scalar_value=0.0,
        spacing_xyz=(1.0, 1.0, 1.0),
        voxels=2.0,          # thick enough to include near-surface cells
        use_gradient=False,  # tol_scalar = thickness_world
    )

    assert pairs == frozenset({(0, 1)})


def test_compute_domain_pairs_from_fault_band_empty_band_yields_empty_pairs():
    domain_map_xyz = np.zeros((3, 1, 1), dtype=int)
    scalar_field_xyz = np.array([10.0, 11.0, 12.0], dtype=float).reshape(3, 1, 1)

    # scalar_value far away, band will be empty
    pairs = compute_domain_pairs_from_fault_band(
        domain_map_xyz=domain_map_xyz,
        scalar_field_xyz=scalar_field_xyz,
        scalar_value=0.0,
        spacing_xyz=(1.0, 1.0, 1.0),
        voxels=1.0,
        use_gradient=False,
    )
    assert pairs == frozenset()


def test_compute_domain_pairs_from_fault_band_gradient_guard_handles_flat_field():
    domain_map_xyz = np.zeros((3, 2, 2), dtype=int)
    domain_map_xyz[2:, :, :] = 1

    # Flat scalar field -> gradient magnitude = 0 everywhere (guard path)
    scalar_field_xyz = np.ones((3, 2, 2), dtype=float) * 5.0

    pairs = compute_domain_pairs_from_fault_band(
        domain_map_xyz=domain_map_xyz,
        scalar_field_xyz=scalar_field_xyz,
        scalar_value=5.0,
        spacing_xyz=(1.0, 1.0, 1.0),
        voxels=1.0,
        use_gradient=True,
    )

    # With a perfectly flat field: band includes everything,
    # but pos/neg split uses > and <=, so pos_ids is empty -> no pairs.
    assert pairs == frozenset()



# -----------------------------------------------------------------------------
# check_fault_crosscuts_via_isovalue_bands
# -----------------------------------------------------------------------------

def test_check_fault_crosscuts_returns_when_less_than_two_valid_faults():
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(2, 2, 2))
    f1 = FaultElement("F1")
    # missing scalar_value / scalar_field -> ignored
    ff = FaultFrame([f1])
    ff.set_grid(grid)

    # Should not raise
    check_fault_crosscuts_via_isovalue_bands(ff)


def test_check_fault_crosscuts_raises_on_band_overlap_without_gradient():
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(3, 3, 3))
    ff = FaultFrame(fault_elements=[FaultElement("F1"), FaultElement("F2")])
    ff.set_grid(grid)

    # Create scalar fields where both have isovalue=0 and overlap band at center voxel
    sf1 = np.zeros(grid.resolution, dtype=float)
    sf2 = np.zeros(grid.resolution, dtype=float)

    ff.fault_elements[0].set_scalar_field(sf1)
    ff.fault_elements[0].set_scalar_value(0.0)

    ff.fault_elements[1].set_scalar_field(sf2)
    ff.fault_elements[1].set_scalar_value(0.0)

    with pytest.raises(ValueError, match="crosscuts"):
        check_fault_crosscuts_via_isovalue_bands(ff, thickness_world=0.1, use_gradient=False)


def test_check_fault_model_not_crosscutting_raises_when_cross_cutting_true():
    ff = FaultFrame(fault_elements=[FaultElement("F1")])
    fault_model = FaultModelResults(fault_frame=ff, cross_cutting=True)

    with pytest.raises(ValueError, match="cross-cutting"):
        check_fault_model_not_crosscutting(fault_model)


def test_check_fault_model_not_crosscutting_passes_when_cross_cutting_false():
    ff = FaultFrame(fault_elements=[FaultElement("F1")])
    fault_model = FaultModelResults(fault_frame=ff, cross_cutting=False)

    check_fault_model_not_crosscutting(fault_model)  # must not raise


def test_check_fault_model_not_crosscutting_passes_when_none():
    check_fault_model_not_crosscutting(None)  # must not raise -- fault_model is optional


def test_check_fault_crosscuts_does_not_raise_when_bands_do_not_overlap():
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(3, 3, 3))
    ff = FaultFrame([FaultElement("F1"), FaultElement("F2")])
    ff.set_grid(grid)

    # F1 band: only x==0 slice is near level 0
    sf1 = np.ones(grid.resolution, dtype=float) * 10.0
    sf1[0, :, :] = 0.0

    # F2 band: only x==2 slice is near level 100
    sf2 = np.ones(grid.resolution, dtype=float) * 0.0
    sf2[2, :, :] = 100.0

    ff.fault_elements[0].set_scalar_field(sf1)
    ff.fault_elements[0].set_scalar_value(0.0)

    ff.fault_elements[1].set_scalar_field(sf2)
    ff.fault_elements[1].set_scalar_value(100.0)

    # Small thickness so only the exact slices count as "in band"
    check_fault_crosscuts_via_isovalue_bands(ff, thickness_world=0.01, use_gradient=False)



# -----------------------------------------------------------------------------
# build_fault_frame (wiring)
# -----------------------------------------------------------------------------

class FakeInputFaultElements:
    def __init__(self, fault_names, fault_surface_points, fault_orientations):
        self.fault_names = fault_names
        self.fault_surface_points = fault_surface_points
        self.fault_orientations = fault_orientations


def test_build_fault_frame_creates_elements_reversed_and_sets_defaults():
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(2, 2, 2))

    fault_names = ["F_old", "F_young"]
    sp = pd.DataFrame({"X": [0.0, 0.5], "Y": [0.0, 0.0], "Z": [0.0, 0.0], "formation": ["F_old", "F_young"]})
    ori = pd.DataFrame({
        "X": [0.0, 0.5], "Y": [0.0, 0.0], "Z": [0.0, 0.0],
        "G_x": [1.0, 1.0], "G_y": [0.0, 0.0], "G_z": [0.0, 0.0],
        "formation": ["F_old", "F_young"],
    })

    inp = FakeInputFaultElements(fault_names=fault_names, fault_surface_points=sp, fault_orientations=ori)

    ff = build_fault_frame(inp, grid)

    assert ff.grid is grid
    assert ff.fault_surface_points_df is not None
    assert ff.fault_orientations_df is not None

    # Code builds elements in reversed(zip(fault_names,...)) -> ["F_young","F_old"]
    assert [f.name for f in ff.fault_elements] == ["F_young", "F_old"]
    assert all(f.color == "#000000" for f in ff.fault_elements)


# -----------------------------------------------------------------------------
# compute_fault_domains -- end to end, non-cross-cutting vs. cross-cutting
# -----------------------------------------------------------------------------

def _plane_points(axis: str, level: float, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Surface points + orientation rows for a flat fault plane normal to `axis`."""
    coords = np.linspace(0.1, 0.9, 5)
    rows_sp, rows_ori = [], []
    normal = {"G_x": 0.0, "G_y": 0.0, "G_z": 0.0}
    normal[f"G_{axis}"] = 1.0
    for u in coords:
        for v in coords:
            point = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            axes = [a for a in ("x", "y", "z") if a != axis]
            point[axes[0].upper()] = u
            point[axes[1].upper()] = v
            point[axis.upper()] = level
            rows_sp.append({**point, "formation": name})
            rows_ori.append({**point, **normal, "formation": name})
    return pd.DataFrame(rows_sp), pd.DataFrame(rows_ori)


def _two_fault_frame(level_a: float, axis_a: str, level_b: float, axis_b: str):
    grid = RegularGrid(extent=(0, 1, 0, 1, 0, 1), resolution=(6, 6, 6))
    sp_a, ori_a = _plane_points(axis_a, level_a, "F_a")
    sp_b, ori_b = _plane_points(axis_b, level_b, "F_b")
    inp = FakeInputFaultElements(
        fault_names=["F_a", "F_b"],
        fault_surface_points=pd.concat([sp_a, sp_b], ignore_index=True),
        fault_orientations=pd.concat([ori_a, ori_b], ignore_index=True),
    )
    return build_fault_frame(inp, grid)


def test_compute_fault_domains_non_crosscutting_populates_domain_data():
    # Two parallel vertical planes (both normal to X) -- never cross.
    ff = _two_fault_frame(0.3, "x", 0.7, "x")

    result = compute_fault_domains(ff)

    assert result.cross_cutting is False
    assert result.fault_frame.domain_map is not None
    assert result.fault_frame.domain_masks
    for fault in result.fault_frame.fault_elements:
        assert fault.domain_pairs is not None
        assert fault.separated_domains is not None
        vertices, _ = fault.get_mesh(FaultMeshType.UNMASKED)  # surface meshes were still extracted
        assert len(vertices) > 0


def test_compute_fault_domains_crosscutting_skips_domaining_but_keeps_meshes():
    # A vertical plane (normal X) and a horizontal plane (normal Z) necessarily
    # cross inside the unit cube (at X=0.5, Z=0.5, for every Y).
    ff = _two_fault_frame(0.5, "x", 0.5, "z")

    result = compute_fault_domains(ff)

    assert result.cross_cutting is True
    # Domaining skipped entirely -- left at the safe None defaults.
    assert result.fault_frame.domain_map is None
    for fault in result.fault_frame.fault_elements:
        assert fault.domain_pairs is None
        assert fault.separated_domains is None
        # Surface meshes are still extracted so the model can be inspected.
        assert fault.scalar_field is not None
        vertices, _ = fault.get_mesh(FaultMeshType.UNMASKED)
        assert len(vertices) > 0
