from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.structural_modeling_components.general_faults import (  # type: ignore
    compute_domain_pairs_from_fault_band,
    check_fault_crosscuts_via_isovalue_bands,
    build_fault_frame,
)

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_objects.structural_objects import FaultElement, FaultFrame


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
