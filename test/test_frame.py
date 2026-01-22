import numpy as np
import pandas as pd
import os
import pytest

from core.structuralmodeling_components.interpolators_per_group import general
from concepts.archive.objects import StructuralFrame, StructuralGroup, StructuralElement
from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

cwd = os.getcwd()

print(cwd)

@pytest.fixture
def create_frame():
    import pandas as pd

    element_names = [
        'shallow_rock3', 'shallow_rock2', 'shallow_rock1',
        'medium_rock3', 'medium_rock2', 'medium_rock1',
        'deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1'
    ]

    surface_points_data = []
    for i, name in enumerate(element_names):
        for j in range(3):  # 3 points per element
            surface_points_data.append({
                "X": i * 10 + j,
                "Y": i * 5 + j,
                "Z": i * 2 + j,
                "formation": name
            })

    surface_points = pd.DataFrame(surface_points_data)

    orientations_data = []
    for i, name in enumerate(element_names):
        for j in range(2):  # 2 orientations per element
            orientations_data.append({
                "X": i * 10 + j,
                "Y": i * 5 + j,
                "Z": i * 2 + j,
                "G_x": 0.0,
                "G_y": 0.0,
                "G_z": 1.0,
                "formation": name
            })

    orientations = pd.DataFrame(orientations_data)

    extent = np.array([0, 1000, 0, 500, 0, 1000])
    resolution = np.array([50, 50, 50])

    mapping_object = {
        "Shallow_Strat": ('shallow_rock3', 'shallow_rock2', 'shallow_rock1'),
        "Medium_Strat": ('medium_rock3', 'medium_rock2', 'medium_rock1'),
        "Deep_Strat": ('deep_rock4', 'deep_rock3', 'deep_rock2', 'deep_rock1'),
    }

    return general.build_structural_frame(mapping_object, extent, resolution, surface_points, orientations)


def test_structural_frame_structure(create_frame):
    frame = create_frame

    # Frame basics
    assert isinstance(frame, StructuralFrame)
    assert isinstance(frame.grid, RegularGrid)

    # Check grid extent and resolution
    assert frame.grid.extent.shape == (6,)
    assert frame.grid.resolution.shape == (3,)

    # Surface points and orientations
    assert isinstance(frame.surface_points, pd.DataFrame)
    assert "formation" in frame.surface_points.columns
    assert all(col in frame.surface_points.columns for col in ["X", "Y", "Z"])

    assert isinstance(frame.orientations, pd.DataFrame)
    assert all(col in frame.orientations.columns for col in ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"])

    # Structural groups
    assert isinstance(frame.structural_groups, list)
    assert len(frame.structural_groups) == 3  # 3 groups in mapping

    for group in frame.structural_groups:
        assert isinstance(group, StructuralGroup)
        assert group.name in ["Shallow_Strat", "Medium_Strat", "Deep_Strat"]
        assert isinstance(group.structural_elements, list)
        assert all(isinstance(e, StructuralElement) for e in group.structural_elements)
        assert group.interpolation_method is not None

        for elem in group.structural_elements:
            assert isinstance(elem.name, str)
            assert elem.color is not None
            assert elem.scalar_value is None  # Not set until interpolation
            assert isinstance(elem.vertices, dict)
            assert isinstance(elem.edges, dict)
