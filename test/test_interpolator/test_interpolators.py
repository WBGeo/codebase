import pytest
import pandas as pd
import numpy as np
from core.structural_objects.objects import StructuralGroup, StructuralElement, InterpolationMethod
from core.grids.grid_classes import RegularGrid

from core.interpolator_components.interpolators_per_group.ordinary_kriging_per_group import \
    interpolate_group_ordinary_kriging
from core.interpolator_components.interpolators_per_group.radial_basis_function_per_group import \
    interpolate_group_radial_basis_function
from core.interpolator_components.interpolators_per_group.universal_cokriging_per_group import \
    interpolate_group_universal_cokriging
from core.interpolator_components.interpolators_per_group.loop_structural_per_group import \
    interpolate_group_loop_structural
from core.interpolator_components.interpolators_per_group.geoinr_per_group import interpolate_group_geo_inr



@pytest.fixture
def basic_group_and_data():
    # Two stacked units: A below B (horizontally layered)
    elements = [StructuralElement(name="UnitA"), StructuralElement(name="UnitB")]
    group = StructuralGroup(name="TestGroup", structural_elements=elements)

    surface_points = pd.DataFrame({
        "X": [4, 5, 6, 7, 8, 4, 5, 6, 7, 8],
        "Y": [5] * 10,
        "Z": [3] * 5 + [7] * 5,
        "formation": ["UnitA"] * 5 + ["UnitB"] * 5
    })

    orientations = pd.DataFrame({
        "X": [4, 6, 8, 5, 7],
        "Y": [5] * 5,
        "Z": [3, 3, 3, 7, 7],
        "G_x": [0] * 5,
        "G_y": [0] * 5,
        "G_z": [1] * 5,
        "formation": ["UnitA", "UnitA", "UnitA", "UnitB", "UnitB"]
    })

    grid = RegularGrid(
        extent=np.array([0, 10, 0, 10, 0, 10]),
        resolution=np.array([5, 5, 5])
    )

    return group, surface_points, orientations, grid

def test_all_interpolators_with_defaults(basic_group_and_data):
    group, surface_points, orientations, grid = basic_group_and_data

    interpolators = [
        (InterpolationMethod.ORDINARY_KRIGING, interpolate_group_ordinary_kriging),
        (InterpolationMethod.RADIAL_BASIS_FUNCTION, interpolate_group_radial_basis_function),
        (InterpolationMethod.GEOINR, interpolate_group_geo_inr),
        (InterpolationMethod.LOOP_STRUCTURAL, interpolate_group_loop_structural),
        (InterpolationMethod.UNIVERSAL_COKRIGING, interpolate_group_universal_cokriging),
    ]

    for method_enum, interpolator_func in interpolators:
        # Reset state
        group.set_scalar_field(None)
        for elem in group.structural_elements:
            elem.set_scalar_value(None)

        # Set interpolation method and parameters
        group.set_interpolation_method(method_enum)

        # Run interpolator
        if interpolator_func == interpolate_group_geo_inr:
            interpolator_func(group=group, group_surface_points_df=surface_points, group_orientations_points_df=orientations, grid=grid)
        elif interpolator_func == interpolate_group_loop_structural:
            interpolator_func(group=group, group_surface_points_df=surface_points, group_orientations_points_df=orientations, grid=grid)
        elif interpolator_func == interpolate_group_universal_cokriging:
            interpolator_func(group=group, group_surface_points_df=surface_points, group_orientations_points_df=orientations, grid=grid)
        else:
            interpolator_func(group=group, group_surface_points_df=surface_points,  grid=grid)

        # Check output
        assert group.scalar_field is not None, f"{method_enum.value} failed to produce scalar field"
        assert group.scalar_field.shape == tuple(grid.resolution), f"{method_enum.value} returned wrong shape"

        # Get assigned scalar values from structural elements
        assigned_values = [elem.scalar_value for elem in group.structural_elements]
        min_val, max_val = min(assigned_values), max(assigned_values)

        # Ensure scalar values fall within the range of the interpolated field
        field_min, field_max = np.min(group.scalar_field), np.max(group.scalar_field)

        assert field_min <= min_val <= field_max, (
            f"{method_enum.value} failed: min scalar value {min_val} not in field range [{field_min}, {field_max}]"
        )
        assert field_min <= max_val <= field_max, (
            f"{method_enum.value} failed: max scalar value {max_val} not in field range [{field_min}, {field_max}]"
        )