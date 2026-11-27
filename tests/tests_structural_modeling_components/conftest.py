"""Small structural models of synthetic models 1 and 2 (model 2 also on an octree grid), shared by the structural modeling tests."""
import os

import pandas as pd
import pytest

from core.object_components import InputData_FaultElements, InputData_StructuralElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid, RegularGrid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO_ROOT, "examples", "synthetic_examples", "model{0}", "input_data", "geological_data",
                    "model{0}_{1}_df.csv")


def _read(model, kind):
    return pd.read_csv(DATA.format(model, kind))


@pytest.fixture(scope="module")
def small_model1():
    grid = RegularGrid(extent=(0, 1000, 0, 1000, 0, 1000), resolution=(14, 12, 10))
    elements = InputData_StructuralElements(name="Model_1", mapping_object={"Strat_Series1": ("rock2", "rock1")},
                                            surface_points=_read(1, "surface_points"),
                                            orientations=_read(1, "orientations"))
    frame = general.build_structural_frame(input_data_elements=elements, grid=grid)
    return general.compute_structural_model(frame, extract_meshes=True, verbose=False)


def _model2(grid):
    faults = general_faults.compute_fault_domains(general_faults.build_fault_frame(
        input_data_fault_elements=InputData_FaultElements(
            name="Faults_Model_2", fault_surface_points=_read(2, "surface_points"),
            fault_orientations=_read(2, "orientations"), fault_names=["fault"]),
        grid=grid))
    elements = InputData_StructuralElements(
        name="Model_2", mapping_object={"Strat_Series2": ("rock4", "rock3"), "Strat_Series1": ("rock2", "rock1")},
        surface_points=_read(2, "surface_points"), orientations=_read(2, "orientations"))
    frame = general.build_structural_frame(input_data_elements=elements, grid=grid, fault_model_results=faults)
    frame.set_fault_activity_by_group("fault", "Strat_Series1")
    return general.compute_structural_model(frame, extract_meshes=True, verbose=False)


@pytest.fixture(scope="module")
def small_model2():
    """Faulted, with the fault restricted to the older group, so fault_activity is not the default."""
    return _model2(RegularGrid(extent=(0, 2500, 0, 1000, 0, 1000), resolution=(30, 15, 12)))


@pytest.fixture(scope="module")
def small_model2_octree():
    """As small_model2, on an octree grid (fault model and structural model on their own octree cells)."""
    return _model2(OctreeGrid(extent=(0, 2500, 0, 1000, 0, 1000), base_resolution=(5, 2, 2), max_level=2))
