import os

import pandas as pd
import pytest

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL1_DATA_DIR = os.path.join(REPO_ROOT, "examples/synthetic_examples/model1/input_data/geological_data")
MODEL2_DATA_DIR = os.path.join(REPO_ROOT, "examples/synthetic_examples/model2/input_data/geological_data")


@pytest.fixture(scope="module")
def model1_structural_result():
    """Small (12^3), fault-free structural model (basement + rock1/rock2)."""
    grid = RegularGrid(extent=(0, 1000, 0, 1000, 0, 1000), resolution=(12, 12, 12))
    data_elements = InputData_StructuralElements(
        name="Model_1",
        mapping_object={"Strat_Series1": ("rock2", "rock1")},
        surface_points=pd.read_csv(os.path.join(MODEL1_DATA_DIR, "model1_surface_points_df.csv")),
        orientations=pd.read_csv(os.path.join(MODEL1_DATA_DIR, "model1_orientations_df.csv")),
    )
    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid)
    return general.compute_structural_model(frame, extract_meshes=True, verbose=False)


@pytest.fixture(scope="module")
def model1_implicit_mesh(model1_structural_result):
    """Implicit mesh built from model1_structural_result (fast, no GMSH/sfepy needed)."""
    return create_implicit_structured_mesh(geomodel_result=model1_structural_result)


def _build_model2_structural_result(*, restrict_fault_to_older_group: bool = False):
    """Small (30x15x15), faulted structural model (1 fault, 2 stratigraphic groups).

    restrict_fault_to_older_group must be applied (via set_fault_activity_by_group)
    *before* compute_structural_model -- that's when the restriction actually
    takes effect in the real lith_block (domain merging), not just the raw
    fault_activity dict -- see HydrothermalProblemBuilder's class docstring.
    """
    grid = RegularGrid(extent=(0, 2500, 0, 1000, 0, 1000), resolution=(30, 15, 15))
    data_faults = InputData_FaultElements(
        name="Faults_Model_2",
        fault_surface_points=pd.read_csv(os.path.join(MODEL2_DATA_DIR, "model2_surface_points_df.csv")),
        fault_orientations=pd.read_csv(os.path.join(MODEL2_DATA_DIR, "model2_orientations_df.csv")),
        fault_names=["fault"],
    )
    fault_frame = general_faults.build_fault_frame(input_data_fault_elements=data_faults, grid=grid)
    fault_model_result = general_faults.compute_fault_domains(fault_frame)

    data_elements = InputData_StructuralElements(
        name="Model_2",
        mapping_object={"Strat_Series2": ("rock4", "rock3"), "Strat_Series1": ("rock2", "rock1")},
        surface_points=pd.read_csv(os.path.join(MODEL2_DATA_DIR, "model2_surface_points_df.csv")),
        orientations=pd.read_csv(os.path.join(MODEL2_DATA_DIR, "model2_orientations_df.csv")),
    )
    frame = general.build_structural_frame(
        input_data_elements=data_elements, grid=grid, fault_model_results=fault_model_result
    )
    if restrict_fault_to_older_group:
        frame.set_fault_activity_by_group("fault", "Strat_Series1")
    return general.compute_structural_model(frame, extract_meshes=True, verbose=False)


@pytest.fixture(scope="module")
def model2_faulted_structural_result():
    return _build_model2_structural_result()


@pytest.fixture(scope="module")
def model2_faulted_structural_result_restricted_activity():
    """Same as model2_faulted_structural_result, but the fault is restricted
    (before compute_structural_model) to only affect its older group."""
    return _build_model2_structural_result(restrict_fault_to_older_group=True)


@pytest.fixture(scope="module")
def model2_implicit_mesh(model2_faulted_structural_result):
    """Implicit mesh built from model2_faulted_structural_result."""
    return create_implicit_structured_mesh(geomodel_result=model2_faulted_structural_result)


@pytest.fixture(scope="module")
def model2_implicit_mesh_restricted_activity(model2_faulted_structural_result_restricted_activity):
    return create_implicit_structured_mesh(geomodel_result=model2_faulted_structural_result_restricted_activity)
