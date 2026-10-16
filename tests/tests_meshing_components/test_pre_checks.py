"""
The meshing components refuse structural models computed on an OctreeGrid
(pre_check_regular_grid), both as a pre-check and inside the component.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import core.structural_modeling_components.general as general
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.pre_checks import pre_check_regular_grid
from core.object_components import InputData_StructuralElements, StructuralModelResults
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid


def _model(grid) -> StructuralModelResults:
    """One group of two horizontal surfaces."""
    xs = np.linspace(0.1, 0.9, 4)
    rows = [{"X": float(x), "Y": float(y), "Z": z, "formation": name}
            for name, z in [("rock2", 0.6), ("rock1", 0.3)] for x in xs for y in xs]
    elements = InputData_StructuralElements(
        name="layers", mapping_object={"Series": ("rock2", "rock1")},
        surface_points=pd.DataFrame(rows),
        orientations=pd.DataFrame([dict(row, G_x=0.0, G_y=0.0, G_z=1.0) for row in rows]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = general.build_structural_frame(input_data_elements=elements, grid=grid)
        return general.compute_structural_model(frame, verbose=False)


@pytest.fixture(scope="module")
def results() -> tuple[StructuralModelResults, StructuralModelResults]:
    octree = OctreeGrid(extent=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), base_resolution=(2, 2, 2), max_level=2)
    return _model(octree), _model(octree.equivalent_regular_grid())


def test_the_pre_check_refuses_an_octree_model_only(results):
    octree, regular = results
    with pytest.raises(ValueError, match="OctreeGrid"):
        pre_check_regular_grid(octree)
    pre_check_regular_grid(regular)  # must not raise


@pytest.mark.parametrize("component, kwargs", [
    (create_structured_mesh_data, {"refinement_data": [2, 2]}),
    (create_unstructured_mesh_data, {}),
    (create_implicit_structured_mesh, {}),
], ids=["structured", "unstructured", "implicit"])
def test_meshing_components_refuse_an_octree_model(results, component, kwargs):
    with pytest.raises(ValueError, match="OctreeGrid"):
        component(geomodel_result=results[0], **kwargs)
