import unittest
import os
import numpy as np
import pandas as pd
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.object_components import InputData_StructuralElements
from core.meshing_components.explicit.structured.mesh_data import (
    prepare_surface_vertices_from_geomodel
)

###################
# Build Model1 once
###################
def build_model1():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../")
    )

    data_dir = os.path.join(
        base_dir,
        "examples/synthetic_examples/model1/input_data/geological_data"
    )

    grid = RegularGrid(
        extent=(0, 1000, 0, 1000, 0, 1000),
        resolution=(20, 20, 20)   # faster for testing
    )

    data_elements = InputData_StructuralElements(
        name="Model_1",
        mapping_object={"Strat_Series1": ("rock2", "rock1")},
        surface_points=pd.read_csv(
            os.path.join(data_dir, "model1_surface_points_df.csv")
        ),
        orientations=pd.read_csv(
            os.path.join(data_dir, "model1_orientations_df.csv")
        ),
    )

    frame = general.build_structural_frame(
        input_data_elements=data_elements,
        grid=grid
    )

    return general.compute_structural_model(
        frame,
        extract_meshes=True,
        verbose=False
    )


############
# UNIT TEST
# #########
class TestPrepareSurfaceVerticesModel1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = build_model1()

    def test_structure_is_correct(self):

        result = prepare_surface_vertices_from_geomodel(self.model)

        # extent check
        self.assertEqual(len(result.extent), 6)

        # structure size check
        self.assertEqual(len(result.surface_meshes_vertices), 3)

        # legacy padding must be empty
        self.assertEqual(result.surface_meshes_vertices[0], [])
        self.assertEqual(result.surface_meshes_vertices[1], [])

        # combined surfaces exist
        combined = result.surface_meshes_vertices[2]
        self.assertGreater(len(combined), 0)

        # each surface must be Nx3
        for surf in combined:
            self.assertEqual(surf.shape[1], 3)

    def test_no_empty_surfaces(self):

        result = prepare_surface_vertices_from_geomodel(self.model)

        combined = result.surface_meshes_vertices[2]

        for surf in combined:
            self.assertTrue(len(surf) > 0)

    def test_consistency_of_extent(self):

        result = prepare_surface_vertices_from_geomodel(self.model)

        # sanity check: min/max structure exists
        self.assertTrue(np.all(np.isfinite(result.extent)))


###########################################
if __name__ == "__main__":
    unittest.main()
