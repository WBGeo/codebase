import unittest
import os
import pandas as pd
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.object_components import (
    InputData_StructuralElements,
    InputData_FaultElements,
    StructuralModelResults
)
from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.unstructured.create_clean_surface import data_preparation

class TestDataPreprationModel2(unittest.TestCase):

    def setUp(self):

        data_dir = os.path.dirname(__file__) + "/geological_data/"
        self.surface_path = os.path.join(
            data_dir,
            "model2_surface_points_df.csv"
        )

        self.orient_path = os.path.join(
            data_dir,
            "model2_orientations_df.csv"
        )

        self.grid = RegularGrid(
            extent=(0, 2500, 0, 1000, 0, 1000),
            resolution=(125, 50, 50)
        )

        self.surface_df = pd.read_csv(self.surface_path)
        self.orient_df = pd.read_csv(self.orient_path)

    # FULL PIPELINE FIXTURE
    def build_model(self):

        data_faults = InputData_FaultElements(
            name="Faults_Model_2",
            fault_surface_points=self.surface_df,
            fault_orientations=self.orient_df,
            fault_names=["fault"]
        )

        fault_frame = general_faults.build_fault_frame(
            input_data_fault_elements=data_faults,
            grid=self.grid
        )

        fault_model_result = general_faults.compute_fault_domains(fault_frame)

        data_elements = InputData_StructuralElements(
            name="Model_2",
            mapping_object={
                "Strat_Series2": ("rock4", "rock3"),
                "Strat_Series1": ("rock2", "rock1")
            },
            surface_points=self.surface_df,
            orientations=self.orient_df
        )

        frame = general.build_structural_frame(
            input_data_elements=data_elements,
            grid=self.grid,
            fault_model_results=fault_model_result
        )

        structural_model_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=True,
        )

        return StructuralModelResults(structural_frame=frame)

    # TEST 1: FULL RUN
    def test_data_prepration_full_run(self):

        geomodel = self.build_model()

        cleaned_surfaces, ref_indices, grid_litho = data_preparation(
            geomodel,
            DISTANCE_THRESHOLD=50,
            PROJECTION_THRESHOLD=60,
            EXTRUSION_FACTOR=100,
            z_threshold=10
        )

        # basic sanity checks
        self.assertIsInstance(cleaned_surfaces, list)
        self.assertIsInstance(ref_indices, dict)
        self.assertTrue(len(cleaned_surfaces) > 0)

        self.assertTrue(hasattr(grid_litho, "shape") or isinstance(grid_litho, pd.DataFrame))

    # TEST 2: lith_block MUST EXIST
    def test_lith_block_exists(self):

        geomodel = self.build_model()
        frame = geomodel.structural_frame

        self.assertIsNotNone(
            frame.lith_block,
            "lith_block is None → structural model failed before data_prepration"
        )

    # TEST 3: STRUCTURE CONSISTENCY
    def test_ref_surface_indices(self):

        geomodel = self.build_model()

        cleaned_surfaces, ref_indices, _ = data_preparation(
            geomodel,
            DISTANCE_THRESHOLD=50,
            PROJECTION_THRESHOLD=60,
            EXTRUSION_FACTOR=100,
            z_threshold=10
        )

        # ref indices should not be empty if faults exist
        self.assertIsInstance(ref_indices, dict)

##################################
if __name__ == "__main__":
    unittest.main()
