import unittest
import os
import numpy as np
import pandas as pd
from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.meshing_components.explicit.structured.mesh_data import (
    create_structured_mesh_data, pre_check_no_faults_for_structured_mesh,
)

class StructuredMeshTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        base_dir = os.path.dirname(__file__)

        data_dir = os.path.join(
            base_dir,
            "../../../../examples/synthetic_examples/model1/input_data/geological_data/"
        )

        # Grid
        cls.grid = RegularGrid(
            extent=(0, 1000, 0, 1000, 0, 1000),
            resolution=(50, 50, 50)
        )

        # Input data
        cls.input_data = InputData_StructuralElements(
            name="Model_1",
            mapping_object={"Strat_Series": ("rock2", "rock1")},
            surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
            orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv")),
        )

        # Structural model
        frame = general.build_structural_frame(
            input_data_elements=cls.input_data,
            grid=cls.grid
        )

        cls.structural_model_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=False
        )

    def test_structured_mesh_basic_properties(self):

        mesh_generated = create_structured_mesh_data(
            geomodel_result=self.structural_model_result,
            refinement_data=(10, 10, 10),
            mesh_division=(20, 20),
            z_threshold=0.1,
            tolerance=1
        )

        nodes = mesh_generated.nodes
        elements = mesh_generated.elements

        # Basic existence
        self.assertIsNotNone(nodes)
        self.assertIsNotNone(elements)

        self.assertGreater(len(nodes), 0, "No nodes generated")
        self.assertGreater(len(elements), 0, "No element blocks generated")

        # Node structure
        self.assertEqual(nodes.shape[1], 3, "Nodes must be 3D")

        # Numerical sanity
        self.assertFalse(np.isnan(nodes).any(), "NaNs in nodes")
        self.assertFalse(np.isinf(nodes).any(), "Infs in nodes")

        # Bounds check
        self.assertTrue(
            np.all((nodes >= 0) & (nodes <= 1000)),
            "Nodes outside domain"
        )

        # Size sanity
        self.assertTrue(
            2000 < len(nodes) < 20000,
            f"Unexpected number of nodes: {len(nodes)}"
        )

        # Uniqueness
        unique_nodes = np.unique(nodes, axis=0)
        self.assertEqual(len(unique_nodes), len(nodes), "Duplicate nodes")

        # Elements validity
        n_nodes = len(nodes)

        for i, block in enumerate(elements):

            self.assertGreater(len(block.data), 0, f"Block {i} empty")

            self.assertTrue(
                np.all(block.data < n_nodes),
                f"Block {i}: invalid indices (too large)"
            )

            self.assertTrue(
                np.all(block.data >= 0),
                f"Block {i}: negative indices"
            )

        # Optional: check structured nature
        # (e.g. layers in Z direction exist)
        z_vals = nodes[:, 2]
        self.assertGreater(
            len(np.unique(np.round(z_vals, 3))),
            5,
            "Too few Z layers → not structured?"
        )

    def test_pre_check_passes_for_fault_free_model(self):
        # cls.structural_model_result (setUpClass) has no faults.
        pre_check_no_faults_for_structured_mesh(self.structural_model_result)  # must not raise

    def test_structured_mesh_rejects_faulted_structural_model(self):
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            base_dir, "../../../../examples/synthetic_examples/model2/input_data/geological_data/"
        )
        grid = RegularGrid(extent=(0, 2500, 0, 1000, 0, 1000), resolution=(15, 8, 8))

        data_faults = InputData_FaultElements(
            name="Faults_Model_2",
            fault_surface_points=pd.read_csv(os.path.join(data_dir, "model2_surface_points_df.csv")),
            fault_orientations=pd.read_csv(os.path.join(data_dir, "model2_orientations_df.csv")),
            fault_names=["fault"],
        )
        fault_frame = general_faults.build_fault_frame(input_data_fault_elements=data_faults, grid=grid)
        fault_model_result = general_faults.compute_fault_domains(fault_frame)

        data_elements = InputData_StructuralElements(
            name="Model_2",
            mapping_object={"Strat_Series2": ("rock4", "rock3"), "Strat_Series1": ("rock2", "rock1")},
            surface_points=pd.read_csv(os.path.join(data_dir, "model2_surface_points_df.csv")),
            orientations=pd.read_csv(os.path.join(data_dir, "model2_orientations_df.csv")),
        )
        frame = general.build_structural_frame(
            input_data_elements=data_elements, grid=grid, fault_model_results=fault_model_result
        )
        faulted_result = general.compute_structural_model(frame, extract_meshes=True, verbose=False)

        with self.assertRaisesRegex(ValueError, "does not support structural models with faults"):
            pre_check_no_faults_for_structured_mesh(faulted_result)

        with self.assertRaisesRegex(ValueError, "does not support structural models with faults"):
            create_structured_mesh_data(geomodel_result=faulted_result, refinement_data=[5, 5, 5])


########################################
if __name__ == "__main__":
    unittest.main()
