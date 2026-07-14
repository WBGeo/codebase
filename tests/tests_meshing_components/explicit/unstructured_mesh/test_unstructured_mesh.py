import unittest
import os
import numpy as np
import pandas as pd
from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
    load_wells_from_csv,
    load_shafts_from_csv,
    load_sources_from_csv,
    load_planes_from_csv,
    load_ellipses_from_csv,
    load_triangulations_planes_from_csv,
)

########
# Paths
########
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../loading_engineering_objects/data/geological_data/"
)

engineering_dir = os.path.join(
    base_dir,
    "../../loading_engineering_objects/data/engineering_objects/"
)


###########
# TEST CASE
###########
class UnstructuredMeshTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Grid
        cls.grid = RegularGrid(
            extent=(0, 1000, 0, 1000, 0, 1000),
            resolution=(50, 50, 50)
        )

        # Input data
        cls.input_data = InputData_StructuralElements(
            name='Model_1',
            mapping_object={"Strat_Series": ('rock2', 'rock1')},
            surface_points=pd.read_csv(
                os.path.join(data_dir, "model1_surface_points_df.csv")
            ),
            orientations=pd.read_csv(
                os.path.join(data_dir, "model1_orientations_df.csv")
            )
        )

        # Structural model
        frame = general.build_structural_frame(
            input_data_elements=cls.input_data,
            grid=cls.grid
        )

        cls.geomodel_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=False
        )

        # Engineering objects
        cls.wells = load_wells_from_csv(
            os.path.join(engineering_dir, "model_1_wells.csv")
        )
        cls.shafts = load_shafts_from_csv(
            os.path.join(engineering_dir, "model_1_shafts.csv")
        )
        cls.sources = load_sources_from_csv(
            os.path.join(engineering_dir, "model_1_sources.csv")
        )
        cls.planes = load_planes_from_csv(
            os.path.join(engineering_dir, "model_1_planes.csv")
        )
        cls.ellipses = load_ellipses_from_csv(
            os.path.join(engineering_dir, "model_1_ellipses.csv")
        )
        cls.triangulations = load_triangulations_planes_from_csv(
            os.path.join(engineering_dir, "seismic_plane_new_offset.csv")
        )

    # TEST
    def test_unstructured_mesh_basic_properties(self):

        mesh_generated = create_unstructured_mesh_data(
            geomodel_result=self.geomodel_result,
            wells=self.wells,
            sources=self.sources,
            shafts=self.shafts,
            triangulations=self.triangulations,
            extra_planes=self.planes,
            ellipses=self.ellipses,
            mesh_size=75,
            curve_mesh_size=5
        )

        nodes = mesh_generated.nodes
        elements = mesh_generated.elements

        # Basic existence
        self.assertIsNotNone(nodes)
        self.assertIsNotNone(elements)

        self.assertGreater(len(nodes), 0, "No nodes generated")
        self.assertGreater(len(elements), 0, "No element blocks generated")

        # Node structure
        self.assertEqual(nodes.shape[1], 3, "Nodes must be 3D coordinates")

        # Numerical sanity
        self.assertFalse(np.isnan(nodes).any(), "NaN values in nodes")
        self.assertFalse(np.isinf(nodes).any(), "Inf values in nodes")

        # Bounds check (based on grid)
        tol = 1e-6
        self.assertTrue(
            np.all((nodes >= -tol) & (nodes <= 1000 + tol)),
            "Nodes outside expected domain"
        )

        # Size sanity (tolerant)
        self.assertTrue(
            4000 < len(nodes) < 18000,
            f"Unexpected number of nodes: {len(nodes)}"
        )

        # Uniqueness (no duplicate nodes)
        unique_nodes = np.unique(nodes, axis=0)
        self.assertEqual(
            len(unique_nodes),
            len(nodes),
            "Duplicate nodes detected"
        )

        # Elements validity
        n_nodes = len(nodes)

        for i, block in enumerate(elements):

            self.assertGreater(
                len(block.data),
                0,
                f"Block {i} has no elements"
            )

            self.assertTrue(
                np.all(block.data < n_nodes),
                f"Block {i}: invalid node indices (too large)"
            )

            self.assertTrue(
                np.all(block.data >= 0),
                f"Block {i}: invalid node indices (negative)"
            )

        # mapping_litho defaults to "automatic_centers" -- confirm the real
        # lithology mapping (cell_data["block_id"]) is still produced under
        # that default post-rename (see test_lithology_mapping_mode.py for
        # the enum-level automatic_centers/automatic_corners rename checks;
        # deliberately not a second real GMSH meshing call here -- observed
        # GMSH physical-group-tag state bleeding across separate
        # create_unstructured_mesh_data calls within one pytest process).
        self.assertIn("block_id", mesh_generated.cell_data or {})

######################################
if __name__ == "__main__":
    unittest.main()
