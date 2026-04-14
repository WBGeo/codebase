import unittest
import pickle
import gzip
import os
import numpy as np
import pandas as pd

from core.object_components import InputData_StructuralElements, MeshResults
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh


# ------------------------
# Helper: load pkl or pkl.gz
# ------------------------
def load_pickle(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------
# Paths
# ------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

pkl_file = os.path.join(base_dir, "implicit_mesh.pkl.gz")


# ------------------------
# Test case
# ------------------------
class ImplicitStructuredMeshTestCase(unittest.TestCase):

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
            mapping_object={"Strat_Series1": ('rock2', 'rock1')},
            surface_points=pd.read_csv(
                os.path.join(data_dir, "model1_surface_points_df.csv")
            ),
            orientations=pd.read_csv(
                os.path.join(data_dir, "model1_orientations_df.csv")
            )
        )

        # Build model
        frame = general.build_structural_frame(
            input_data_elements=cls.input_data,
            grid=cls.grid
        )

        cls.structural_model_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=False
        )

        # Load reference PKL (CORE MeshResults)
        cls.mesh_pkl = load_pickle(pkl_file)

    def test_implicit_structured_mesh_matches_pkl(self):

        mesh_generated = create_implicit_structured_mesh(
            geomodel_result=self.structural_model_result
        )

        # ------------------------
        # CORE MeshResults wrapper
        # ------------------------
        mesh_gen_wrapped = MeshResults(
            nodes=mesh_generated.nodes,
            elements=mesh_generated.elements
        )

        # ------------------------
        # Nodes
        # ------------------------
        self.assertEqual(
            mesh_gen_wrapped.nodes.shape,
            self.mesh_pkl.nodes.shape,
            "Node count mismatch"
        )

        self.assertTrue(
            np.allclose(mesh_gen_wrapped.nodes, self.mesh_pkl.nodes, atol=1e-6),
            "Node coordinates mismatch"
        )

        # ------------------------
        # Elements (IMPORTANT: use SAME field name)
        # ------------------------
        self.assertEqual(
            len(mesh_gen_wrapped.elements),
            len(self.mesh_pkl.elements),
            "Number of element blocks mismatch"
        )

        for i, (cb_gen, cb_pkl) in enumerate(
            zip(mesh_gen_wrapped.elements,
                self.mesh_pkl.elements)
        ):
            self.assertEqual(
                cb_gen.type,
                cb_pkl.type,
                f"Element type mismatch in block {i}"
            )

            self.assertEqual(
                len(cb_gen.data),
                len(cb_pkl.data),
                f"Element count mismatch in block {i}"
            )

            self.assertTrue(
                np.array_equal(cb_gen.data, cb_pkl.data),
                f"Element connectivity mismatch in block {i}"
            )


if __name__ == "__main__":
    unittest.main()
