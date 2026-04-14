import unittest
import pickle
import gzip
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

# -----------------------------
# Minimal wrapper for PKL
# -----------------------------
class MeshResults:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements_unstructured = elements


# -----------------------------
# Paths
# -----------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)
print(data_dir)
engineering_dir = os.path.join(data_dir, "../Engineering_objects/")

# 🔥 IMPORTANT: gz file
pkl_file = os.path.join(base_dir, "mesh_test.pkl.gz")


# -----------------------------
# Helper loader (supports gz)
# -----------------------------
def load_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class UnstructuredMeshTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # -----------------------------
        # Grid
        # -----------------------------
        cls.grid = RegularGrid(
            extent=(0, 1000, 0, 1000, 0, 1000),
            resolution=(50, 50, 50)
        )

        # -----------------------------
        # Input data
        # -----------------------------
        cls.input_data = InputData_StructuralElements(
            name='Model_1',
            mapping_object={"Strat_Series": ('rock2', 'rock1')},
            surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
            orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv"))
        )

        # -----------------------------
        # Build structural model
        # -----------------------------
        frame = general.build_structural_frame(
            input_data_elements=cls.input_data,
            grid=cls.grid
        )

        cls.geomodel_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=False
        )

        # -----------------------------
        # Engineering objects
        # -----------------------------
        cls.wells = load_wells_from_csv(os.path.join(engineering_dir, "model_1_wells.csv"))
        cls.shafts = load_shafts_from_csv(os.path.join(engineering_dir, "model_1_shafts.csv"))
        cls.sources = load_sources_from_csv(os.path.join(engineering_dir, "model_1_sources.csv"))
        cls.planes = load_planes_from_csv(os.path.join(engineering_dir, "model_1_planes.csv"))
        cls.ellipses = load_ellipses_from_csv(os.path.join(engineering_dir, "model_1_ellipses.csv"))
        cls.triangulations = load_triangulations_planes_from_csv(
            os.path.join(engineering_dir, "seismic_plane_new_offset.csv")
        )

        # -----------------------------
        # Load PKL (compressed)
        # -----------------------------
        cls.mesh_pkl = load_pickle(pkl_file)

    def test_unstructured_mesh_matches_pkl(self):

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

        mesh_gen_wrapped = MeshResults(
            nodes=mesh_generated.nodes,
            elements=mesh_generated.elements
        )

        # -----------------------------
        # Nodes (shape only — safe for unstructured mesh)
        # -----------------------------
        self.assertEqual(
            mesh_gen_wrapped.nodes.shape,
            self.mesh_pkl.nodes.shape,
            "Node count mismatch"
        )

        self.assertTrue(
            np.allclose(mesh_gen_wrapped.nodes, self.mesh_pkl.nodes, atol=1e-6),
            "Node coordinates mismatch"
        )

        # -----------------------------
        # Elements
        # -----------------------------
        self.assertEqual(
            len(mesh_gen_wrapped.elements_unstructured),
            len(self.mesh_pkl.elements_unstructured),
            "Number of element blocks mismatch"
        )

        for i, (cb_gen, cb_pkl) in enumerate(
            zip(mesh_gen_wrapped.elements_unstructured,
                self.mesh_pkl.elements_unstructured)
        ):

            self.assertEqual(cb_gen.type, cb_pkl.type,
                             f"Element type mismatch in block {i}")

            self.assertEqual(len(cb_gen.data), len(cb_pkl.data),
                             f"Element count mismatch in block {i}")

            self.assertTrue(
                np.array_equal(cb_gen.data, cb_pkl.data),
                f"Element connectivity mismatch in block {i}"
            )


if __name__ == "__main__":
    unittest.main()
