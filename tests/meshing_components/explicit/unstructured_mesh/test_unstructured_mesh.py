import unittest
import pickle
import gzip
import os
import numpy as np
import pandas as pd

from core.object_components import MeshResults
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
# Paths
# -----------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

engineering_dir = os.path.join(data_dir, "../Engineering_objects/")

pkl_file = os.path.join(base_dir, "mesh_test.pkl.gz")


# -----------------------------
# Loader
# -----------------------------
def load_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# TEST CASE
# -----------------------------
class UnstructuredMeshTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.grid = RegularGrid(
            extent=(0, 1000, 0, 1000, 0, 1000),
            resolution=(50, 50, 50)
        )

        cls.input_data = InputData_StructuralElements(
            name='Model_1',
            mapping_object={"Strat_Series": ('rock2', 'rock1')},
            surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
            orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv"))
        )

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

        # -----------------------------
        # Wrap generated result using CORE MeshResults
        # -----------------------------
        mesh_gen = MeshResults(
            nodes=mesh_generated.nodes,
            elements=mesh_generated.elements
        )

        mesh_ref = self.mesh_pkl

        # -----------------------------
        # Nodes
        # -----------------------------
        self.assertEqual(mesh_gen.nodes.shape, mesh_ref.nodes.shape)

        self.assertTrue(
            np.allclose(mesh_gen.nodes, mesh_ref.nodes, atol=1e-6),
            "Node coordinates mismatch"
        )

        # -----------------------------
        # Elements (meshio CellBlocks)
        # -----------------------------
        self.assertEqual(
            len(mesh_gen.elements),
            len(mesh_ref.elements),
            "Number of element blocks mismatch"
        )

        for i, (a, b) in enumerate(zip(mesh_gen.elements, mesh_ref.elements)):

            self.assertEqual(a.type, b.type, f"Block {i}: type mismatch")

            self.assertEqual(
                len(a.data),
                len(b.data),
                f"Block {i}: element count mismatch"
            )

            self.assertTrue(
                np.array_equal(a.data, b.data),
                f"Block {i}: connectivity mismatch"
            )


if __name__ == "__main__":
    unittest.main()
