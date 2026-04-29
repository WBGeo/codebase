import unittest
import numpy as np
import pandas as pd
import os
import gzip
import pickle

from core.object_components import InputData_StructuralElements
from core.object_components import MeshResults
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data


# ------------------------
# Paths
# ------------------------
data_dir = os.path.join(
    os.path.dirname(__file__),
    "../../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

pkl_file = os.path.join(os.path.dirname(__file__), "structured_mesh.pkl.gz")


# ------------------------
# SAFE PICKLE LOADER
# ------------------------
def load_pickle(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------
# SAFE EXTRACTORS
# ------------------------
def get_nodes(mesh):
    if hasattr(mesh, "nodes"):
        return np.asarray(mesh.nodes)

    if hasattr(mesh, "__dict__") and "nodes" in mesh.__dict__:
        return np.asarray(mesh.__dict__["nodes"])

    raise AttributeError(f"No nodes found in {type(mesh)}")


def get_elements(mesh):
    if hasattr(mesh, "elements"):
        return mesh.elements

    if hasattr(mesh, "elements_structured"):
        return mesh.elements_structured

    if hasattr(mesh, "__dict__"):
        if "elements" in mesh.__dict__:
            return mesh.__dict__["elements"]
        if "elements_structured" in mesh.__dict__:
            return mesh.__dict__["elements_structured"]

    raise AttributeError(f"No elements found in {type(mesh)}")


# ------------------------
# TEST CASE
# ------------------------
class StructuredMeshTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.grid = RegularGrid(
            extent=(0, 1000, 0, 1000, 0, 1000),
            resolution=(50, 50, 50)
        )

        cls.input_data = InputData_StructuralElements(
            name="Model_1",
            mapping_object={"Strat_Series": ("rock2", "rock1")},
            surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
            orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv")),
        )

        frame = general.build_structural_frame(
            input_data_elements=cls.input_data,
            grid=cls.grid
        )

        cls.structural_model_result = general.compute_structural_model(
            frame,
            extract_meshes=True,
            verbose=False
        )

        cls.mesh_pkl = load_pickle(pkl_file)

    def test_structured_mesh_matches_pkl(self):

        mesh_generated = create_structured_mesh_data(
            geomodel_result=self.structural_model_result,
            refinement_data=(10, 10, 10),
            mesh_devision=(30, 30),
            z_threshold=0.1,
            tolerance=1
        )

        # Wrap generated mesh
        mesh_gen_wrapped = MeshResults(
            nodes=mesh_generated.nodes,
            elements=mesh_generated.elements
        )

        # Extract nodes
        gen_nodes = get_nodes(mesh_gen_wrapped)
        ref_nodes = get_nodes(self.mesh_pkl)

        # ------------------------
        # Shape must match
        # ------------------------
        self.assertEqual(gen_nodes.shape, ref_nodes.shape)

        # ------------------------
        # Sort nodes to avoid ordering issues
        # ------------------------
        def sort_nodes(arr):
            return arr[np.lexsort(arr.T)]

        gen_nodes_sorted = sort_nodes(gen_nodes)
        ref_nodes_sorted = sort_nodes(ref_nodes)

        # ------------------------
        # Robust comparison
        # ------------------------
        diff = np.abs(gen_nodes_sorted - ref_nodes_sorted)

        self.assertTrue(
            np.allclose(gen_nodes_sorted, ref_nodes_sorted, atol=1e-5),
            msg=f"Max node diff: {diff.max()}"
        )

        # ------------------------
        # Compare elements
        # ------------------------
        gen_elems = get_elements(mesh_gen_wrapped)
        ref_elems = get_elements(self.mesh_pkl)

        self.assertEqual(len(gen_elems), len(ref_elems))

        for i, (a, b) in enumerate(zip(gen_elems, ref_elems)):
            self.assertEqual(a.type, b.type)
            self.assertTrue(
                np.array_equal(a.data, b.data),
                msg=f"Connectivity mismatch in block {i}"
            )


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    unittest.main()
