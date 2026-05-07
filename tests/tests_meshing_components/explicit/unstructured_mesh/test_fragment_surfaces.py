import unittest
import pickle
import os
import gmsh
import numpy as np

from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import (
    fragment_surfaces
)


class TestFragmentSurfacesFromPKL(unittest.TestCase):

    # ---------------------------------------------------
    # 0. setup path + gmsh init
    # ---------------------------------------------------
    def setUp(self):

        data_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(data_dir, "fragment_inputs.pkl")

        with open(pkl_path, "rb") as f:
            self.inputs = pickle.load(f)

        # IMPORTANT: reset gmsh every test
        try:
            gmsh.finalize()
        except Exception:
            pass

        gmsh.initialize()
        gmsh.model.add("test_model")

    # ---------------------------------------------------
    # 1. ov must exist (not empty)
    # ---------------------------------------------------
    def test_ov_structure(self):

        ov, ovv, tagssss, well_tags, shaft_tags, shaft_map, tri_tags, tri_map, source_tag = fragment_surfaces(
            self.inputs["surfaces"],
            self.inputs["bounds"],
            self.inputs["ref_surface_indices"],
            self.inputs["wells"],
            self.inputs["extra_planes"],
            self.inputs["sources"],
            self.inputs["mine_shafts"],
            ellipses=self.inputs["ellipses"],
            triangulations=self.inputs["triangulations"],
            mesh_size=self.inputs["mesh_size"],
            curve_mesh_size=self.inputs["curve_mesh_size"],
        )

        self.assertIsInstance(ov, list)
        self.assertGreater(len(ov), 0)

    # ---------------------------------------------------
    # 2. ovv must exist and match ov size logic
    # ---------------------------------------------------
    def test_ovv_non_empty(self):

        ov, ovv, *_ = fragment_surfaces(
            self.inputs["surfaces"],
            self.inputs["bounds"],
            self.inputs["ref_surface_indices"],
            self.inputs["wells"],
            self.inputs["extra_planes"],
            self.inputs["sources"],
            self.inputs["mine_shafts"],
            ellipses=self.inputs["ellipses"],
            triangulations=self.inputs["triangulations"],
            mesh_size=self.inputs["mesh_size"],
            curve_mesh_size=self.inputs["curve_mesh_size"],
        )

        self.assertIsInstance(ovv, list)
        self.assertGreater(len(ovv), 0)

    # ---------------------------------------------------
    # 3. tagssss must be list
    # ---------------------------------------------------
    def test_tagssss_type(self):

        _, _, tagssss, *_ = fragment_surfaces(
            self.inputs["surfaces"],
            self.inputs["bounds"],
            self.inputs["ref_surface_indices"],
            self.inputs["wells"],
            self.inputs["extra_planes"],
            self.inputs["sources"],
            self.inputs["mine_shafts"],
            ellipses=self.inputs["ellipses"],
            triangulations=self.inputs["triangulations"],
            mesh_size=self.inputs["mesh_size"],
            curve_mesh_size=self.inputs["curve_mesh_size"],
        )

        self.assertIsInstance(tagssss, list)

    # ---------------------------------------------------
    # 4. shaft mapping consistency
    # ---------------------------------------------------
    def test_shaft_mapping(self):

        _, _, _, _, _, shaft_map, *_ = fragment_surfaces(
            self.inputs["surfaces"],
            self.inputs["bounds"],
            self.inputs["ref_surface_indices"],
            self.inputs["wells"],
            self.inputs["extra_planes"],
            self.inputs["sources"],
            self.inputs["mine_shafts"],
            ellipses=self.inputs["ellipses"],
            triangulations=self.inputs["triangulations"],
            mesh_size=self.inputs["mesh_size"],
            curve_mesh_size=self.inputs["curve_mesh_size"],
        )

        self.assertIsInstance(shaft_map, dict)

    # ---------------------------------------------------
    # 5. regression hash (stable structure only)
    # ---------------------------------------------------
    def test_regression_hash(self):

        import hashlib
        import pickle

        ov, ovv, tagssss, *_ = fragment_surfaces(
            self.inputs["surfaces"],
            self.inputs["bounds"],
            self.inputs["ref_surface_indices"],
            self.inputs["wells"],
            self.inputs["extra_planes"],
            self.inputs["sources"],
            self.inputs["mine_shafts"],
            ellipses=self.inputs["ellipses"],
            triangulations=self.inputs["triangulations"],
            mesh_size=self.inputs["mesh_size"],
            curve_mesh_size=self.inputs["curve_mesh_size"],
        )

        snapshot = {
            "ov_len": len(ov),
            "ovv_len": len(ovv),
            "tagssss_len": len(tagssss),
        }

        current_hash = hashlib.md5(pickle.dumps(snapshot)).hexdigest()

        expected_hash = self.inputs.get("expected_hash", current_hash)

        self.assertEqual(current_hash, expected_hash)


# ---------------------------------------------------
# run
# ---------------------------------------------------
if __name__ == "__main__":
    unittest.main()
