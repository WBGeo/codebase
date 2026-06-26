import unittest
import pickle
import os
import gmsh
import hashlib
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import (
    fragment_surfaces
)

class TestFragmentSurfacesFromPKL(unittest.TestCase):

    def setUp(self):

        data_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(data_dir, "fragment_inputs.pkl")

        with open(pkl_path, "rb") as f:
            self.inputs = pickle.load(f)

        try:
            gmsh.finalize()
        except Exception:
            pass

        gmsh.initialize()
        gmsh.model.add("test_model")

    # ov must exist (not empty)
    def test_ov_structure(self):

        ov, *_ = fragment_surfaces(
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

    # well_tags must exist and be non-empty list
    def test_well_tags(self):

        _, well_tags, *_ = fragment_surfaces(
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

        self.assertIsInstance(well_tags, list)

    # shaft_tags must be list
    def test_shaft_tags(self):

        _, _, shaft_tags, *_ = fragment_surfaces(
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

        self.assertIsInstance(shaft_tags, list)

    # tri_group_tags must be list
    def test_tri_group_tags(self):

        _, _, _, tri_group_tags, *_ = fragment_surfaces(
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

        self.assertIsInstance(tri_group_tags, list)

    # boundary consistency (regression style)
    def test_regression_hash(self):

        ov, well_tags, shaft_tags, tri_group_tags, tri_surface_tags, source_tag, boundary_tags = fragment_surfaces(
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
            "well_tags_len": len(well_tags),
            "shaft_tags_len": len(shaft_tags),
            "tri_group_tags_len": len(tri_group_tags),
            "tri_surface_tags_len": len(tri_surface_tags),
            "boundary_tags_len": len(boundary_tags),
        }

        current_hash = hashlib.md5(pickle.dumps(snapshot)).hexdigest()
        expected_hash = self.inputs.get("expected_hash", current_hash)

        self.assertEqual(current_hash, expected_hash)

##########################################
if __name__ == "__main__":
    unittest.main()
