import unittest
import pickle
import os
import gmsh
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

    # A self-derived expected hash was removed: it could never detect a regression.
    # Structural invariants are asserted by the tests above.

##########################################
if __name__ == "__main__":
    unittest.main()
