import unittest

from core.meshing_components.explicit.unstructured.mesh_data import (
    LithoMappingMode,
    create_unstructured_mesh_data,
)


class LithoMappingModeEnumTestCase(unittest.TestCase):
    """
    Pure enum-level checks (no meshing involved) for the automatic_dev/auto ->
    automatic_centers/automatic_corners rename -- old values must be fully
    gone, not just aliased, and automatic_centers must be the new default.

    Real end-to-end confirmation that the new default still produces a
    usable block_id lithology tag through actual GMSH meshing lives in
    test_unstructured_mesh.py's existing test (which already exercises the
    default mapping_litho) rather than a second, separate real mesh here --
    GMSH carries global physical-group-tag state across separate
    create_unstructured_mesh_data calls that isn't always fully reset
    between calls within the same pytest process (observed: a 'Physical
    surface N already exists' exception when an additional real mesh was
    built here, depending on what other GMSH-using tests happened to run
    earlier in the same process) -- a pre-existing GMSH-state fragility
    unrelated to mapping_litho itself, same family of issue as the
    documented GMSH+Windows CreateProcess quirk.
    """

    def test_new_values_are_accepted(self):
        self.assertEqual(LithoMappingMode("automatic_centers"), LithoMappingMode.AUTOMATIC_CENTERS)
        self.assertEqual(LithoMappingMode("automatic_corners"), LithoMappingMode.AUTOMATIC_CORNERS)
        self.assertEqual(LithoMappingMode("manual"), LithoMappingMode.MANUAL)
        self.assertEqual(LithoMappingMode("none"), LithoMappingMode.NONE)

    def test_old_values_are_rejected(self):
        with self.assertRaises(ValueError):
            LithoMappingMode("auto")
        with self.assertRaises(ValueError):
            LithoMappingMode("automatic_dev")

    def test_default_is_automatic_centers(self):
        import inspect
        sig = inspect.signature(create_unstructured_mesh_data)
        self.assertEqual(sig.parameters["mapping_litho"].default, LithoMappingMode.AUTOMATIC_CENTERS.value)


if __name__ == "__main__":
    unittest.main()
