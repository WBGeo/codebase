import os
import pickle
import unittest

import numpy as np
import pandas as pd

from concepts.archive.universal_cokriging import universal_cokriging_interpolator
from core.object_components import InputData, GeomodelResults

data_dir = os.path.dirname(__file__) + "/../examples/data/"


# Perform regression test on the universal cokriging
# based on model 2
class TestUniversalCokriging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # every test method of this class uses the data_test InputData
        cls.data_test = InputData(name='Model 2',
                                  extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                                  resolution=np.array([20, 20, 20]),
                                  mapping_object={"Strat_Series": ('rock2', 'rock1')},
                                  surface_points=pd.read_csv(
                                      data_dir + "model2_surface_points_df.csv"),
                                  orientations=pd.read_csv(
                                      data_dir + "model2_orientations_df.csv"),
                                  )

    def test_data_folder_existing(self):
        # if these asserts fail, every other test will also likely fail -> just helps us to narrow failures down
        self.assertTrue(os.path.exists(data_dir), "data directory missing")
        self.assertTrue(os.path.exists(data_dir + "model2_surface_points_df.csv"), "csv file missing")

    def test_run_kriging_regression(self):
        results_test = universal_cokriging_interpolator(self.data_test)
        self.assertIsNotNone(results_test)
        self.assertEqual(results_test.name, self.data_test.name)
        # And now we perform some regression testing: As in, detect breaking changes to previous calculated values:
        # see https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html
        # uncomment the following line to update the data
        # self.update_data_files(results_test)
        # Load a universal co-kriging results - when we deviate from this object, an error occured
        with open(os.path.dirname(__file__) + '/model2_universal_cokriging_res.pkl', 'rb') as f:
            results_to_regression_test_against: GeomodelResults = pickle.load(f)
        np.testing.assert_array_equal(results_test.lith_block, results_to_regression_test_against.lith_block,
                                      "lith block data missmatch")
        self.assertEqual(len(results_test.surface_meshes_vertices),
                                      len(results_to_regression_test_against.surface_meshes_vertices),
                                      "surface meshes vertices length missmatch")

        for i in range(0, len(results_test.surface_meshes_vertices)):
            np.testing.assert_array_equal(results_test.surface_meshes_vertices[i],
                                          results_to_regression_test_against.surface_meshes_vertices[i],
                                          "surface meshes vertices data missmatch " + str(i))
        self.assertEqual(len(results_test.surface_meshes_edges),
                         len(results_to_regression_test_against.surface_meshes_edges),
                         "surface meshes edges length data missmatch")
        for i in range(0, len(results_to_regression_test_against.surface_meshes_edges)):
            np.testing.assert_array_equal(results_test.surface_meshes_edges[i],
                                          results_to_regression_test_against.surface_meshes_edges[i],
                                          "surface meshes edges {} data missmatch".format(i))
        np.testing.assert_array_equal(results_test.grid, results_to_regression_test_against.grid,
                                      "surface meshes edges data missmatch")

    @classmethod
    def update_data_files(cls, results_test: GeomodelResults):
        with open(os.path.dirname(__file__) + '/model2_universal_cokriging_res.pkl', 'wb') as f:
            pickle.dump(results_test, f)


if __name__ == '__main__':
    unittest.main()
