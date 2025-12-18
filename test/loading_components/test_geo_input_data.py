import pickle
import unittest
import os

import numpy as np

from core.loading_components.geo_input_data import geo_input_data_fix
from core.object_components import InputData

data_dir = os.path.dirname(__file__) + "/../../examples/data/"


class GeoInputDataTestCase(unittest.TestCase):
    """This test"""
    @classmethod
    def setUpClass(cls):
        # every test method of this class uses the data_test InputData

        # call a @wbeo_component manually
        cls.data_test = geo_input_data_fix('Model 2',
                                           extent_str='0, 1000, 0, 1000, 0, 1000',
                                           resolution_str='20, 20, 20',
                                           surface_points_file=data_dir + "model2_surface_points_df.csv",
                                           orientations_file=data_dir + "model2_orientations_df.csv",
                                           mapping_file=data_dir + "model_2_mapping.json",
                                           with_faults=False
                                           )
        # the loaded data is evaluated in the methods of this test-suite

    def test_surface_points_formation(self):
        # Check that the formation data of the surface_points matches our expected values
        self.assertEqual(len(self.data_test.surface_points['formation']), 36)
        self.assertEqual(self.data_test.surface_points['formation'][0], 'rock2')
        self.assertEqual(self.data_test.surface_points['formation'][17], 'rock2')
        self.assertEqual(self.data_test.surface_points['formation'][18], 'rock1')
        self.assertEqual(self.data_test.surface_points['formation'][35], 'rock1')

    def test_regression(self):
        # And now we perform some regression testing: As in, detect breaking changes to previous calculated values:
        # see https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html
        # uncomment the following line to update the data
        # self.update_data_files(self.data_test)
        # Load a universal cokriging results - when we deviate from this object, an error occured
        with open(os.path.dirname(__file__) + '/model2_input_data.pkl', 'rb') as f:
            regression_data: InputData = pickle.load(f)
        self.assertEqual(self.data_test.name, regression_data.name)
        np.testing.assert_array_equal(self.data_test.extent, regression_data.extent)
        np.testing.assert_array_equal(self.data_test.resolution, regression_data.resolution)
        self.assertEqual(self.data_test.mapping_object, regression_data.mapping_object)
        np.testing.assert_array_equal(self.data_test.surface_points, regression_data.surface_points)
        np.testing.assert_array_equal(self.data_test.orientations, regression_data.orientations)
        self.assertEqual(self.data_test.faults, regression_data.faults)
        np.testing.assert_array_equal(self.data_test.fault_relations, regression_data.fault_relations)

    # to update the pickled data (data of the regression test)
    # @classmethod
    # def update_data_files(cls, data: InputData):
    #     with open(os.path.dirname(__file__) + '/model2_input_data.pkl', 'wb') as f:
    #         pickle.dump(data, f)


class GeoInputDataNegativeTestCase(unittest.TestCase):
    """Test negative/incorrect inputs"""
    def test_illegal_extend(self):
        # when we pass an incorrect extent_str, we expect a ValueError with a given message
        with self.assertRaises(ValueError) as err:
            geo_input_data_fix('Model 2',
                               extent_str='IllegalExtend',  # <-- should throw
                               resolution_str='20, 20, 20',
                               surface_points_file=data_dir + "model2_surface_points_df.csv",
                               orientations_file=data_dir + "model2_orientations_df.csv",
                               mapping_file=data_dir + "model_2_mapping.json",
                               with_faults=False
                               )
        self.assertEqual("Illegal format for extent_str: ", err.exception.args[0])

    def test_illegal_orientations(self):
        # when we pass an incorrect orientations_file, we expect a FileNotFoundError
        with self.assertRaises(FileNotFoundError) as err:
            geo_input_data_fix('Model 2',
                               extent_str='0, 1000, 0, 1000, 0, 1000',  # <-- should throw
                               resolution_str='20, 20, 20',
                               surface_points_file=data_dir + "model2_surface_points_df.csv",
                               orientations_file=data_dir + "missing",
                               mapping_file=data_dir + "model_2_mapping.json",
                               with_faults=False
                               )


if __name__ == '__main__':
    unittest.main()
