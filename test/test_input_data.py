import unittest

from core.object_components import InputData
import numpy as np
import pandas as pd
import pickle
import os

cwd = os.getcwd() + "/.."
data_dir = os.getcwd() + "/../examples/data/"


# based on model 2
class TestInputData(unittest.TestCase):

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

    def test_surface_points_formation(self):
        # the formation data is post-initialized
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
        with open('model2_input_data.pkl', 'rb') as f:
            regression_data: InputData = pickle.load(f)
        self.assertEqual(self.data_test.name, regression_data.name)
        np.testing.assert_array_equal(self.data_test.extent, regression_data.extent)
        np.testing.assert_array_equal(self.data_test.resolution, regression_data.resolution)
        self.assertEqual(self.data_test.mapping_object, regression_data.mapping_object)
        np.testing.assert_array_equal(self.data_test.surface_points, regression_data.surface_points)
        np.testing.assert_array_equal(self.data_test.orientations, regression_data.orientations)
        self.assertEqual(self.data_test.faults, regression_data.faults)
        np.testing.assert_array_equal(self.data_test.fault_relations, regression_data.fault_relations)

    @classmethod
    def update_data_files(cls, data: InputData):
        with open('model2_input_data.pkl', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    unittest.main()
