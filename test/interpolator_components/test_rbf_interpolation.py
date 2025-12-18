import pickle
import unittest
import os

import numpy as np
import pandas as pd

from core.interpolator_components.rbf_interpolation import rbf_interpolator
from core.object_components import InputData, GeomodelResults
from test.interpolator_components.interpolation_test_utils import InterpolationTestCase

data_dir = os.path.dirname(__file__) + "/../../examples/data/"




class RBFInterpTestCase(InterpolationTestCase):
    """This test"""

    @classmethod
    def setUpClass(cls):
        # every test method of this class uses the data_test InputData
        cls.input_data = InputData(name='Model 2',
                                   extent=np.array([0, 1000, 0, 1000, 0, 1000]),
                                   resolution=np.array([100, 100, 100]),
                                   mapping_object={"Strat_Series": ('rock2', 'rock1')},
                                   surface_points=pd.read_csv(
                                       data_dir + "model2_surface_points_df.csv"),
                                   orientations=pd.read_csv(
                                       data_dir + "model2_orientations_df.csv"),
                                   )

    def test_linear_regression(self):
        res = rbf_interpolator(self.input_data, kernel='linear')
        self.assert_geomodel_result(res, 'model2_linear.pkl', do_update_stored_objects=False)

    def test_cubic_regression(self):
        res = rbf_interpolator(self.input_data, kernel='cubic')
        self.assert_geomodel_result(res, 'model2_cubic.pkl', do_update_stored_objects=False)


    ######################################
    ### Test negative/incorrect inputs ###
    ######################################

    def test_illegal_kernel(self):
        # when we pass an incorrect kernel
        with self.assertRaisesRegex(ValueError, '`kernel` must be one of'):
            rbf_interpolator(input_data=self.input_data, kernel='error')

    def test_with_faults(self):
        # the interpolator does not support faults
        # clone the input_data and add some faults:
        with_faults = pickle.loads(pickle.dumps(self.input_data))
        with_faults.faults = [True]
        with self.assertRaisesRegex(ValueError, 'Interpolator can not handle faults in the current state'):
            rbf_interpolator(input_data=with_faults)

    def test_no_smoothing_and_non_distinct_data_points(self):
        # from the RBF library:
        # > If `smoothing` is 0, then each data point location must be distinct.
        # todo: not yet implemented
        pass


if __name__ == '__main__':
    unittest.main()
