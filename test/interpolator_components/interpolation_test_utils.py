import os
import pickle

import numpy as np

from core.object_components import GeomodelResults
from test.test_utils import CommonTestCase


class InterpolationTestCase(CommonTestCase):
    """Common methods for all interpolation tests"""

    def assert_geomodel_result(self, res: GeomodelResults, filename: str, do_update_stored_objects: bool = False):
        if do_update_stored_objects:
            with open(os.path.dirname(__file__) + '/' + filename, 'wb') as f:
                pickle.dump(res, f)
        with open(os.path.dirname(__file__) + '/' + filename, 'rb') as f:
            regression_data: GeomodelResults = pickle.load(f)
        self.assertEqual(res.name, regression_data.name)
        np.testing.assert_array_almost_equal(res.lith_block, regression_data.lith_block)
        np.testing.assert_array_almost_equal(res.grid, regression_data.grid)
        np.testing.assert_array_almost_equal(res.extent, regression_data.extent)
        np.testing.assert_array_almost_equal(res.resolution, regression_data.resolution)
        self.assertEqual(res.faults, regression_data.faults)
        self.assert_list_list_array(res.surface_meshes_vertices, regression_data.surface_meshes_vertices)
        self.assert_list_list_array(res.surface_meshes_edges, regression_data.surface_meshes_edges)
        self.assertEqual(res.mapping_object, regression_data.mapping_object)
        if res.scalar_fields is None:
            self.assertEqual(res.scalar_fields, regression_data.scalar_fields)
        else:
            self.assert_list_array(res.scalar_fields, regression_data.scalar_fields)
        return regression_data
