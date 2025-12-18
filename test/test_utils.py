import unittest
from typing import List, Union

from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64

import numpy as np


class CommonTestCase(unittest.TestCase):
    """Common test methods"""

    def assert_list_list_array(self, actual: List[List[Union[NpNDArrayFp64, NpNDArrayInt64]]],
                               expected: List[List[Union[NpNDArrayFp64, NpNDArrayInt64]]], decimal=6):
        self.assertEqual(len(actual), len(expected))
        for res_ii, res_kk in zip(actual, expected):
            self.assert_list_array(res_ii, res_kk, decimal)

    def assert_list_array(self, actual: List[Union[NpNDArrayFp64, NpNDArrayInt64]],
                          expected: List[Union[NpNDArrayFp64, NpNDArrayInt64]], decimal=6):
        self.assertEqual(len(actual), len(expected))
        for res_ii, res_kk in zip(actual, expected):
            np.testing.assert_array_almost_equal(res_ii, res_kk, decimal)
