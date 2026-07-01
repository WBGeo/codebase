import unittest
import numpy as np
import pandas as pd
from core.meshing_components.explicit.structured.mesh_data import (
    sort_points_by_x_y,
    sort_surfaces_by_z,
    store_points_in_array
)

class TestSortingAndStorage(unittest.TestCase):

    def make_surfaces(self):
        # Surface 1 (mean Z = 5)
        df1 = pd.DataFrame({
            "X": [1, 0, 1, 0],
            "Y": [0, 0, 1, 1],
            "Z": [5, 5, 5, 5]
        })

        # Surface 2 (mean Z = 10)
        df2 = pd.DataFrame({
            "X": [1, 0, 1, 0],
            "Y": [0, 0, 1, 1],
            "Z": [10, 10, 10, 10]
        })

        return [df1, df2]

    # sort_points_by_x_y
    def test_sort_points_by_x_y(self):
        df = pd.DataFrame({
            "X": [2, 1, 2, 1],
            "Y": [1, 1, 0, 0],
            "Z": [10, 20, 30, 40]
        })

        result = sort_points_by_x_y([df])[0]

        # correct lexicographic expectation (X then Y)
        expected_order = [(1, 0), (1, 1), (2, 0), (2, 1)]
        self.assertEqual(list(zip(result["X"], result["Y"])), expected_order)

    # sort_surfaces_by_z
    def test_sort_surfaces_by_z(self):
        dfs = self.make_surfaces()

        sorted_dfs = sort_surfaces_by_z(dfs)

        self.assertAlmostEqual(sorted_dfs[0]["Z"].mean(), 5)
        self.assertAlmostEqual(sorted_dfs[1]["Z"].mean(), 10)

    # store_points_in_array
    def test_store_points_shape_and_values(self):
        dfs = self.make_surfaces()

        output, n_gx, n_gy, n_gz = store_points_in_array(dfs)

        self.assertEqual(n_gx, 2)
        self.assertEqual(n_gy, 2)
        self.assertEqual(n_gz, 1)

        self.assertEqual(output.shape, (2, 12))

    def test_store_points_structure(self):
        dfs = self.make_surfaces()

        output, n_gx, n_gy, _ = store_points_in_array(dfs)

        n = n_gx * n_gy

        x = output[0, :n]
        y = output[0, n:2*n]
        z = output[0, 2*n:3*n]

        self.assertCountEqual(x, dfs[0]["X"].values)
        self.assertCountEqual(y, dfs[0]["Y"].values)
        self.assertCountEqual(z, dfs[0]["Z"].values)

    def test_inconsistent_grid_raises(self):
        dfs = [
            pd.DataFrame({"X": [0, 1], "Y": [0, 1], "Z": [0, 0]}),
            pd.DataFrame({"X": [0, 1, 2], "Y": [0, 1, 2], "Z": [1, 1, 1]})
        ]

        with self.assertRaises(ValueError):
            store_points_in_array(dfs)

#######################################
if __name__ == "__main__":
    unittest.main()
