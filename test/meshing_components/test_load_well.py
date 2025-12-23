import re
import unittest
import os

from core.meshing_components.explicit.unstructured.mesh_data import load_wells_from_csv

data_dir = os.path.dirname(__file__) + "/../../examples/data/"


class LoadWellsTestCase(unittest.TestCase):
  """This test tests the well loading """

  def test_correct(self):
    ret = load_wells_from_csv(data_dir + 'model_10_wells.csv')
    self.assertEqual(2, len(ret), "Length did not match")
    self.assertEqual([100.0, 100.0, 100.0, 100.0, 100.0, 500.0], ret[0])
    self.assertEqual([500.0, 500.0, 500.0, 500.0, 500.0, 900.0], ret[1])

  def test_not_a_vertice(self):
    # one well is missing a second point -> no vertice could be created
    with self.assertRaisesRegex(ValueError,
                                re.escape("Some well(s) ['1'] are missing their second point")):
      load_wells_from_csv(os.path.dirname(__file__) + "/data/not_a_vertice_wells.csv")


if __name__ == '__main__':
  unittest.main()
