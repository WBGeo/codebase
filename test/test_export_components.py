import unittest
import os

import py_api_wbgeo.nodesapi

from core.export_components import export_mesh_results_to_vtu, export_mesh_results_to_exodus
from core.object_components import MeshResults

data_dir = os.path.dirname(__file__) + "/meshing_components/data/"


class ExportMeshTestCase(unittest.TestCase):
  """This test checks that a mesh can be exported to file

  To update the json.gz file:
  from pydantic_core import to_jsonable_python
  import json
  import zlib
  with open('outmesh.json.gz', 'wb') as f:
    f.write(zlib.compress(json.dumps(to_jsonable_python(mesh_test)).encode('utf-8')))

  """

  def setUp(self):
    import json
    import zlib
    with open(data_dir + "model_2_structured_mesh.json.gz", 'rb') as f:
      self.mesh = MeshResults(**json.loads(zlib.decompress(f.read())))

  def test_export_vtk(self):
    out = export_mesh_results_to_vtu(self.mesh)
    try:
      self.assertIsInstance(out, py_api_wbgeo.nodesapi.GeoTempFile)
      self.assertRegex(out.__getattribute__('filename'), ".*export.vtu$")
    finally:
      out.close_and_remove()

  def test_export_exodus(self):
    out = export_mesh_results_to_exodus(self.mesh)
    try:
      self.assertIsInstance(out, py_api_wbgeo.nodesapi.GeoTempFile)
      self.assertRegex(out.__getattribute__('filename'), ".*export.exo$")
    finally:
      out.close_and_remove()


if __name__ == '__main__':
  unittest.main()
