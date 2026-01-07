import io

from py_api_wbgeo.nodesapi import wbgeo_type, wbgeo_component, AnnotatedScriptType, BasicallyABufferedFile

from core.object_components import GeomodelResults, MeshResults


@wbgeo_component(title='Export Mesh to VTU',
                 description='Export Mesh to VTU',
                 group='Export',
                 identifier='wbgeo::expert_mesh_results',
                 )
def export_mesh_results_to_vtu(res: MeshResults) -> BasicallyABufferedFile:
  buf = io.BytesIO() # create a new BytesIO (buffered file)
  res.export_vtu(buf)
  # we can specify a name for the exported file
  buf.filename = 'export.vtu'
  # as long as we return a BytesIO (buffered file)
  return buf


@wbgeo_component(description='Import Mesh from VTU',
                 title='Import Mesh from VTU',  # The title shown in the GUI
                 group='Import',
                 identifier='wbgeo::import_mesh_results'
                 )
def import_mesh_results_to_vtu(file: BasicallyABufferedFile) -> MeshResults:
  # see the examples for how to read the BytesIO
  # TODO: Decide if we want/need to import a mesh from a .vtu file?
  raise ValueError("not yet supported")
