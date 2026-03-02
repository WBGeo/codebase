import io

from py_api_wbgeo.nodesapi import wbgeo_type, wbgeo_component, GeoTempFile, BasicallyABufferedFile

from core.object_components import MeshResults


@wbgeo_component(title='Download Mesh as VTZ',
                 description='Export Mesh to VTU',
                 group='Export',
                 identifier='wbgeo::expert_mesh_results_vtu',
                 )
def export_mesh_results_to_vtu(mesh: MeshResults) -> BasicallyABufferedFile:
  """
  Provides a mesh as a downloadable vtu file
  res: the Mesh to export
  """
  buf = GeoTempFile()  # create a new buffered file
  mesh.export_vtu(buf)
  # we can specify a name for the exported file
  buf.filename = 'export.vtu'
  # as long as we return a BytesIO (buffered file)
  return buf

@wbgeo_component(title='Download Mesh as GMSH',
                 description='Export Mesh to GMSH .msh format (structured or unstructured)',
                 group='Export',
                 identifier='wbgeo::export_mesh_results_gmsh',
                 )
def export_mesh_results_to_gmsh(mesh: MeshResults) -> BasicallyABufferedFile:
  """
  Provides a mesh as a downloadable GMSH .msh file.
  Supports both structured hexahedral and unstructured meshes.
  mesh: the Mesh to export
  """
  buf = GeoTempFile()
  mesh.export_gmsh(buf)
  buf.filename = 'export.msh'
  return buf


@wbgeo_component(title='Download Mesh as Exodus',
                 description='Export Mesh to Exodus',
                 group='Export',
                 identifier='wbgeo::expert_mesh_results_exodus',
                 )
def export_mesh_results_to_exodus(mesh: MeshResults) -> BasicallyABufferedFile:
  """
  Provides a mesh as a downloadable exodus (.exo) file
  mesh: the Mesh to export
  """
  # exodus/netCDF4.Dataset does not appear to accept an io.BytesIO, we thus use real files
  buf = GeoTempFile()  # create a new file-backed buffer
  mesh.export_exodus(buf)
  buf.filename = 'export.exo'  # we can specify a name for the exported file
  return buf
