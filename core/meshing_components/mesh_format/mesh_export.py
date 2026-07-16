import enum
import typing

from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile, AnnotatedScriptType

from core.meshing_components.mesh_format.abaqus.Abaqus_format import export_mesh_results_to_abaqus
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus, exodus_type_for
from core.meshing_components.mesh_format.feflow.Feflow_format import export_mesh_results_to_feflow
from core.meshing_components.mesh_format.gmsh.GMSH_format import export_mesh_results_to_gmsh
from core.meshing_components.mesh_format.stl.STL_format import export_mesh_results_to_stl
from core.meshing_components.mesh_format.vtk.VTK_format import export_mesh_results_to_vtk
from core.meshing_components.mesh_format.vtm.VTM_format import export_mesh_results_to_vtm
from core.meshing_components.mesh_format.vtu.VTU_format import export_mesh_results_to_vtu
from core.object_components import MeshResults

"""
Expose all export functions via one singular component
"""

class MeshFormatType(enum.Enum):
  VTU = 1
  Abaqus = 2
  ANSYS = 3
  Exodus = 4
  Feflow = 5
  Gmsh = 6
  STL = 7
  VTK = 8
  VTM_ZIP = 9


MeshFormatType_A = typing.Annotated[
  MeshFormatType, AnnotatedScriptType(name='MeshFormatType', color='aqua',
                                      identifier='MeshFormatType',
                                      controlled="Select|" + "|".join(
                                        [e.name for e in MeshFormatType]))]


@wbgeo_component(
  title="Download Mesh",
  description="Export Mesh to a selection of formats",
  group="Export",
  identifier="wbgeo::export_mesh_results",
)
def export_mesh_results(mesh: MeshResults, format: MeshFormatType_A = MeshFormatType.Exodus) -> BasicallyABufferedFile:
  if format == MeshFormatType.VTU:
    return export_mesh_results_to_vtu(mesh)
  elif format == MeshFormatType.Abaqus:
    return export_mesh_results_to_abaqus(mesh)
  elif format == MeshFormatType.ANSYS:
    return export_mesh_results_to_ansys(mesh)
  elif format == MeshFormatType.Exodus:
    # Passes the mesh's own recorded type explicitly -- without this,
    # export_mesh_results_to_exodus's default (MeshType.UNSTRUCTURED)
    # applies regardless of the mesh's true type, which crashes outright
    # for implicit/structured meshes (they only ever have 1 element block,
    # fewer than the unstructured path's minimum) and silently produces
    # weaker side-sets even after the dim-based fix.
    return export_mesh_results_to_exodus(mesh, type=exodus_type_for(mesh))
  elif format == MeshFormatType.Feflow:
    return export_mesh_results_to_feflow(mesh)
  elif format == MeshFormatType.Gmsh:
    return export_mesh_results_to_gmsh(mesh)
  elif format == MeshFormatType.STL:
    return export_mesh_results_to_stl(mesh)
  elif format == MeshFormatType.VTK:
    return export_mesh_results_to_vtk(mesh)
  elif format == MeshFormatType.VTM_ZIP:
    return export_mesh_results_to_vtm(mesh)
  else:
    raise ValueError(f"Unhandled format {format}")
