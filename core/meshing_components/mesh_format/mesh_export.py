import enum
import typing

from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile, AnnotatedScriptType

from core.meshing_components.mesh_format.abaqus.Abaqus_format import export_mesh_results_to_abaqus
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
from core.meshing_components.mesh_format.exodus.Exo_format import (
    MeshType as ExodusMeshType,
    export_mesh_results_to_exodus,
)
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
  title="Export Mesh",
  description="Export Mesh to a selection of formats",
  group="Meshing",
  positionInGroup=-1,
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
    # MeshResults records its origin as "implicit", "structured", or
    # "unstructured"; the Exodus exporter uses "imp", "str", or "unstr".
    # Preserve the old unstructured default for legacy meshes without type
    # metadata, while rejecting unknown non-null types explicitly.
    mesh_type = getattr(mesh, "mesh_type", None)
    exodus_types = {
      "implicit": ExodusMeshType.IMPLICIT,
      "structured": ExodusMeshType.STRUCTURED,
      "unstructured": ExodusMeshType.UNSTRUCTURED,
      "imp": ExodusMeshType.IMPLICIT,
      "str": ExodusMeshType.STRUCTURED,
      "unstr": ExodusMeshType.UNSTRUCTURED,
    }
    if mesh_type is None:
      exodus_type = ExodusMeshType.UNSTRUCTURED
    else:
      exodus_type = exodus_types.get(str(mesh_type).lower())
      if exodus_type is None:
        raise ValueError(f"Unsupported mesh type for Exodus export: {mesh_type!r}")
    return export_mesh_results_to_exodus(mesh, type=exodus_type)
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
