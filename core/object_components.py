import collections
import os
from dataclasses import field

import meshio
import pydantic
import pyvista
import typing
import numpy as np
from pydantic.dataclasses import dataclass
from typing import Optional, Tuple
from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64
import pandas as pd
from typing import Dict, List, Any
from typing import Optional, Union, List
from pydantic_numpy.typing import NpNDArrayInt64, NpNDArrayFp64
from py_api_wbgeo.nodesapi import wbgeo_type


from core.meshing_components.mesh_format.EXUDOS.Exo_format import ExosInputs
from core.meshing_components.mesh_format.VTU.VTU_format import VTUInputs
from core.meshing_components.mesh_format.VTM.VTM_format import VTMInputs
from core.meshing_components.mesh_format.STL.STL_format import STLInputs
from core.meshing_components.mesh_format.GMSH.GMSH_format import GMSHInputs
from core.meshing_components.mesh_format.ABAQUS.Abaqus_format import AbaqusInputs
from core.meshing_components.mesh_format.FEFLOW.Feflow_format import FeflowInputs, C_FeFlow
from core.meshing_components.mesh_format.ANSYS.Ansys_format import AnsysInputs

from core.meshing_components.geometry.Elements import Elements
from core.meshing_components.geometry.Nodes import Nodes
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, \
  PlainValidator, Field, ConfigDict, PrivateAttr

from core.structural_modeling_components.structural_objects.structural_objects import StructuralFrame, FaultFrame
from core.utility.pydantic_bridge import PandasDataFrame, MeshIOCellBlock

@wbgeo_type(name='Input input_data for the rock elements of a structural geological model',
            color='#b0dfa9',
            identifier='InputData_StructuralElements')
@dataclass
class InputData_StructuralElements:
    """
    A class to represent the input input_data for a geological model.

        Attributes:
            name (str): The name of the model.
            mapping_object (dict): Mapping of structural groups to structural elements.
            surface_points (pd.DataFrame): DataFrame containing surface points.
            orientations (Optional[pd.DataFrame]): DataFrame containing orientations.
    """
    name: str
    mapping_object: Dict[str, Tuple[str, ...]]
    surface_points: PandasDataFrame
    orientations: Optional[PandasDataFrame] = None

    @field_validator('mapping_object', mode='before')
    @classmethod
    def coerce_mapping_values_to_tuples(cls, v):
        if isinstance(v, dict):
            return {k: (val,) if isinstance(val, str) else tuple(val) for k, val in v.items()}
        return v

    def __post_init__(self):
        # reorder surface_points DataFrame by formation column for colormaps
        formation_order = [item for sublist in self.mapping_object.values() for item in
                           (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        formation_cat_type = pd.CategoricalDtype(categories=formation_order, ordered=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(formation_cat_type)
        self.surface_points = self.surface_points.sort_values(by='formation').reset_index(drop=True)
        self.surface_points['formation'] = self.surface_points['formation'].astype(str)

        # Remove duplicate surface points (same X, Y, Z, formation)
        _before = len(self.surface_points)
        self.surface_points = self.surface_points.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.surface_points)
        if _removed > 0:
            print(f"[InputData_StructuralElements '{self.name}'] "
                  f"Removed {_removed} duplicate surface point(s) (identical X, Y, Z, formation).")

        # Remove duplicate orientations (same X, Y, Z, formation)
        if self.orientations is not None and not self.orientations.empty:
            _before = len(self.orientations)
            self.orientations = self.orientations.drop_duplicates(
                subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
            _removed = _before - len(self.orientations)
            if _removed > 0:
                print(f"[InputData_StructuralElements '{self.name}'] "
                      f"Removed {_removed} duplicate orientation(s) (identical X, Y, Z, formation).")


@wbgeo_type(name='Input input_data for the fault elements of a structural geological model',
            color='orange',
            identifier='InputData_FaultElements')
@dataclass
class InputData_FaultElements:
    """
    A class to represent the input input_data for a geological model.

        Attributes:
            name (str): The name of the model.
            fault_surface_points (pd.DataFrame): DataFrame containing surface points.
            fault_orientations (pd.DataFrame): DataFrame containing orientations.
    """
    name: str
    fault_surface_points: PandasDataFrame
    fault_orientations: PandasDataFrame  # Might be optional in future when not only UCK is used here
    fault_names: List[str]  # This allows us to use one input data file

    def __post_init__(self):
        # Remove duplicate fault surface points (same X, Y, Z, formation)
        _before = len(self.fault_surface_points)
        self.fault_surface_points = self.fault_surface_points.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.fault_surface_points)
        if _removed > 0:
            print(f"[InputData_FaultElements '{self.name}'] "
                  f"Removed {_removed} duplicate fault surface point(s) (identical X, Y, Z, formation).")

        # Remove duplicate fault orientations (same X, Y, Z, formation)
        _before = len(self.fault_orientations)
        self.fault_orientations = self.fault_orientations.drop_duplicates(
            subset=['X', 'Y', 'Z', 'formation']).reset_index(drop=True)
        _removed = _before - len(self.fault_orientations)
        if _removed > 0:
            print(f"[InputData_FaultElements '{self.name}'] "
                  f"Removed {_removed} duplicate fault orientation(s) (identical X, Y, Z, formation).")


@wbgeo_type(name='Result of a structural geological model', color='#8cb369', identifier='StructuralModelResults')
@dataclass
class StructuralModelResults:
    """
    A class to represent the results of a geological model.

        Attributes:.
            structural_frame (StructuralFrame): The structural frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    structural_frame: StructuralFrame  # this is a deepcopy of the structural frame object


@wbgeo_type(name='Result of a structural fault model', color='blue', identifier='FaultModelResults')
@dataclass
class FaultModelResults:
    """
    A class to represent the results of a fault model.

        Attributes:.
            fault_frame (FaultFrame): The fault frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    fault_frame: FaultFrame  # this is a deepcopy of the structural frame object

@wbgeo_type(name='Meshing results', color='green', identifier='MeshResults')
class MeshResults(BaseModel):
    """
    Container class holding unstructured mesh results.

    Attributes
    ----------
    nodes : np.ndarray
        Array of node coordinates with shape (N, 3)

    elements : list[meshio.CellBlock]
        Mesh elements stored as MeshIO CellBlocks.

    cell_data : dict[str, list[np.ndarray]], optional
        Per-cell data arrays associated with the mesh.
    """


    nodes: NpNDArrayFp64
    elements: List[MeshIOCellBlock]
    cell_data: Optional[Dict[str, List[NpNDArrayFp64]]] = None

    # transient / derived
    _mesh : Optional[pyvista.MultiBlock]  = PrivateAttr(default=None) #  exclude this field from serialization
    @property
    def mesh(self): # getter/setter due to private/transient field
      return self._mesh

    @mesh.setter
    def mesh(self, m): # getter/setter due to private/transient field
      self._mesh = m
      
    # -----------------------------
    # Post init
    # -----------------------------
    def __post_init__(self):
        self.vtm_in = VTMInputs(
            self.nodes,
            self.elements
        )

        self.mesh = self.vtm_in.create_mesh()

@wbgeo_type(name='Exporters', color='grey', identifier='Exporters')
class Exporters(MeshResults):
    """
    Export utility class for MeshResults.

    This class extends `MeshResults` and provides methods to export
    the mesh into various standard geoscientific and engineering formats.

    Supported mesh types depend on the export format:
    - Structured meshes: VTU, VTK, VTM, Exodus, Ansys
    - Unstructured meshes: STL, Abaqus, Gmsh, FeFlow, VTU, VTK, VTM, Exodus, Ansys

    """

    def export_vtu(self, filename: str):
        """
        Export the mesh to a VTU (VTK Unstructured Grid) file.

        Args:
        filename: Output filename ending with `.vtu`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        vtu_in = VTUInputs(self.nodes, self.elements )
        # Create the VTU mesh
        mesh = vtu_in.create_mesh()
        # Write the mesh to a VTU file
        mesh.write(filename, file_format="vtu")
        print(f"VTU file '{filename}' created successfully!")


    def export_stl(self, filename: str):
        """
        Export the mesh surface to STL format.

        Args:
        filename: Output STL filename.

        Supported Mesh Types:
        - Unstructured meshes ONLY
        """
        stl_in = STLInputs(self.nodes, self.elements)
        stl_in.output_filename = filename
        stl_in.create_mesh()
        print(f"Stl files '{filename}' created successfully!")


    def export_exodus(self, filename: str):
        """
        Export the mesh to an Exodus (.exo) file.

        Args:
        filename: Output filename ending with `.exo`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        exo_in = ExosInputs(self.nodes, self.elements)
        # Create mesh
        mesh = exo_in.create_mesh()
        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="exodus")
        print(f"Exodus file '{filename}' created successfully!")


    def export_abaqus(self, filename: str):
        """
        Export the mesh to an Abaqus input (.inp) file.

        Args:
        filename:Output Abaqus input filename.

        Supported Mesh Types:
        --------------------
        - Unstructured meshes ONLY
        """
        abq = AbaqusInputs(self.nodes,self.elements)
        abq.write(filename)
        print(f"Abaqus file '{filename}' created successfully!")



    def export_ansys(self, filename: str):
        """
        Export the mesh to an Ansys-compatible format.

        Args:
        filename:Output filename.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        Ansys_in = AnsysInputs(self.nodes, self.elements)
        # Create mesh
        mesh = Ansys_in.create_mesh()
        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="ansys")
        print(f"Ansys file '{filename}' created successfully!")



    def export_gmsh(self, filename: str):
        """
        Export the mesh to Gmsh (.msh) format.

        Args:
        filename: Output filename ending with `.msh`.

        Supported Mesh Types:
        --------------------
        - Unstructured meshes ONLY
        """
        elements =self.elements
        if not isinstance(elements, list):
            raise TypeError("Gmsh export requires unstructured CellBlocks. You can try " \
            "exporting to structured formats like VTU, Exodus, VTM, VTK, ...")
        gmsh_in = GMSHInputs(self.nodes, elements,)
        mesh = gmsh_in.create_mesh()
        meshio.write(filename, mesh, file_format="gmsh")
        print(f"GMSH file '{filename}' created successfully!")



    def export_vtk(self, filename: str):
        """
        Export the mesh to a legacy VTK file.

        Args:
        ----------
        filename: Output filename ending with `.vtk`.

        Supported Mesh Types:
        --------------------
        - Structured meshes
        - Unstructured meshes
        """
        self.mesh.save(filename)
        print(f"VTK file '{filename}' with created successfully!")



    def export_vtm(self, filename: str):
        """
        Export the mesh to a VTM (VTK MultiBlock) file.

        Args:
        filename: Output filename ending with `.vtm`.

        Supported Mesh Types:
        - Structured meshes
        - Unstructured meshes
        """
        self.mesh.save(filename)
        print(f"VTM files '{filename}' created successfully!")



    # TODO: export_resqml is not yet fully working (Petrel/CMG compatibility issues
    #       with ControlPointParameters namespace and K-direction convention).
    #       Uncomment and continue when ready to finalize RESQML export.
    #
    # def export_resqml(self, filename: str, title: str = "GeoModel",
    #                   use_parametric_lines: bool = True) -> None:
    #     """Export structured mesh to RESQML .epc (explicit IjkGrid + lithology property).
    #
    #     Writes two files: <filename> (.epc) and a companion .h5 HDF5 file.
    #     Only works for structured (hexahedral) meshes.
    #
    #     Args:
    #         filename:              Path to the output .epc file (companion .h5 is written alongside).
    #         title:                 Title string stored in the RESQML model.
    #         use_parametric_lines:  If False (default), writes fully explicit Point3dHdf5Array
    #                                geometry — broadest compatibility with Petrel, CMG, etc. If True,
    #                                writes pillar-based parametric lines geometry; note resqpy's
    #                                ControlPointParameters XML has a namespace issue that causes
    #                                validation failures in Petrel and CMG.
    #
    #     TODOs:
    #         1. IJK dimension robustness — store n_gx/n_gy in MeshResults from
    #            create_structured_mesh_data rather than inferring from unique node coords.
    #         2. Faulted models — use structural_frame.lith_block sampled at cell centres
    #            instead of surface_id from the element table.
    #         3. Georeferenced CRS — populate epsg_code / xy_units from model extent for
    #            real-world data (currently uses a local metre CRS).
    #         4. Formation name lookup — add a RESQML StringLookup table so CMG/Petrel
    #            shows formation names instead of integer IDs.
    #         5. Pillar geometry — some CMG versions prefer pillar-based geometry; explicit
    #            corner-point is simpler and universally supported.
    #         6. resqpy API version — tested against the version pinned in requirements.txt;
    #            resqpy has had breaking API changes in the past, verify if upgrading.
    #     """
    #     import resqpy.model as rq
    #     import resqpy.grid as grr
    #     import resqpy.crs as rqc
    #     import resqpy.property as rqp
    #
    #     assert self.elements_structured is not None, (
    #         "export_resqml requires a structured mesh (elements_structured must not be None)"
    #     )
    #
    #     elems = self.elements_structured            # (N, 10): [elem_id, n0..n7, surface_id]
    #     xyz   = self.nodes[:, 1:]                  # (M, 3):  strip node_id column
    #
    #     # --- IJK dimensions ---
    #     # TODO (1): replace with stored n_gx/n_gy for robustness on non-regular grids
    #     ni = int(np.unique(xyz[:, 0].round(6)).size) - 1   # cells in X/I
    #     nj = int(np.unique(xyz[:, 1].round(6)).size) - 1   # cells in Y/J
    #     nk = len(elems) // (ni * nj)                        # cells in Z/K (layers)
    #     print(f"[export_resqml] Grid dimensions: ni={ni}, nj={nj}, nk={nk} "
    #           f"(total cells={ni*nj*nk})")
    #
    #     # --- Vectorised element index → (k, j, i) mapping ---
    #     n   = len(elems)
    #     ks  = np.arange(n) // (nj * ni)
    #     js  = (np.arange(n) % (nj * ni)) // ni
    #     is_ = np.arange(n) % ni
    #
    #     # --- Corner-point node array (nk+1, nj+1, ni+1, 3) ---
    #     corners = xyz[elems[:, 1:9].astype(int)]  # (N, 8, 3)
    #     points  = np.empty((nk + 1, nj + 1, ni + 1, 3), dtype=float)
    #     # VTK hexahedron node ordering (cols 1–4 = bottom face, 5–8 = top face):
    #     #   n0=(k,j,i)   n1=(k,j,i+1)   n2=(k,j+1,i+1)   n3=(k,j+1,i)
    #     #   n4=(k+1,j,i) n5=(k+1,j,i+1) n6=(k+1,j+1,i+1) n7=(k+1,j+1,i)
    #     points[ks,     js,     is_    ] = corners[:, 0]
    #     points[ks,     js,     is_ + 1] = corners[:, 1]
    #     points[ks,     js + 1, is_ + 1] = corners[:, 2]
    #     points[ks,     js + 1, is_    ] = corners[:, 3]
    #     points[ks + 1, js,     is_    ] = corners[:, 4]
    #     points[ks + 1, js,     is_ + 1] = corners[:, 5]
    #     points[ks + 1, js + 1, is_ + 1] = corners[:, 6]
    #     points[ks + 1, js + 1, is_    ] = corners[:, 7]
    #
    #     # --- Lithology per cell (KJI order) ---
    #     # TODO (2): for faulted models use structural_frame.lith_block at cell centres
    #     lith_kji = elems[:, -1].reshape(nk, nj, ni).astype(np.int32)
    #
    #     # --- Flip K to down convention (K=0 = top/shallowest, K increases downward) ---
    #     # Our mesh has K=0 at the base; reservoir simulators (CMG, Eclipse) expect K-down.
    #     # Flipping both arrays keeps geometry and lithology consistent.
    #     points  = points[::-1, :, :, :]
    #     lith_kji = lith_kji[::-1, :, :]
    #
    #     # --- Build RESQML model ---
    #     model = rq.Model(filename, new_epc=True)
    #
    #     # TODO (3): pass epsg_code and georeferenced xy_units for real-world data
    #     crs = rqc.Crs(model, z_inc_down=False, xy_units='m', z_units='m',
    #                   title='local_CRS')
    #     crs.create_xml()
    #
    #     # resqpy 5.x: Grid.__init__ no longer accepts extent_kji/crs_uuid —
    #     # create an empty grid then set all geometry attributes manually.
    #     grid = grr.Grid(model, title=title)
    #     grid.extent_kji = (nk, nj, ni)
    #     grid.nk, grid.nj, grid.ni = nk, nj, ni
    #     grid.crs_uuid = crs.uuid
    #     grid.k_direction_is_down           = True    # K=0 at top, increases downward (simulator convention)
    #     grid.grid_is_right_handed          = True
    #     grid.pillar_shape                  = 'straight'
    #     grid.has_split_coordinate_lines    = False
    #     grid.k_gaps                        = None
    #     grid.points_cached                 = points
    #     grid.geometry_defined_for_all_pillars_cached = True
    #     grid.geometry_defined_for_all_cells_cached   = True
    #     # TODO (5): for grids with non-straight pillars (e.g. thrust faults), use_parametric_lines=False
    #     grid.write_hdf5(use_parametric_lines=use_parametric_lines)
    #     grid.create_xml(write_active=False, use_parametric_lines=use_parametric_lines)
    #
    #     pc = rqp.PropertyCollection(support=grid)
    #     # TODO (4): add a StringLookup table keyed on lith IDs so CMG/Petrel shows
    #     #           formation names rather than raw integer IDs.
    #     pc.add_cached_array_to_imported_list(
    #         lith_kji,
    #         source_info='WBGeo',
    #         keyword='ROCK_TYPE',
    #         discrete=True,
    #         uom='Euc',
    #         property_kind='discrete rock volume',
    #         indexable_element='cells',
    #     )
    #     pc.write_hdf5_for_imported_list()
    #     pc.create_xml_for_imported_list_and_add_parts_to_model()
    #
    #     model.store_epc()
    #
    #     # Post-process: remove the optional ControlPointParameters element from the
    #     # IjkGrid XML when using parametric lines.  resqpy writes it with a namespace
    #     # that triggers a validation error in Petrel ("tag name or namespace mismatch")
    #     # and a fatal import crash when combined with explicit geometry.
    #     # ControlPointParameters is optional per the RESQML 2.0.1 spec; removing it
    #     # leaves a valid file that Petrel and CMG can read without errors.
    #     if use_parametric_lines:
    #         import re, zipfile as _zf
    #         with _zf.ZipFile(filename, 'r') as zin:
    #             entries = {name: zin.read(name) for name in zin.namelist()}
    #         grid_part = next(
    #             (n for n in entries if 'IjkGrid' in n and '_rels' not in n), None)
    #         if grid_part:
    #             xml = entries[grid_part].decode('utf-8')
    #             xml = re.sub(
    #                 r'\s*<resqml2:ControlPointParameters\b[^>]*>.*?</resqml2:ControlPointParameters>',
    #                 '', xml, flags=re.DOTALL)
    #             entries[grid_part] = xml.encode('utf-8')
    #             with _zf.ZipFile(filename, 'w', _zf.ZIP_DEFLATED) as zout:
    #                 for name, data in entries.items():
    #                     zout.writestr(name, data)
    #
    #     print(f"[export_resqml] RESQML model saved to {filename}")

    def export_feflow(self, filename: str):
        """
        Export the mesh to a FeFlow (.fem) file.

        Args:
        filename: Output filename ending with `.fem`.

        Supported Mesh Types:
        - Unstructured meshes ONLY
        """
        elements = self.elements
        # Reject structured meshes
        if not isinstance(elements, list):
            raise TypeError("FeFlow export supports ONLY unstructured meshes.\n"
            "You can export a structured mesh using other formats (e.g., VTU, VTK, VTM, Exodus,..).")

        feflow = FeflowInputs(self.nodes, elements)
        feflow.write(filename)
        print(f"Feflow file '{filename}' created successfully!")


@wbgeo_type(name='SimulationResults', color='pink', identifier='SimulationResults')
@dataclass
class SimulationResults:
    """
    Container class for all simulation results timesteps.

    Attributes
    ----------
    nodes_by_time : Dict[float, np.ndarray]
        Node coordinates for each timestep.

    cells_by_time : Dict[float, np.ndarray]
        Cell connectivity for each timestep.

    celltypes_by_time : Dict[float, np.ndarray]
        Cell types for each timestep.

    node_data_by_time : Dict[float, Dict[str, np.ndarray]]
        Node-based data arrays for each timestep.

    cell_data_by_time : Dict[float, Dict[str, np.ndarray]]
        Cell-based data arrays for each timestep.
    """
    nodes_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    cells_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    celltypes_by_time: Dict[float, np.ndarray] = field(default_factory=dict)
    node_data_by_time: Dict[float, Dict[str, np.ndarray]] = field(default_factory=dict)
    cell_data_by_time: Dict[float, Dict[str, np.ndarray]] = field(default_factory=dict)
