import collections
import os

import pydantic
import pyvista
import typing
import numpy as np
from pydantic.dataclasses import dataclass
from typing import Optional, Tuple
from pydantic_numpy import NpNDArrayFp64, NpNDArrayInt64
import pandas as pd
from typing import TypeVar, Dict, List
from typing import Dict, List, Any
import pyvista as pv
import meshio
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
from pydantic import BaseModel, field_serializer, field_validator, BeforeValidator, PlainSerializer, PlainValidator, Field, ConfigDict

from core.structural_modeling_components.structural_objects.structural_objects import StructuralFrame, FaultFrame

# Pydantic adapter for panda DataFrame
def df_serializer(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def df_validator(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    elif isinstance(value, list):
        return pd.DataFrame(value)
    raise TypeError("Expected a pandas DataFrame or a list of dictionaries.")


PandasDataFrame = typing.Annotated[
    pd.DataFrame, PlainSerializer(df_serializer), BeforeValidator(df_validator)]

@wbgeo_type(name='Input input_data for the rock elements of a structural geological model',
            color='orange',
            identifier='InputData_StructuralElements')
@dataclass(config={"arbitrary_types_allowed": True})
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
@dataclass(config={"arbitrary_types_allowed": True})
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


@wbgeo_type(name='Result of a structural geological model', color='blue', identifier='StructuralModelResults')
@dataclass(config={"arbitrary_types_allowed": True})
class StructuralModelResults:
    """
    A class to represent the results of a geological model.

        Attributes:.
            structural_frame (StructuralFrame): The structural frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    structural_frame: StructuralFrame  # this is a deepcopy of the structural frame object


@wbgeo_type(name='Result of a structural fault model', color='blue', identifier='FaultModelResults')
@dataclass(config={"arbitrary_types_allowed": True})
class FaultModelResults:
    """
    A class to represent the results of a fault model.

        Attributes:.
            fault_frame (FaultFrame): The fault frame of the model.
    """
    # TODO: ALEX: This is the simplest version I could think of - does this work for you
    fault_frame: FaultFrame  # this is a deepcopy of the structural frame object

# todo: Move into common class?
def cellblock_encoder(obj: meshio.CellBlock):
  import pickle
  import codecs
  return codecs.encode(pickle.dumps(obj), "base64").decode()

@wbgeo_type(name='Meshing results', color='green', identifier='MeshResults')
@dataclass(config={"arbitrary_types_allowed": True, "json_encoders" : {meshio.CellBlock: cellblock_encoder}})
class MeshResults:
    nodes: NpNDArrayFp64 # TODO: int or FP array?
    # mesh is a transient/derived field
    mesh : Optional[pyvista.MultiBlock]  = Field(default=None, exclude = True) #  exclude this field from serialization

    elements_structured: typing.Optional[NpNDArrayFp64] = None
    elements_unstructured: typing.Optional[List[meshio.CellBlock]] = None # todo: NpNDArrayFp64 for structured, CellBlock for unstructured - union not possible!


    # todo: Move into common class?
    @pydantic.field_validator('elements_unstructured', mode="before")
    @classmethod
    def decode_cellblock(cls, v):
      if v is None:
        return None
      if isinstance(v, typing.List) or isinstance(v, list) or isinstance(v, collections.abc.Iterable) or True:
        import pickle
        import codecs
        return [
            e if isinstance(e, meshio.CellBlock) or e is None else pickle.loads(codecs.decode(e.encode(), "base64")) for e in v
          ]
      raise ValueError("Unhandled cellblock", type(v))

    def get_union_elems(self):
      if self.elements_structured is not None:
        return self.elements_structured
      elif self.elements_unstructured is not None:
        return self.elements_unstructured
      else:
        raise Exception("Elements not initialized")

    def __post_init__(self):
        self.vtm_in = VTMInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        print('[INFO] VTMInputs initialized successfully.')

        self.mesh = self.vtm_in.create_mesh()

        # initialize node/element objects only for ndarray elements
        if isinstance(self.get_union_elems(), np.ndarray):
            self.nodes_obj = Nodes(node_array=self.nodes)
            self.elements_obj = Elements(element_array=self.get_union_elems(), node_array=self.nodes)

    def export_vtu(self, filename: typing.Union[str|os.PathLike]):
        """
        Export the mesh input_data to a VTU file.
        Args:
            filename (str): The name of the VTU file to export.
        """
        vtu_in = VTUInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())

        # Create the VTU mesh
        mesh = vtu_in.create_mesh()

        # Write the mesh to a VTU file
        mesh.write(filename, file_format="vtu")
        print(f"VTU file '{filename}' created successfully!")

    def export_exodus(self, filename: typing.Union[str, os.PathLike]):
        """
        Export the mesh input_data to an Exodus file.
        Args:
            filename (str): The name of the Exodus file to export.
        """
        exo_in = ExosInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        # Create mesh
        mesh = exo_in.create_mesh()

        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="exodus")
        print(f"Exodus file '{filename}' created successfully!")



    def export_abaqus(self, filename: str):
        """
        Export mesh data to an Abaqus .inp file with nodes, tetrahedral (C3D4),
        and triangular (CP3S) elements, including a valid material and section definition.
        """
        abq_in = AbaqusInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        mesh = abq_in.create_mesh()
        node_array = mesh.points
        elements = mesh.cells

        element_type_map = {
            "line": "T3D2",
            "tetra": "C3D4",
            "triangle": "S3R",
            "hexahedron": "C3D8"
        }

        with open(filename, 'w') as f:
            # Write header
            f.write("*" * 37 + "\n")
            f.write("*HEADING\n")
            f.write("ICEM - ABAQUS INTERFACES VERSION 4.3.1\n")
            f.write("*" * 37 + "\n")

            # Write nodes
            f.write("*NODE, NSET=All\n")
            for i, coord in enumerate(node_array, start=1):
                x, y, z = coord
                f.write(f"{i}, {x:.8E}, {y:.8E}, {z:.8E}\n")

            # Track ELSETs
            solid_elsets = []
            tus_elsets = []
            shel_elsets = []

            # Write elements
            element_id = 1
            for i, block in enumerate(elements):
                abaqus_type = element_type_map.get(block.type)
                if abaqus_type is None:
                    print(f"⚠️ Skipping unsupported element type: {block.type}")
                    continue

                elset_name = f"ELSET{i+1}"
                f.write(f"*ELEMENT,TYPE={abaqus_type},ELSET={elset_name}\n")
                for conn in block.data:
                    conn_str = ", ".join(str(int(n) + 1) for n in conn)
                    f.write(f"{element_id}, {conn_str}\n")
                    element_id += 1

                if abaqus_type == "C3D4" or abaqus_type == "C3D8":
                    solid_elsets.append(elset_name)
                elif abaqus_type == "T3D2":
                    tus_elsets.append(elset_name)
                elif abaqus_type == "S3":
                    shel_elsets.append(elset_name)

            # Hardcoded material block (no input)
            # For tetras or hexas
            f.write("*MATERIAL, NAME=DefaultMaterial\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")  # Young's modulus, Poisson's ratio

            for elset in solid_elsets:
                f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL=DefaultMaterial\n")

            # For triangles
            f.write("*MATERIAL, NAME=myrock\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")
            for elset in shel_elsets:
                f.write(f"*SHELL SECTION, ELSET={elset}, MATERIAL=myrock\n")
                f.write("0.01\n")

            # For lines
            f.write("*MATERIAL, NAME=STEEL\n")
            f.write("*ELASTIC\n")
            f.write("2.100000E+05, 0.300000\n")
            for elset in tus_elsets:
                f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL=STEEL\n")
                f.write("0.01\n")

        print(f"✅ Abaqus .inp file '{filename}' written successfully.")



    def export_ansys(self, filename: str):
        """
        Export the mesh data to an Ansys file.
        Args:
            filename (str): The name of the Ansys file to export.
        """
        Ansys_in = AnsysInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        # Create mesh
        mesh = Ansys_in.create_mesh()

        # Write the mesh to an Exodus file
        mesh.write(filename, file_format="ansys")
        print(f"Ansys file '{filename}' created successfully!")

    def export_gmsh(self, filename: typing.Union[str, os.PathLike]):
        """
        Export the mesh data to a GMSH file.
        Supports both structured (hexahedral) and unstructured meshes.
        Args:
            filename: Path to the output .msh file.
        """
        gmsh_in = GMSHInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        mesh = gmsh_in.create_mesh()
        mesh.write(filename, file_format="gmsh22")
        print(f"GMSH file '{filename}' created successfully!")

    def export_stl(self, filename: str):
        """
        Export the mesh data to STL files.
        Args:
            filename (str): The name of the STL files to export.
        """
        stl_in = STLInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        stl_in.output_filename = filename  # <--- REQUIRED!
        stl_in.create_mesh()


    def export_vtm(self, filename: str):
        """
        Export the mesh input_data to a VTM file.
        Args:
            filename (str): The name of the VTM file to export.
        """
        self.mesh.save(filename)
        print(f"VTM file '{filename}' with multiple blocks created successfully!")

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
        Export mesh data to an Feflow.fem file with nodes, tetrahedral,
        and triangular and line elements.
        """
        feflow_in = FeflowInputs(nodes_array=self.nodes, elements_array=self.get_union_elems())
        mesh = feflow_in.create_mesh()
        node_array = mesh.points
        elements = mesh.cells

        # Gather all tetra, triangle, and line elements, and assign group-based markers
        tetra_all = []
        tetra_markers = []
        triangle_all = []
        triangle_markers = []
        edge_all = []
        edge_markers = []

        for idx, block in enumerate(elements):
            if block.type == "tetra":
                tetra_all.append(block.data)
                tetra_markers.append(np.full(len(block.data), idx + 1))  # use idx+1 as region ID
            elif block.type == "triangle":
                triangle_all.append(block.data)
                triangle_markers.append(np.full(len(block.data), idx + 1))  # same logic
            elif block.type == "line":
                edge_all.append(block.data)
                edge_markers.append(np.full(len(block.data), idx + 1))

        # Concatenate all
        tetra = np.vstack(tetra_all) if tetra_all else np.empty((0, 4), dtype=int)
        tetra_markers = np.concatenate(tetra_markers) if tetra_markers else None

        triangles = np.vstack(triangle_all) if triangle_all else np.empty((0, 3), dtype=int)
        triangle_markers = np.concatenate(triangle_markers) if triangle_markers else None

        edges = np.vstack(edge_all) if edge_all else np.empty((0, 2), dtype=int)
        edge_markers = np.concatenate(edge_markers) if edge_markers else None

        # Start writing FeFlow file
        with open(filename, 'w') as f:
            num_points = len(node_array)
            num_tetra = len(tetra)
            f.write("PROBLEM:\n")
            f.write("CLASS (v.7)\n")
            f.write("   2    1    0    3    0    0    8    8    0    0\n")
            f.write("DIMENS\n")
            f.write(f"   {num_points}     {num_tetra}     0      1      0      0      0      0      0      2     0      0      1      0      0      0      0\n")

            f.write("SCALE\n")
            f.write("   1.0, 1.0, 1.0, 1.0, 0.0, 0.0\n")
            f.write("VARNODE\n")
            f.write(f"   {num_tetra}     4     4\n")

            # Write tetrahedra connectivity (1-based)
            for t in range(num_tetra):
                tet_nodes = tetra[t] + 1
                f.write(f"   6     {tet_nodes[0]}     {tet_nodes[1]}     {tet_nodes[2]}     {tet_nodes[3]}\n")

            f.write("XYZCOOR\n")
            for x, y, z in node_array:
                f.write(f"     {x}, {y}, {z}\n")

            # ELEMENTALSETS
            if tetra_markers is not None:
                f.write("ELEMENTALSETS\n")
                for m in sorted(set(tetra_markers)):
                    f.write(f"     \"Region: Name: R{m}\"")
                    havewritten = 0
                    for t, mark in enumerate(tetra_markers):
                        if mark == m:
                            if (havewritten % 10) == 0:
                                f.write("\n\t\t")
                            f.write(f"{t + 1} ")
                            havewritten += 1
                    f.write("\n")

            # Create all unique triangles from tets
            FeFlowObj =C_FeFlow()
            print(tetra)
            FeFlowObj.generateAllTriangles(tetra)
            print('done')

            if triangle_markers is not None and len(triangle_markers) > 0:
                f.write("FACESETS\n")
                minMat = int(np.min(triangle_markers))
                maxMat = int(np.max(triangle_markers))

                for m in range(minMat, maxMat + 1):
                    mask = triangle_markers == m
                    triangles_with_marker = triangles[mask]

                    # Reuse FeFlowObj (already has all unique triangles from tets)
                    FeFlowObj.generateUndefinedTriangles(
                        marker=m,
                        triangle_markers=triangle_markers,
                        triangle_list=triangles
                    )
                    FeFlowObj.generateDefinedTriangles()

                    marker_triangle_indices = [tri.index for tri in FeFlowObj.definedTriangles]

                    f.write(f'     "Surface: Name: S{m}"')
                    havewritten = 0
                    for tri in marker_triangle_indices:
                        if (havewritten % 10) == 0:
                            f.write("\n\t\t")
                        f.write(f"{tri + 1} ")
                        havewritten += 1
                    f.write("\n")



            # EDGESETS (Lines)
            # Create all unique edges from lines
            FeFlowObj.generateAllEdges(tetra)

            if edge_markers is not None and len(edge_markers) > 0:
                f.write("EDGESETS\n")
                minEdgeMat = int(np.min(edge_markers))
                maxEdgeMat = int(np.max(edge_markers))

                for m in range(minEdgeMat, maxEdgeMat + 1):
                    marker_to_extract = m

                    # Boolean mask for edges with the desired marker
                    mask = edge_markers == marker_to_extract

                    # Apply mask to get edges block
                    edges_with_marker = edges[mask]
                    marker_edge_indices = FeFlowObj.generateMarkerEdges(edges_with_marker)

                    f.write(f'     "Polyline: Name: P{m}"')
                    havewritten = 0
                    for edge in marker_edge_indices:
                        if (havewritten % 10) == 0:
                            f.write("\n\t\t")
                        f.write(f"{edge + 1} ")
                        havewritten += 1
                    f.write("\n")


            f.write("END\n")

        print(f"✅ Feflow file '{filename}' written successfully.")
