import logging
import io
import meshio
import numpy as np
import netCDF4 as nc
import tempfile
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional
from core.object_components import MeshResults
from py_api_wbgeo.nodesapi import BasicallyABufferedFile
from enum import Enum, StrEnum

logger = logging.getLogger(__name__)

class MeshType(StrEnum):
    UNSTRUCTURED = "unstr"
    STRUCTURED = "str"
    IMPLICIT = "imp"


######################
# EXODUS INPUT CLASS
#####################
class ExodusInput:
    """
    Create and export meshes in ExodusII format.
    This class converts mesh data stored in a MeshResults object
    into an ExodusII-compatible NetCDF file. It supports structured,
    unstructured, and implicit mesh types, including:
    - Node coordinates
    - Element connectivity
    - Element blocks
    - Side sets (boundary faces)
    - Node sets (point groups)
    - Exodus metadata and QA records

    Supported element types:
        - vertex
        - line
        - triangle
        - quad
        - tetra
        - hexahedron
        - wedge

    Parameters
    mesh (MeshResults): Mesh container holding nodes, elements, and optional sets.
    mesh_type (str, optional): Type of mesh topology:
            - "unstr" : unstructured mesh
            - "str"   : structured mesh
            - "imp"   : implicit mesh
        Default is "unstr".
    """

    MAX_STR_LENGTH = 32
    MAX_LINE_LENGTH = 80

    ##############
    # FACE TABLES
    ##############
    FACE_TABLES = {

        "tetra": {
            1: [0, 1, 3],
            2: [1, 2, 3],
            3: [0, 3, 2],
            4: [0, 2, 1],
        },

        "hexahedron": {
            1: [0, 1, 5, 4],
            2: [1, 2, 6, 5],
            3: [2, 3, 7, 6],
            4: [0, 4, 7, 3],
            5: [0, 3, 2, 1],
            6: [4, 5, 6, 7],
        },

        "wedge": {
            1: [0, 1, 4, 3],
            2: [1, 2, 5, 4],
            3: [0, 3, 5, 2],
            4: [0, 2, 1],
            5: [3, 4, 5],
        }
    }


    ########
    # INIT
    #######
    def __init__(
        self,
        mesh: MeshResults,
        mesh_type: MeshType = MeshType.UNSTRUCTURED
    ):
        if isinstance(mesh_type, str):
            try:
                mesh_type = MeshType(mesh_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid mesh_type='{mesh_type}'. "
                    f"Valid options are: {[m.value for m in MeshType]}"
                )

        self.mesh = mesh
        self.mesh_type = mesh_type
    ######################
    # EXODUS ELEMENT TYPES
    ######################
    @staticmethod
    def exo_elem_type(cell_type: str, n_nodes: Optional[int] = None) -> str:

        cell_type = cell_type.lower()

        # vertex
        if cell_type == "vertex":

            if n_nodes == 1:
                return "SPHERE"
            raise ValueError(
                f"Unsupported vertex element with {n_nodes} nodes"
            )

        # line

        if cell_type == "line":
            if n_nodes == 2:
                return "BAR2"

            elif n_nodes == 3:
                return "BAR3"

            raise ValueError(
                f"Unsupported line element with {n_nodes} nodes"
            )

        # validation only (no remapping)
        allowed = {"triangle", "quad", "tetra", "hexahedron", "wedge"}

        if cell_type not in allowed:

            raise ValueError(f"Unsupported element type: {cell_type}")

        return cell_type

    ###############
    # WRITE STRINGS
    ###############
    @staticmethod
    def write_string_array(var: nc.Variable, strings: List[str]) -> None:

        arr = np.full(
            var.shape,
            b" ",
            dtype="S1"
        )

        for i, s in enumerate(strings):

            s = s[:var.shape[-1]]

            encoded = np.frombuffer(
                s.encode("ascii"),
                dtype="S1"
            )

            arr[i, :len(encoded)] = encoded

        var[:] = arr

    ##################################
    # BUILD SIDE SETS FOR UNSTRUCTURED
    ##################################
    def build_side_sets_unstructured(
        self,
        volume_blocks: List[meshio.CellBlock],
        boundary_blocks: List[meshio.CellBlock],
    ) -> List[Dict[str, Any]]:

        face_lookup = {}

        elem_id = 1

        for blk in volume_blocks:

            conn = np.asarray(
                blk.data,
                dtype=np.int32
            )

            etype = blk.type.lower()

            if etype not in self.FACE_TABLES:
                elem_id += len(conn)
                continue

            face_table = self.FACE_TABLES[etype]

            for elem in conn:

                for side_num, local_nodes in face_table.items():
                    face_nodes = tuple(
                        sorted(elem[local_nodes])
                    )

                    face_lookup[face_nodes] = (
                        elem_id,
                        side_num
                    )

                elem_id += 1

        side_sets = []

        for side_id, blk in enumerate(
            boundary_blocks,
            start=1
        ):

            conn = np.asarray(
                blk.data,
                dtype=np.int32
            )

            elem_list = []
            side_list = []

            for face in conn:

                key = tuple(sorted(face))

                match = face_lookup.get(key)

                if match is None:

                    logger.warning("Face not matched: %s", face)

                    continue

                elem_id, side_num = match
                elem_list.append(elem_id)
                side_list.append(side_num)

            side_sets.append({
                "id": side_id,
                "name": f"sideset_{side_id}",
                "elements": np.asarray(
                    elem_list,
                    dtype=np.int32
                ),
                "sides": np.asarray(
                    side_list,
                    dtype=np.int32
                ),
            })

        return side_sets

    ###########################################
    # BUILD SIDE SETS FOR STRUCTURED / IMPLICIT
    ###########################################
    def build_side_sets_from_elements(
        self,
        volume_blocks: List[meshio.CellBlock],
    ) -> List[Dict[str, Any]]:

        face_map = defaultdict(list)

        elem_id = 1

        for blk in volume_blocks:

            conn = np.asarray(
                blk.data,
                dtype=np.int32
            )

            etype = blk.type.lower()

            if etype not in self.FACE_TABLES:
                elem_id += len(conn)
                continue

            face_table = self.FACE_TABLES[etype]

            for elem in conn:

                for side_num, local_nodes in face_table.items():
                    raw_face = elem[local_nodes]
                    key = tuple(sorted(raw_face))
                    face_map[key].append(
                        (
                            elem_id,
                            side_num,
                            raw_face
                        )
                    )

                elem_id += 1

        exterior_faces = []

        for key, occurrences in face_map.items():

            if len(occurrences) == 1:
                exterior_faces.append(
                    occurrences[0]
                )

        grouped = defaultdict(list)

        for elem_id, side_num, raw_face in exterior_faces:
            grouped[side_num].append(elem_id)

        side_sets = []

        for side_num in sorted(grouped.keys()):
            elems = np.asarray(
                grouped[side_num],
                dtype=np.int32
            )

            sides = np.full(
                len(elems),
                side_num,
                dtype=np.int32
            )

            side_sets.append({
                "id": side_num,
                "name": f"sideset_{side_num}",
                "elements": elems,
                "sides": sides,
            })

        return side_sets

    ##############
    # WRITE EXODUS
    ##############
    def write(self, filename: str) -> None:

        nodes = np.asarray(
            self.mesh.nodes,
            dtype=np.float64
        )

        all_blocks = self.mesh.elements

        # MESH TYPE HANDLING
        if self.mesh_type == MeshType.UNSTRUCTURED:

            NUM_SIDE_BLOCKS = 6

            if len(all_blocks) < NUM_SIDE_BLOCKS:
                raise ValueError(
                    f"Mesh has only {len(all_blocks)} blocks"
                )

            boundary_blocks = all_blocks[-NUM_SIDE_BLOCKS:]
            volume_blocks = all_blocks[:-NUM_SIDE_BLOCKS]

            side_sets = self.build_side_sets_unstructured(
                volume_blocks,
                boundary_blocks
            )

        elif self.mesh_type in [MeshType.STRUCTURED, MeshType.IMPLICIT]:
            volume_blocks = all_blocks
            side_sets = self.build_side_sets_from_elements(
                volume_blocks
            )

        else:
            raise ValueError(
                f"Unknown mesh type: {self.mesh_type}"
            )

        elements = volume_blocks

        # COUNTS
        num_nodes = nodes.shape[0]
        num_dim = nodes.shape[1]
        num_elem_blk = len(elements)

        num_elem = sum(
            len(b.data)
            for b in elements
        )

        num_side_sets = len(side_sets)
        point_sets = self.mesh.point_sets or {}

        node_set_names = list(
            point_sets.keys()
        )

        num_node_sets = len(node_set_names)

        # CREATE FILE
        ds = nc.Dataset(
            filename,
            "w",
            format="NETCDF3_64BIT_OFFSET"
        )

        ds.set_fill_off()

        # GLOBAL ATTRIBUTES
        ds.title = "WBGeo Exodus Export"

        ds.api_version = np.float32(7.22)

        ds.version = np.float32(7.22)

        ds.floating_point_word_size = 8

        # DIMENSIONS
        ds.createDimension(
            "len_string",
            self.MAX_STR_LENGTH + 1
        )

        ds.createDimension(
            "len_line",
            self.MAX_LINE_LENGTH + 1
        )

        ds.createDimension("four", 4)

        ds.createDimension(
            "time_step",
            None
        )

        ds.createDimension(
            "num_dim",
            num_dim
        )

        ds.createDimension(
            "num_nodes",
            num_nodes
        )

        ds.createDimension(
            "num_elem",
            num_elem
        )

        ds.createDimension(
            "num_el_blk",
            num_elem_blk
        )

        if num_node_sets > 0:
            ds.createDimension(
                "num_node_sets",
                num_node_sets
            )

        if num_side_sets > 0:
            ds.createDimension(
                "num_side_sets",
                num_side_sets
            )

        # QA RECORDS
        ds.createDimension(
            "num_qa_rec",
            1
        )

        qa_records = ds.createVariable(
            "qa_records",
            "S1",
            (
                "num_qa_rec",
                "four",
                "len_string"
            )
        )

        qa_data = np.full(
            (
                1,
                4,
                self.MAX_STR_LENGTH + 1
            ),
            b" ",
            dtype="S1"
        )

        qa_strings = [
            "WBGeo",
            "1.0",
            "2025-01-01",
            "00:00:00"
        ]

        for j, txt in enumerate(qa_strings):
            enc = np.frombuffer(
                txt.encode("ascii"),
                dtype="S1"
            )

            qa_data[0, j, :len(enc)] = enc

        qa_records[:] = qa_data

        # TIME
        time_whole = ds.createVariable(
            "time_whole",
            "f8",
            ("time_step",)
        )

        time_whole[:] = [0.0]

        # COORDINATES
        coord = ds.createVariable(
            "coord",
            "f8",
            ("num_dim", "num_nodes")
        )

        coord[0, :] = nodes[:, 0]

        if num_dim > 1:
            coord[1, :] = nodes[:, 1]

        if num_dim > 2:
            coord[2, :] = nodes[:, 2]

        # COORD NAMES
        coor_names = ds.createVariable(
            "coor_names",
            "S1",
            ("num_dim", "len_string")
        )

        if num_dim == 1:
            names = ["X"]

        elif num_dim == 2:
            names = ["X", "Y"]

        else:
            names = ["X", "Y", "Z"]

        self.write_string_array(
            coor_names,
            names
        )

        # NODE MAP
        node_num_map = ds.createVariable(
            "node_num_map",
            "i4",
            ("num_nodes",)
        )

        node_num_map[:] = np.arange(
            1,
            num_nodes + 1
        )

        # ELEMENT MAP
        elem_num_map = ds.createVariable(
            "elem_num_map",
            "i4",
            ("num_elem",)
        )

        elem_num_map[:] = np.arange(
            1,
            num_elem + 1
        )

        # ELEMENT BLOCKS
        eb_status = ds.createVariable(
            "eb_status",
            "i4",
            ("num_el_blk",)
        )

        eb_status[:] = np.ones(
            num_elem_blk,
            dtype=np.int32
        )

        eb_prop1 = ds.createVariable(
            "eb_prop1",
            "i4",
            ("num_el_blk",)
        )

        eb_prop1.setncattr(
            "name",
            "ID"
        )

        eb_prop1[:] = np.arange(
            1,
            num_elem_blk + 1
        )

        eb_names = ds.createVariable(
            "eb_names",
            "S1",
            ("num_el_blk", "len_string")
        )

        block_names = [
            f"block_{i+1}"
            for i in range(num_elem_blk)
        ]

        self.write_string_array(
            eb_names,
            block_names
        )

        for ib, block in enumerate(
            elements,
            start=1
        ):

            conn = np.asarray(
                block.data,
                dtype=np.int32
            )

            if conn.ndim == 1:
                conn = conn.reshape(-1, 1)

            num_el = conn.shape[0]
            num_nod = conn.shape[1]

            exo_type = self.exo_elem_type(
                block.type,
                num_nod
            )

            ds.createDimension(
                f"num_el_in_blk{ib}",
                num_el
            )

            ds.createDimension(
                f"num_nod_per_el{ib}",
                num_nod
            )

            connect = ds.createVariable(
                f"connect{ib}",
                "i4",
                (
                    f"num_el_in_blk{ib}",
                    f"num_nod_per_el{ib}"
                )
            )

            connect.elem_type = exo_type
            connect[:] = conn + 1

        # SIDE SETS
        if num_side_sets > 0:

            ss_status = ds.createVariable(
                "ss_status",
                "i4",
                ("num_side_sets",)
            )

            ss_status[:] = np.ones(
                num_side_sets,
                dtype=np.int32
            )

            ss_prop1 = ds.createVariable(
                "ss_prop1",
                "i4",
                ("num_side_sets",)
            )

            ss_prop1.setncattr(
                "name",
                "ID"
            )

            ss_prop1[:] = np.arange(
                1,
                num_side_sets + 1
            )

            ss_names = ds.createVariable(
                "ss_names",
                "S1",
                (
                    "num_side_sets",
                    "len_string"
                )
            )

            side_names = [
                ss["name"]
                for ss in side_sets
            ]

            self.write_string_array(
                ss_names,
                side_names
            )

            for i, ss in enumerate(
                side_sets,
                start=1
            ):

                nside = len(
                    ss["elements"]
                )

                ds.createDimension(
                    f"num_side_ss{i}",
                    nside
                )

                elem_ss = ds.createVariable(
                    f"elem_ss{i}",
                    "i4",
                    (f"num_side_ss{i}",)
                )

                side_ss = ds.createVariable(
                    f"side_ss{i}",
                    "i4",
                    (f"num_side_ss{i}",)
                )

                elem_ss[:] = ss["elements"]


                side_ss[:] = ss["sides"]

        # NODE SETS
        if num_node_sets > 0:

            ns_status = ds.createVariable(
                "ns_status",
                "i4",
                ("num_node_sets",)
            )

            ns_status[:] = np.ones(
                num_node_sets,
                dtype=np.int32
            )

            ns_prop1 = ds.createVariable(
                "ns_prop1",
                "i4",
                ("num_node_sets",)
            )

            ns_prop1.setncattr(
                "name",
                "ID"
            )

            ns_prop1[:] = np.arange(
                1,
                num_node_sets + 1
            )

            ns_names = ds.createVariable(
                "ns_names",
                "S1",
                (
                    "num_node_sets",
                    "len_string"
                )
            )

            self.write_string_array(
                ns_names,
                node_set_names
            )

            for i, set_name in enumerate(
                node_set_names,
                start=1
            ):

                node_ids = np.asarray(
                    point_sets[set_name],
                    dtype=np.int32
                ).ravel()

                node_ids = np.unique(
                    node_ids
                )

                node_ids = node_ids + 1
                n_nodes_in_set = len(
                    node_ids
                )

                ds.createDimension(
                    f"num_nod_ns{i}",
                    n_nodes_in_set
                )

                node_list = ds.createVariable(
                    f"node_ns{i}",
                    "i4",
                    (f"num_nod_ns{i}",)
                )

                node_list[:] = node_ids

        # CLOSE
        ds.close()

#########
# WRAPPER
# ########
def export_mesh_results_to_exodus(
    mesh: MeshResults,
    type: MeshType = MeshType.UNSTRUCTURED,
) -> BasicallyABufferedFile:
    """
    Export a MeshResults object to an in-memory ExodusII file.
    This is a convenience wrapper around the ExodusInput class.
    The mesh is written to a temporary ExodusII file, then loaded
    into a BytesIO buffer for further use such as downloading,
    saving, or transferring through APIs.

    Parameters
    mesh (MeshResults): Mesh object containing nodes, element blocks, and optional node/side sets.
    type (str, optional): Mesh type definition:
            - "unstr" : unstructured mesh
            - "str"   : structured mesh
            - "imp"   : implicit mesh
        Default is "unstr".

    Returns
    io.BytesIO: In-memory buffer containing the ExodusII file data.
        The buffer includes a filename attribute ("mesh.exo").

    Notes
    Temporary files created during export are automatically removed.
    """

    if isinstance(type, str):
        try:
            type = MeshType(type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mesh type '{type}'. "
                f"Valid options are: {[m.value for m in MeshType]}"
            )
    exo = ExodusInput(
        mesh=mesh,
        mesh_type=type
    )

    with tempfile.NamedTemporaryFile(
        suffix=".exo",
        delete=False
    ) as tmp:

        path = tmp.name

    try:

        exo.write(path)

        with open(path, "rb") as f:
            buf = io.BytesIO(
                f.read()
            )

        buf.filename = "mesh.exo"

    finally:

        if os.path.exists(path):

            os.remove(path)

    return buf
