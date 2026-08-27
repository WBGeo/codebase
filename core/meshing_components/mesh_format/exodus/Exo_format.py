import io
import logging
import os
import tempfile
from collections import defaultdict
from enum import StrEnum
from typing import Any, Dict, List, Optional

import meshio
import netCDF4 as nc
import numpy as np

from core.object_components import MeshResults
from py_api_wbgeo.nodesapi import BasicallyABufferedFile


logger = logging.getLogger(__name__)


class MeshType(StrEnum):
    UNSTRUCTURED = "unstr"
    STRUCTURED = "str"
    IMPLICIT = "imp"


class ExodusInput:
    """
    Create and export meshes in ExodusII format.

    The exporter writes a classic ExodusII-compatible NetCDF file
    suitable for consumers such as libMesh.

    Supported volume/element types:

        vertex
        line
        line3
        triangle
        quad
        tetra
        hexahedron
        wedge

    The important Exodus/libMesh element names used here are:

        SPHERE
        BAR2
        BAR3
        TRI3
        QUAD4
        TETRA4
        HEX8
        WEDGE6
    """

    MAX_STR_LENGTH = 32
    MAX_LINE_LENGTH = 80

    # ------------------------------------------------------------------
    # Element classification
    # ------------------------------------------------------------------

    VOLUME_TYPES = {
        "tetra",
        "tetra4",
        "hexahedron",
        "hex",
        "hex8",
        "wedge",
        "wedge6",
        "pyramid",
        "pyramid5",
    }

    SURFACE_TYPES = {
        "triangle",
        "tri",
        "tri3",
        "quad",
        "quad4",
    }

    EDGE_TYPES = {
        "line",
        "line2",
        "line3",
        "bar2",
        "bar3",
    }

    NODE_TYPES = {
        "vertex",
    }

    # ------------------------------------------------------------------
    # Exodus side/node tables
    #
    # These tables use Exodus side numbering.
    #
    # Hex8:
    #
    #   side 1 = nodes 1,2,6,5
    #   side 2 = nodes 2,3,7,6
    #   side 3 = nodes 3,4,8,7
    #   side 4 = nodes 1,5,8,4
    #   side 5 = nodes 1,4,3,2
    #   side 6 = nodes 5,6,7,8
    #
    # libMesh's Exodus helper has:
    #
    #   hex_face_map = {1,2,3,4,0,5}
    #
    # and therefore the Exodus side numbering is important here.
    # ------------------------------------------------------------------

    FACE_TABLES = {
        "tetra": {
            1: [0, 1, 3],
            2: [1, 2, 3],
            3: [0, 3, 2],
            4: [0, 2, 1],
        },

        "tetra4": {
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

        "hex": {
            1: [0, 1, 5, 4],
            2: [1, 2, 6, 5],
            3: [2, 3, 7, 6],
            4: [0, 4, 7, 3],
            5: [0, 3, 2, 1],
            6: [4, 5, 6, 7],
        },

        "hex8": {
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
        },

        "wedge6": {
            1: [0, 1, 4, 3],
            2: [1, 2, 5, 4],
            3: [0, 3, 5, 2],
            4: [0, 2, 1],
            5: [3, 4, 5],
        },

        # 2D elements.
        #
        # For a TRI3, Exodus sides are its three edges.
        "triangle": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 0],
        },

        "tri": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 0],
        },

        "tri3": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 0],
        },

        "quad": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 3],
            4: [3, 0],
        },

        "quad4": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 3],
            4: [3, 0],
        },

        # A BAR2 has two sides, corresponding to its two end nodes.
        "line": {
            1: [0],
            2: [1],
        },

        "line2": {
            1: [0],
            2: [1],
        },

        "bar2": {
            1: [0],
            2: [1],
        },
    }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        mesh: MeshResults,
        mesh_type: MeshType = MeshType.UNSTRUCTURED,
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

    # ------------------------------------------------------------------
    # Exodus element names
    # ------------------------------------------------------------------

    @staticmethod
    def exo_elem_type(
        cell_type: str,
        n_nodes: Optional[int] = None,
    ) -> str:
        """
        Convert meshio cell names to Exodus element names understood
        by libMesh.

        In particular:

            meshio hexahedron -> Exodus HEX8

        NOT:

            HEXAHEDRON
        """

        cell_type = str(cell_type).lower()

        # --------------------------------------------------------------
        # 0D
        # --------------------------------------------------------------

        if cell_type == "vertex":
            if n_nodes == 1:
                return "SPHERE"

            raise ValueError(
                f"Unsupported vertex element with {n_nodes} nodes"
            )

        # --------------------------------------------------------------
        # 1D
        # --------------------------------------------------------------

        if cell_type in {"line", "line2", "bar2"}:
            if n_nodes == 2:
                return "BAR2"

            raise ValueError(
                f"Unsupported line element with {n_nodes} nodes"
            )

        if cell_type in {"line3", "bar3"}:
            if n_nodes == 3:
                return "BAR3"

            raise ValueError(
                f"Unsupported quadratic line element with {n_nodes} nodes"
            )

        # --------------------------------------------------------------
        # 2D
        # --------------------------------------------------------------

        if cell_type in {"triangle", "tri", "tri3"}:
            if n_nodes == 3:
                return "TRI3"

            raise ValueError(
                f"Unsupported triangle element with {n_nodes} nodes"
            )

        if cell_type in {"quad", "quad4"}:
            if n_nodes == 4:
                return "QUAD4"

            raise ValueError(
                f"Unsupported quadrilateral element with {n_nodes} nodes"
            )

        # --------------------------------------------------------------
        # 3D
        # --------------------------------------------------------------

        if cell_type in {"tetra", "tetra4"}:
            if n_nodes == 4:
                return "TETRA4"

            raise ValueError(
                f"Unsupported tetrahedron element with {n_nodes} nodes"
            )

        if cell_type in {"hexahedron", "hex", "hex8"}:
            if n_nodes == 8:
                # CRITICAL:
                #
                # libMesh recognizes HEX / HEX8.
                # It does NOT recognize HEXAHEDRON.
                return "HEX8"

            raise ValueError(
                f"Unsupported hexahedron element with {n_nodes} nodes"
            )

        if cell_type in {"wedge", "wedge6"}:
            if n_nodes == 6:
                return "WEDGE6"

            raise ValueError(
                f"Unsupported wedge element with {n_nodes} nodes"
            )

        raise ValueError(
            f"Unsupported element type '{cell_type}' "
            f"with {n_nodes} nodes"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalized_cell_type(block: meshio.CellBlock) -> str:
        return str(block.type).lower()

    @classmethod
    def is_volume_block(cls, block: meshio.CellBlock) -> bool:
        return cls.normalized_cell_type(block) in cls.VOLUME_TYPES

    @classmethod
    def is_surface_block(cls, block: meshio.CellBlock) -> bool:
        return cls.normalized_cell_type(block) in cls.SURFACE_TYPES

    @classmethod
    def is_edge_block(cls, block: meshio.CellBlock) -> bool:
        return cls.normalized_cell_type(block) in cls.EDGE_TYPES

    @classmethod
    def is_node_block(cls, block: meshio.CellBlock) -> bool:
        return cls.normalized_cell_type(block) in cls.NODE_TYPES

    @classmethod
    def has_face_table(cls, cell_type: str) -> bool:
        return cell_type.lower() in cls.FACE_TABLES

    # ------------------------------------------------------------------
    # String arrays
    # ------------------------------------------------------------------

    @staticmethod
    def write_string_array(
        var: nc.Variable,
        strings: List[str],
    ) -> None:
        """
        Write a list of strings into an Exodus character array.
        """

        if len(var.shape) != 2:
            raise ValueError(
                f"Expected a 2D character array, got shape {var.shape}"
            )

        arr = np.full(
            var.shape,
            b" ",
            dtype="S1",
        )

        max_len = var.shape[-1]

        for i, value in enumerate(strings):
            if i >= var.shape[0]:
                break

            text = str(value)[:max_len]

            encoded = np.frombuffer(
                text.encode("ascii", errors="replace"),
                dtype="S1",
            )

            count = min(
                len(encoded),
                max_len,
            )

            if count:
                arr[i, :count] = encoded[:count]

        var[:] = arr

    # ------------------------------------------------------------------
    # Connectivity validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_connectivity(
        conn: np.ndarray,
        num_nodes: int,
        block_type: str,
    ) -> None:
        """
        Validate zero-based mesh connectivity before writing.
        """

        if conn.ndim != 2:
            raise ValueError(
                f"Connectivity for '{block_type}' must be 2D, "
                f"got shape {conn.shape}"
            )

        if conn.shape[0] == 0:
            return

        if not np.issubdtype(conn.dtype, np.integer):
            raise ValueError(
                f"Connectivity for '{block_type}' must contain integers"
            )

        min_node = int(conn.min())
        max_node = int(conn.max())

        if min_node < 0:
            raise ValueError(
                f"Negative node index {min_node} found in "
                f"'{block_type}' connectivity"
            )

        if max_node >= num_nodes:
            raise ValueError(
                f"Node index {max_node} found in '{block_type}' "
                f"connectivity, but mesh has only {num_nodes} nodes"
            )

    # ------------------------------------------------------------------
    # Side sets
    # ------------------------------------------------------------------

    def build_side_sets_unstructured(
        self,
        volume_blocks: List[meshio.CellBlock],
        boundary_blocks: List[meshio.CellBlock],
    ) -> List[Dict[str, Any]]:
        """
        Build Exodus side sets by matching boundary faces/edges/nodes
        against the volume elements.

        Element numbering is global across all volume blocks.
        """

        face_lookup: Dict[tuple, tuple] = {}

        elem_id = 1

        # --------------------------------------------------------------
        # Build lookup from volume element faces to:
        #
        #     (global element id, Exodus side number)
        # --------------------------------------------------------------

        for block in volume_blocks:
            conn = np.asarray(
                block.data,
                dtype=np.int64,
            )

            if conn.ndim == 1:
                conn = conn.reshape(-1, 1)

            cell_type = self.normalized_cell_type(block)

            face_table = self.FACE_TABLES.get(cell_type)

            if face_table is None:
                elem_id += len(conn)
                continue

            for elem in conn:
                for side_num, local_nodes in face_table.items():
                    face_nodes = tuple(
                        sorted(
                            int(elem[i])
                            for i in local_nodes
                        )
                    )

                    face_lookup[face_nodes] = (
                        elem_id,
                        side_num,
                    )

                elem_id += 1

        # --------------------------------------------------------------
        # Match boundary blocks
        # --------------------------------------------------------------

        side_sets: List[Dict[str, Any]] = []

        for side_id, block in enumerate(
            boundary_blocks,
            start=1,
        ):
            conn = np.asarray(
                block.data,
                dtype=np.int64,
            )

            if conn.ndim == 1:
                conn = conn.reshape(-1, 1)

            elem_list: List[int] = []
            side_list: List[int] = []

            for boundary_entity in conn:
                key = tuple(
                    sorted(
                        int(node)
                        for node in boundary_entity
                    )
                )

                match = face_lookup.get(key)

                if match is None:
                    logger.warning(
                        "Could not match boundary entity %s "
                        "of type '%s' to a volume element",
                        boundary_entity,
                        block.type,
                    )
                    continue

                global_elem_id, exodus_side = match

                elem_list.append(global_elem_id)
                side_list.append(exodus_side)

            side_sets.append(
                {
                    "id": side_id,
                    "name": f"sideset_{side_id}",
                    "elements": np.asarray(
                        elem_list,
                        dtype=np.int32,
                    ),
                    "sides": np.asarray(
                        side_list,
                        dtype=np.int32,
                    ),
                }
            )

        return side_sets

    # ------------------------------------------------------------------

    def build_side_sets_from_elements(
        self,
        volume_blocks: List[meshio.CellBlock],
    ) -> List[Dict[str, Any]]:
        """
        Automatically construct exterior side sets for structured and
        implicit meshes.

        Faces occurring once are exterior faces.
        """

        face_map: Dict[
            tuple,
            List[tuple],
        ] = defaultdict(list)

        elem_id = 1

        for block in volume_blocks:
            conn = np.asarray(
                block.data,
                dtype=np.int64,
            )

            if conn.ndim == 1:
                conn = conn.reshape(-1, 1)

            cell_type = self.normalized_cell_type(block)

            face_table = self.FACE_TABLES.get(cell_type)

            if face_table is None:
                elem_id += len(conn)
                continue

            for elem in conn:
                for side_num, local_nodes in face_table.items():
                    raw_face = [
                        int(elem[i])
                        for i in local_nodes
                    ]

                    key = tuple(sorted(raw_face))

                    face_map[key].append(
                        (
                            elem_id,
                            side_num,
                            raw_face,
                        )
                    )

                elem_id += 1

        exterior_faces = []

        for occurrences in face_map.values():
            if len(occurrences) == 1:
                exterior_faces.append(
                    occurrences[0]
                )

        grouped: Dict[int, List[int]] = defaultdict(list)

        for global_elem_id, side_num, _raw_face in exterior_faces:
            grouped[side_num].append(global_elem_id)

        side_sets: List[Dict[str, Any]] = []

        for side_id in sorted(grouped.keys()):
            elems = np.asarray(
                grouped[side_id],
                dtype=np.int32,
            )

            sides = np.full(
                len(elems),
                side_id,
                dtype=np.int32,
            )

            side_sets.append(
                {
                    "id": side_id,
                    "name": f"sideset_{side_id}",
                    "elements": elems,
                    "sides": sides,
                }
            )

        return side_sets

    # ------------------------------------------------------------------
    # Block classification
    # ------------------------------------------------------------------

    def classify_blocks(
        self,
        all_blocks: List[meshio.CellBlock],
    ):
        """
        Separate volume elements from boundary entities.

        The original implementation assumed that the last six blocks
        were always boundary blocks. That is not generally valid.

        Here we classify using the actual meshio cell type.
        """

        if self.mesh_type == MeshType.UNSTRUCTURED:

            volume_blocks = []
            boundary_blocks = []

            for block in all_blocks:
                cell_type = self.normalized_cell_type(block)

                if cell_type in self.VOLUME_TYPES:
                    volume_blocks.append(block)
                    continue

                if (
                    cell_type in self.SURFACE_TYPES
                    or cell_type in self.EDGE_TYPES
                    or cell_type in self.NODE_TYPES
                ):
                    boundary_blocks.append(block)
                    continue

                raise ValueError(
                    f"Unsupported meshio cell block type '{block.type}'"
                )

            if not volume_blocks:
                raise ValueError(
                    "No volume element blocks were found in the mesh"
                )

            side_sets = self.build_side_sets_unstructured(
                volume_blocks,
                boundary_blocks,
            )

            return volume_blocks, side_sets

        if self.mesh_type in (
            MeshType.STRUCTURED,
            MeshType.IMPLICIT,
        ):
            volume_blocks = all_blocks

            side_sets = self.build_side_sets_from_elements(
                volume_blocks,
            )

            return volume_blocks, side_sets

        raise ValueError(
            f"Unknown mesh type: {self.mesh_type}"
        )

    # ------------------------------------------------------------------
    # Write Exodus
    # ------------------------------------------------------------------

    def write(
        self,
        filename: str,
    ) -> None:
        """
        Write the mesh to an ExodusII NetCDF file.
        """

        # --------------------------------------------------------------
        # Nodes
        # --------------------------------------------------------------

        nodes = np.asarray(
            self.mesh.nodes,
            dtype=np.float64,
        )

        if nodes.ndim != 2:
            raise ValueError(
                f"Mesh nodes must be a 2D array, got {nodes.shape}"
            )

        num_nodes = nodes.shape[0]
        num_dim = nodes.shape[1]

        if num_dim < 1 or num_dim > 3:
            raise ValueError(
                f"Exodus export supports 1-3 dimensions, got {num_dim}"
            )

        # --------------------------------------------------------------
        # Element blocks
        # --------------------------------------------------------------

        all_blocks = list(self.mesh.elements)

        volume_blocks, side_sets = self.classify_blocks(
            all_blocks
        )

        elements = volume_blocks

        num_elem_blk = len(elements)

        num_elem = sum(
            len(block.data)
            for block in elements
        )

        # --------------------------------------------------------------
        # Node sets
        # --------------------------------------------------------------

        point_sets = self.mesh.point_sets or {}

        node_set_names = list(
            point_sets.keys()
        )

        num_node_sets = len(
            node_set_names
        )

        num_side_sets = len(
            side_sets
        )

        # --------------------------------------------------------------
        # Validate all volume blocks
        # --------------------------------------------------------------

        for block in elements:
            conn = np.asarray(
                block.data,
                dtype=np.int64,
            )

            if conn.ndim == 1:
                conn = conn.reshape(-1, 1)

            self.validate_connectivity(
                conn,
                num_nodes,
                str(block.type),
            )

            # Validate element type before opening the file.
            self.exo_elem_type(
                block.type,
                conn.shape[1],
            )

        # --------------------------------------------------------------
        # Create NetCDF / Exodus file
        # --------------------------------------------------------------

        ds = nc.Dataset(
            filename,
            "w",
            format="NETCDF3_64BIT_OFFSET",
        )

        try:
            ds.set_fill_off()

            # ----------------------------------------------------------
            # Global Exodus metadata
            # ----------------------------------------------------------

            ds.title = "WBGeo Exodus Export"

            # Exodus/libMesh expect these attributes as floating point.
            ds.api_version = np.float32(7.22)
            ds.version = np.float32(7.22)

            ds.floating_point_word_size = np.int32(8)

            # ----------------------------------------------------------
            # Dimensions
            # ----------------------------------------------------------

            ds.createDimension(
                "len_string",
                self.MAX_STR_LENGTH + 1,
            )

            ds.createDimension(
                "len_line",
                self.MAX_LINE_LENGTH + 1,
            )

            ds.createDimension(
                "four",
                4,
            )

            # Unlimited time dimension.
            ds.createDimension(
                "time_step",
                None,
            )

            ds.createDimension(
                "num_dim",
                num_dim,
            )

            ds.createDimension(
                "num_nodes",
                num_nodes,
            )

            ds.createDimension(
                "num_elem",
                num_elem,
            )

            ds.createDimension(
                "num_el_blk",
                num_elem_blk,
            )

            if num_node_sets > 0:
                ds.createDimension(
                    "num_node_sets",
                    num_node_sets,
                )

            if num_side_sets > 0:
                ds.createDimension(
                    "num_side_sets",
                    num_side_sets,
                )

            # ----------------------------------------------------------
            # QA records
            # ----------------------------------------------------------

            ds.createDimension(
                "num_qa_rec",
                1,
            )

            qa_records = ds.createVariable(
                "qa_records",
                "S1",
                (
                    "num_qa_rec",
                    "four",
                    "len_string",
                ),
            )

            qa_data = np.full(
                (
                    1,
                    4,
                    self.MAX_STR_LENGTH + 1,
                ),
                b" ",
                dtype="S1",
            )

            qa_strings = [
                "WBGeo",
                "1.0",
                "2025-01-01",
                "00:00:00",
            ]

            for j, text in enumerate(qa_strings):
                encoded = np.frombuffer(
                    text.encode("ascii"),
                    dtype="S1",
                )

                count = min(
                    len(encoded),
                    self.MAX_STR_LENGTH + 1,
                )

                qa_data[0, j, :count] = encoded[:count]

            qa_records[:] = qa_data

            # ----------------------------------------------------------
            # Time
            # ----------------------------------------------------------

            time_whole = ds.createVariable(
                "time_whole",
                "f8",
                ("time_step",),
            )

            time_whole[:] = np.asarray(
                [0.0],
                dtype=np.float64,
            )

            # ----------------------------------------------------------
            # Coordinates
            # ----------------------------------------------------------

            coord = ds.createVariable(
                "coord",
                "f8",
                (
                    "num_dim",
                    "num_nodes",
                ),
            )

            coord[0, :] = nodes[:, 0]

            if num_dim > 1:
                coord[1, :] = nodes[:, 1]

            if num_dim > 2:
                coord[2, :] = nodes[:, 2]

            # ----------------------------------------------------------
            # Coordinate names
            # ----------------------------------------------------------

            coor_names = ds.createVariable(
                "coor_names",
                "S1",
                (
                    "num_dim",
                    "len_string",
                ),
            )

            if num_dim == 1:
                coordinate_names = ["X"]
            elif num_dim == 2:
                coordinate_names = ["X", "Y"]
            else:
                coordinate_names = ["X", "Y", "Z"]

            self.write_string_array(
                coor_names,
                coordinate_names,
            )

            # ----------------------------------------------------------
            # Node number map
            # ----------------------------------------------------------

            node_num_map = ds.createVariable(
                "node_num_map",
                "i4",
                ("num_nodes",),
            )

            node_num_map[:] = np.arange(
                1,
                num_nodes + 1,
                dtype=np.int32,
            )

            # ----------------------------------------------------------
            # Element number map
            # ----------------------------------------------------------

            elem_num_map = ds.createVariable(
                "elem_num_map",
                "i4",
                ("num_elem",),
            )

            elem_num_map[:] = np.arange(
                1,
                num_elem + 1,
                dtype=np.int32,
            )

            # ----------------------------------------------------------
            # Element block status
            # ----------------------------------------------------------

            eb_status = ds.createVariable(
                "eb_status",
                "i4",
                ("num_el_blk",),
            )

            eb_status[:] = np.ones(
                num_elem_blk,
                dtype=np.int32,
            )

            # ----------------------------------------------------------
            # Element block IDs
            # ----------------------------------------------------------

            eb_prop1 = ds.createVariable(
                "eb_prop1",
                "i4",
                ("num_el_blk",),
            )

            eb_prop1.setncattr(
                "name",
                "ID",
            )

            eb_prop1[:] = np.arange(
                1,
                num_elem_blk + 1,
                dtype=np.int32,
            )

            # ----------------------------------------------------------
            # Element block names
            # ----------------------------------------------------------

            eb_names = ds.createVariable(
                "eb_names",
                "S1",
                (
                    "num_el_blk",
                    "len_string",
                ),
            )

            block_names = [
                f"block_{i + 1}"
                for i in range(num_elem_blk)
            ]

            self.write_string_array(
                eb_names,
                block_names,
            )

            # ----------------------------------------------------------
            # Element block connectivity
            # ----------------------------------------------------------

            for ib, block in enumerate(
                elements,
                start=1,
            ):
                conn = np.asarray(
                    block.data,
                    dtype=np.int64,
                )

                if conn.ndim == 1:
                    conn = conn.reshape(-1, 1)

                num_el = conn.shape[0]
                num_nod = conn.shape[1]

                exo_type = self.exo_elem_type(
                    block.type,
                    num_nod,
                )

                # Exodus dimensions are block-specific.
                ds.createDimension(
                    f"num_el_in_blk{ib}",
                    num_el,
                )

                ds.createDimension(
                    f"num_nod_per_el{ib}",
                    num_nod,
                )

                connect = ds.createVariable(
                    f"connect{ib}",
                    "i4",
                    (
                        f"num_el_in_blk{ib}",
                        f"num_nod_per_el{ib}",
                    ),
                )

                # This attribute is what Exodus uses for the block
                # element type.
                #
                # IMPORTANT:
                # hexahedron must be written as HEX8.
                connect.elem_type = exo_type

                # Meshio uses zero-based node indices.
                # Exodus uses one-based node indices.
                connectivity = conn + 1

                if np.any(
                    connectivity < 1
                ) or np.any(
                    connectivity > num_nodes
                ):
                    raise ValueError(
                        f"Invalid Exodus connectivity in block {ib}"
                    )

                connect[:] = np.asarray(
                    connectivity,
                    dtype=np.int32,
                )

            # ----------------------------------------------------------
            # Side sets
            # ----------------------------------------------------------

            if num_side_sets > 0:

                ss_status = ds.createVariable(
                    "ss_status",
                    "i4",
                    ("num_side_sets",),
                )

                ss_status[:] = np.ones(
                    num_side_sets,
                    dtype=np.int32,
                )

                ss_prop1 = ds.createVariable(
                    "ss_prop1",
                    "i4",
                    ("num_side_sets",),
                )

                ss_prop1.setncattr(
                    "name",
                    "ID",
                )

                ss_prop1[:] = np.asarray(
                    [
                        ss["id"]
                        for ss in side_sets
                    ],
                    dtype=np.int32,
                )

                ss_names = ds.createVariable(
                    "ss_names",
                    "S1",
                    (
                        "num_side_sets",
                        "len_string",
                    ),
                )

                side_names = [
                    ss["name"]
                    for ss in side_sets
                ]

                self.write_string_array(
                    ss_names,
                    side_names,
                )

                for i, ss in enumerate(
                    side_sets,
                    start=1,
                ):
                    elem_ids = np.asarray(
                        ss["elements"],
                        dtype=np.int32,
                    ).ravel()

                    side_ids = np.asarray(
                        ss["sides"],
                        dtype=np.int32,
                    ).ravel()

                    if len(elem_ids) != len(side_ids):
                        raise ValueError(
                            f"Sideset {i} has "
                            f"{len(elem_ids)} element IDs but "
                            f"{len(side_ids)} side IDs"
                        )

                    nside = len(elem_ids)

                    ds.createDimension(
                        f"num_side_ss{i}",
                        nside,
                    )

                    elem_ss = ds.createVariable(
                        f"elem_ss{i}",
                        "i4",
                        (f"num_side_ss{i}",),
                    )

                    side_ss = ds.createVariable(
                        f"side_ss{i}",
                        "i4",
                        (f"num_side_ss{i}",),
                    )

                    elem_ss[:] = elem_ids
                    side_ss[:] = side_ids

            # ----------------------------------------------------------
            # Node sets
            # ----------------------------------------------------------

            if num_node_sets > 0:

                ns_status = ds.createVariable(
                    "ns_status",
                    "i4",
                    ("num_node_sets",),
                )

                ns_status[:] = np.ones(
                    num_node_sets,
                    dtype=np.int32,
                )

                ns_prop1 = ds.createVariable(
                    "ns_prop1",
                    "i4",
                    ("num_node_sets",),
                )

                ns_prop1.setncattr(
                    "name",
                    "ID",
                )

                ns_prop1[:] = np.arange(
                    1,
                    num_node_sets + 1,
                    dtype=np.int32,
                )

                ns_names = ds.createVariable(
                    "ns_names",
                    "S1",
                    (
                        "num_node_sets",
                        "len_string",
                    ),
                )

                self.write_string_array(
                    ns_names,
                    node_set_names,
                )

                for i, set_name in enumerate(
                    node_set_names,
                    start=1,
                ):
                    node_ids = np.asarray(
                        point_sets[set_name],
                        dtype=np.int64,
                    ).ravel()

                    node_ids = np.unique(
                        node_ids
                    )

                    # MeshResults point sets are assumed to contain
                    # zero-based node indices.
                    if len(node_ids) > 0:
                        if node_ids.min() < 0:
                            raise ValueError(
                                f"Negative node index in node set "
                                f"'{set_name}'"
                            )

                        if node_ids.max() >= num_nodes:
                            raise ValueError(
                                f"Node index {node_ids.max()} in node "
                                f"set '{set_name}' is outside the mesh"
                            )

                    # Exodus uses one-based node numbering.
                    node_ids = node_ids + 1

                    n_nodes_in_set = len(
                        node_ids
                    )

                    ds.createDimension(
                        f"num_nod_ns{i}",
                        n_nodes_in_set,
                    )

                    node_list = ds.createVariable(
                        f"node_ns{i}",
                        "i4",
                        (f"num_nod_ns{i}",),
                    )

                    node_list[:] = np.asarray(
                        node_ids,
                        dtype=np.int32,
                    )

        finally:
            ds.close()


# ======================================================================
# WRAPPER
# ======================================================================

def export_mesh_results_to_exodus(
    mesh: MeshResults,
    type: MeshType = MeshType.UNSTRUCTURED,
) -> BasicallyABufferedFile:
    """
    Export a MeshResults object to an in-memory ExodusII file.

    The mesh is first written to a temporary ExodusII file and then
    loaded into a BytesIO buffer.

    Parameters
    ----------
    mesh:
        MeshResults containing nodes, elements, and optional sets.

    type:
        "unstr", "str", or "imp".

    Returns
    -------
    BasicallyABufferedFile
        In-memory ExodusII file with filename "mesh.exo".
    """

    if isinstance(type, str):
        try:
            type = MeshType(type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mesh type '{type}'. "
                f"Valid options are: "
                f"{[m.value for m in MeshType]}"
            )

    exo = ExodusInput(
        mesh=mesh,
        mesh_type=type,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".exo",
        delete=False,
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
