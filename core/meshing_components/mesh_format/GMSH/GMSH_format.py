import numpy as np
import meshio
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from numpy.typing import NDArray
import io
import os
from py_api_wbgeo.nodesapi import wbgeo_component
import tempfile

class GMSHInputs:
    """
    Class to construct a Gmsh compatible `meshio.Mesh`
    from raw nodes and meshio CellBlocks.

    The class:
    - Groups elements into Gmsh entities per topological dimension
      (points, curves, surfaces, volumes)
    - Assigns gmsh:physical and gmsh:geometrical tags
    - Constructs gmsh:dim_tags for nodes

    """

    #: Mapping from meshio cell types to Gmsh topological dimensions
    CELL_DIM = {
        "vertex": 0,
        "line": 1,
        "line3": 1,
        "triangle": 2,
        "quad": 2,
        "tetra": 3,
        "hexahedron": 3,
    }

    def __init__(
        self,
        nodes,
        elements: List[meshio.CellBlock],
    ) -> None:

        # Normalize nodes to NumPy
        self.points: NDArray[np.float64] = np.asarray(nodes, dtype=float)
        self.blocks: List[meshio.CellBlock] = elements  # List[meshio.CellBlock]

        self.cells, self.cell_data, self.node_entities = self._build_entities()
        self.point_data = self._build_point_dim_tags()

    # Build entities correctly (PER DIMENSION)

    def _build_entities(self) -> Tuple[List[Tuple[str, NDArray[np.int64]]], Dict[str, List[NDArray[np.int64]]],
                                       Dict[int, Set[Tuple[int, int]]]]:
        """
        Build Gmsh entities per topological dimension.

        Each meshio CellBlock is treated as exactly one Gmsh entity.
        Entity numbering is independent per dimension, as required by Gmsh.

        Returns:
        cells (list): List of (cell_type, connectivity) tuples for meshio.
        cell_data (dict): Dictionary with keys:
              - "gmsh:physical"
              - "gmsh:geometrical"
        node_entities (dict): Mapping: node_index → set of (dimension, entity_tag).
        """
        cells: List[Tuple[str, NDArray[np.int64]]] = []
        physical: List[NDArray[np.int64]] = []
        geometrical: List[NDArray[np.int64]] = []

        entity_counter: Dict[int, int] = defaultdict(int)
        node_entities: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

        for cb in self.blocks:
            if cb.type not in self.CELL_DIM:
                continue
            if len(cb.data) == 0:
                continue

            dim: int = self.CELL_DIM[cb.type]
            entity_counter[dim] += 1
            tag: int = entity_counter[dim]

            cells.append((cb.type, cb.data))

            n: int = len(cb.data)
            physical.append(np.full(n, tag, dtype=np.int64))
            geometrical.append(np.full(n, tag, dtype=np.int64))

            # REGISTER NODE → ENTITY RELATION
            for nid in np.unique(cb.data):
                node_entities[nid].add((dim, tag))

        cell_data: Dict[str, List[NDArray[np.int64]]] = {
            "gmsh:physical": physical,
            "gmsh:geometrical": geometrical,
        }

        return cells, cell_data, node_entities

    # gmsh:dim_tags — MUST MATCH ELEMENT ENTITIES
    def _build_point_dim_tags(self) -> Dict[str, NDArray[np.int64]]:
        """
        Construct gmsh:dim_tags for all nodes.

        Each node must belong to exactly one (dimension, entity_tag).
        If a node belongs to multiple entities, the entity with the
        lowest dimension is selected (curve < surface < volume),
        which is Gmsh-safe.

        Returns:
        point_data (dict): Dictionary containing "gmsh:dim_tags".
        """
        n_points: int = self.points.shape[0]
        dim_tags: NDArray[np.int64] = np.zeros((n_points, 2), dtype=np.int64)

        for nid in range(n_points):
            ents = self.node_entities.get(nid)

            if not ents:
                # orphan → assign to volume 1
                dim_tags[nid] = (3, 1)
            else:
                # choose LOWEST dimension entity (Gmsh-safe)
                dim, tag = sorted(ents, key=lambda x: x[0])[0]
                dim_tags[nid] = (dim, tag)

        return {"gmsh:dim_tags": dim_tags}

    # Create mesh
    def create_mesh(self) -> meshio.Mesh:
        """
        Create a fully Gmsh 4.x compatible meshio.Mesh.

        Returns:
        mesh (meshio.Mesh):  Mesh ready for export using meshio.write(..., format="gmsh").
        """
        mesh: meshio.Mesh = meshio.Mesh(
            points=self.points,
            cells=self.cells,
            cell_data=self.cell_data,
            point_data=self.point_data,
        )
        return mesh



@wbgeo_component(
    title="Download Mesh as Gmsh",
    description="Export Mesh to Gmsh",
    group="Export",
    identifier="wbgeo::expert_mesh_results_gmsh",
)
def export_mesh_results_to_gmsh(mesh: "MeshResults"):
    """
    Export the given MeshResults object to a Gmsh file.
    The mesh is first written to a temporary file using the WBGeo
    Exporters interface (required by Gmsh), then read back into memory
    and returned as a downloadable file.

    Notes
    -----
    - Only unstructured meshes are supported.
    - Structured meshes are not supported and may lead to invalid output.

    Args:
    mesh (MeshResults): WBGeo mesh object containing nodes, elements, and metadata to be exported.

    Returrns:
    io.BytesIO: In-memory buffer containing the Gmsh `.msh` file, ready for download.
    """
    from core.object_components import Exporters

    # Create exporters from MeshResults
    exporters = Exporters(**mesh.__dict__)

    # Write to a real temporary file (REQUIRED for Gmsh)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = tmp.name
        exporters.export_gmsh(tmp_path)

    # Read back into memory
    with open(tmp_path, "rb") as f:
        buf = io.BytesIO(f.read())

    # Name for WBGeo download
    buf.filename = "mesh_export_gmsh.msh"
    # Remove the temporary file immediately
    os.remove(tmp_path)
    return buf

