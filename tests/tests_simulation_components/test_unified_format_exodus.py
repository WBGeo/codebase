import os

import meshio
import numpy as np
import pytest

from core.meshing_components.mesh_format.exodus.Exo_format import (
    MeshType,
    export_mesh_results_to_exodus,
)
from core.object_components import MeshResults
from core.simulation_components.output_format.exodus.unified_format_exodus import (
    load_exodus_results,
)


@pytest.fixture
def two_block_exodus(tmp_path):
    """Two hexahedra in separate element blocks sharing one face (4 nodes),
    written with WBGeo's exporter."""
    nodes = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)]
    )
    # node index = x + 3*y + 6*z
    left = [0, 1, 4, 3, 6, 7, 10, 9]
    right = [1, 2, 5, 4, 7, 8, 11, 10]
    mesh = MeshResults(
        nodes=nodes,
        elements=[
            meshio.CellBlock("hexahedron", np.array([left])),
            meshio.CellBlock("hexahedron", np.array([right])),
        ],
    )
    path = os.path.join(tmp_path, "two_blocks.e")
    with open(path, "wb") as fh:
        fh.write(export_mesh_results_to_exodus(mesh, type=MeshType.STRUCTURED).getvalue())
    return path, nodes


def test_nodes_shared_between_blocks_are_loaded_once(two_block_exodus):
    path, nodes = two_block_exodus

    result = load_exodus_results(path)

    (time,) = result.nodes_by_time
    pts = np.asarray(result.nodes_by_time[time])
    assert len(pts) == len(nodes)
    assert len(result.celltypes_by_time[time]) == 2

    # Every loaded point is a distinct file node at its original position.
    gid = np.asarray(result.node_data_by_time[time]["GlobalNodeId"])
    assert sorted(gid) == list(range(1, len(nodes) + 1))
    np.testing.assert_allclose(pts, nodes[gid - 1])
