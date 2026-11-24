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
    np.testing.assert_allclose(np.unique(pts, axis=0), np.unique(nodes, axis=0))


def test_vtk_bookkeeping_arrays_are_not_stored_but_block_ids_are(two_block_exodus):
    path, _ = two_block_exodus

    result = load_exodus_results(path)

    (time,) = result.nodes_by_time
    stored = set(result.node_data_by_time[time]) | set(result.cell_data_by_time[time])
    assert not stored & {"GlobalNodeId", "PedigreeNodeId", "GlobalElementId", "PedigreeElementId"}
    assert sorted(np.unique(result.cell_data_by_time[time]["ObjectId"]).tolist()) == [1, 2]


def test_time_steps_share_one_geometry(tmp_path):
    from core.object_components import SimulationResults
    from core.simulation_components.output_format.exodus.exodus_results_writer import write_exodus_results

    nodes = np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)])
    sim = SimulationResults(block_names={0: "rock"})
    for t in (0.0, 1.0, 2.0):
        sim.nodes_by_time[t] = nodes
        sim.cells_by_time[t] = np.array([8, 0, 1, 3, 2, 4, 5, 7, 6])  # one hexahedron (VTK layout)
        sim.celltypes_by_time[t] = np.array([12])
        sim.node_data_by_time[t] = {"temperature": np.full(8, 10.0 + t)}
        sim.cell_data_by_time[t] = {"mat_id": np.array([0.0])}
    path = write_exodus_results(sim, str(tmp_path / "steps.e"))

    result = load_exodus_results(path)

    assert sorted(result.nodes_by_time) == [0.0, 1.0, 2.0]
    for field in (result.nodes_by_time, result.cells_by_time, result.celltypes_by_time):
        assert len({id(array) for array in field.values()}) == 1
    for t in (0.0, 1.0, 2.0):  # the field values still differ per step
        np.testing.assert_allclose(result.node_data_by_time[t]["temperature"], 10.0 + t)
