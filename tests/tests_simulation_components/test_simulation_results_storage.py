"""
SimulationResults as the Workbench stores it, and the CSV table helpers.

The Workbench saves a result as JSON of its declared fields and rebuilds
it with SimulationResults(**json); everything an export or inspector needs
must survive that.
"""
import json
import logging

import numpy as np
import pydantic
import pytest
from pydantic_core import to_jsonable_python

from core.object_components import SimulationResults
from core.simulation_components.output_format.csv_tables import numeric_csv_text, read_numeric_csv


def _full_result():
    sim = SimulationResults(
        solver="golem",
        solver_output={"golem": "done"},
        solver_tables={"wbgeo.csv": {"time": np.array([0.0, 1.0]), "q": np.array([2.5, 3.5])}},
        input_files={"wbgeo.i": "[Mesh]\n[]\n"},
        block_names={1: "rock1", 2: "fault_1"},
    )
    sim.nodes_by_time[1.0] = np.zeros((2, 3))
    sim.node_data_by_time[1.0] = {"temperature": np.array([1.0, 2.0])}
    return sim


def test_all_fields_survive_the_workbench_json_round_trip():
    sim = _full_result()
    rebuilt = SimulationResults(**json.loads(json.dumps(to_jsonable_python(sim))))

    assert rebuilt.solver == "golem"
    assert rebuilt.solver_output == {"golem": "done"}
    assert rebuilt.input_files == {"wbgeo.i": "[Mesh]\n[]\n"}
    assert rebuilt.block_names == {1: "rock1", 2: "fault_1"}  # int keys restored from JSON strings
    assert list(rebuilt.solver_tables["wbgeo.csv"]) == ["time", "q"]
    np.testing.assert_array_equal(rebuilt.solver_tables["wbgeo.csv"]["q"], [2.5, 3.5])
    np.testing.assert_array_equal(rebuilt.node_data_by_time[1.0]["temperature"], [1.0, 2.0])


def _transient_result(n_steps, n_nodes=50):
    """n_steps saved steps on one unchanged mesh; only "T" changes between steps."""
    nodes, cells, celltypes = np.random.rand(n_nodes, 3), np.arange(9 * 10, dtype=np.int64), np.full(10, 12)
    mat_id = np.repeat([0.0, 1.0], 5)
    sim = SimulationResults(solver="sfepy", block_names={0: "rock1", 1: "rock2"})
    for i in range(n_steps):
        t = float(i * 10)
        sim.nodes_by_time[t], sim.cells_by_time[t], sim.celltypes_by_time[t] = nodes, cells, celltypes
        sim.node_data_by_time[t] = {"T": np.full(n_nodes, 10.0 + i), "p": np.linspace(0, 1, n_nodes)}
        sim.cell_data_by_time[t] = {"mat_id": mat_id.copy()}  # equal but not the same object
    return sim


def _workbench_json(sim):
    return pydantic.TypeAdapter(SimulationResults).dump_json(sim)


def test_unchanged_arrays_are_stored_once():
    stored = json.loads(_workbench_json(_transient_result(4)))

    assert set(stored["nodes_by_time"]) == {"__shared__", "__times__"}
    assert stored["nodes_by_time"]["__times__"] == [0.0, 10.0, 20.0, 30.0]
    assert set(stored["node_data_by_time"]["__shared__"]) == {"p"}
    assert all(set(step) == {"T"} for step in stored["node_data_by_time"]["__steps__"].values())
    assert set(stored["cell_data_by_time"]["__shared__"]) == {"mat_id"}


def test_stored_size_does_not_grow_with_the_number_of_steps_for_static_parts():
    # GitHub issue #3: with an unchanged mesh only the changing field adds per step.
    size = {n: len(_workbench_json(_transient_result(n))) for n in (2, 20)}
    one_step_of_t = len(json.dumps(np.full(50, 10.0).tolist()))
    assert size[20] - size[2] < 18 * (one_step_of_t + 40)


@pytest.mark.parametrize("route", ["validate_json", "constructor"])
def test_compact_json_round_trip_restores_steps_and_shares_arrays(route):
    sim = _transient_result(3)
    blob = _workbench_json(sim)
    if route == "validate_json":
        rebuilt = pydantic.TypeAdapter(SimulationResults).validate_json(blob)
    else:
        rebuilt = SimulationResults(**json.loads(blob))

    assert sorted(rebuilt.nodes_by_time) == [0.0, 10.0, 20.0]
    for t in rebuilt.nodes_by_time:
        np.testing.assert_allclose(rebuilt.nodes_by_time[t], sim.nodes_by_time[t])
        np.testing.assert_array_equal(rebuilt.cells_by_time[t], sim.cells_by_time[t])
        np.testing.assert_allclose(rebuilt.node_data_by_time[t]["T"], sim.node_data_by_time[t]["T"])
        np.testing.assert_allclose(rebuilt.node_data_by_time[t]["p"], sim.node_data_by_time[t]["p"])
        np.testing.assert_allclose(rebuilt.cell_data_by_time[t]["mat_id"], sim.cell_data_by_time[t]["mat_id"])
    # the rebuilt result holds one copy of each unchanged array, not one per step
    assert len({id(a) for a in rebuilt.nodes_by_time.values()}) == 1
    assert len({id(step["p"]) for step in rebuilt.node_data_by_time.values()}) == 1
    assert len({id(step["mat_id"]) for step in rebuilt.cell_data_by_time.values()}) == 1


def test_changing_geometry_and_single_steps_keep_the_per_step_format():
    sim = _transient_result(2)
    sim.nodes_by_time[10.0] = sim.nodes_by_time[10.0] + 1.0  # moved mesh at the second step
    stored = json.loads(_workbench_json(sim))
    assert set(stored["nodes_by_time"]) == {"0.0", "10.0"}
    rebuilt = SimulationResults(**stored)
    np.testing.assert_allclose(rebuilt.nodes_by_time[10.0], sim.nodes_by_time[10.0])

    single = json.loads(_workbench_json(_transient_result(1)))
    assert set(single["nodes_by_time"]) == {"0.0"}


def test_results_stored_in_the_old_per_step_format_still_load():
    old = {
        "nodes_by_time": {"0.0": [[0, 0, 0]], "1.0": [[0, 0, 0]]},
        "node_data_by_time": {"0.0": {"T": [1.0]}, "1.0": {"T": [2.0]}},
        "block_names": {"1": "rock1"},
    }
    rebuilt = SimulationResults(**old)
    assert sorted(rebuilt.nodes_by_time) == [0.0, 1.0]
    np.testing.assert_allclose(rebuilt.node_data_by_time[1.0]["T"], [2.0])


def test_defaults_are_empty_not_shared():
    a, b = SimulationResults(), SimulationResults()
    a.solver_output["x"] = "y"
    assert a.solver is None and b.solver_output == {} and b.solver_tables == {}


def test_csv_round_trip(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("time,flow rate\n0,1.5\n10,2e-3\n")
    table = read_numeric_csv(path)
    assert list(table) == ["time", "flow rate"]
    np.testing.assert_array_equal(table["flow rate"], [1.5, 0.002])
    assert numeric_csv_text(table).splitlines() == ["time,flow rate", "0.0,1.5", "10.0,0.002"]


def test_csv_header_only_gives_empty_columns(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("time,x\n")
    assert {k: v.size for k, v in read_numeric_csv(path).items()} == {"time": 0, "x": 0}


@pytest.mark.parametrize("content, reason", [
    ("", "empty"),
    ("time,name\n0,abc\n", "non-numeric"),
    ("a,a\n1,2\n", "repeated"),
    ("a,b\n1,2,3\n", "different length"),
])
def test_csv_that_is_not_a_numeric_table_is_reported_not_guessed(tmp_path, caplog, content, reason):
    path = tmp_path / "odd.csv"
    path.write_text(content)
    with caplog.at_level(logging.WARNING):
        assert read_numeric_csv(path) is None
    assert "odd.csv" in caplog.text and reason in caplog.text


def test_csv_text_rejects_ragged_tables():
    with pytest.raises(ValueError, match="same length"):
        numeric_csv_text({"a": np.array([1.0]), "b": np.array([1.0, 2.0])})
