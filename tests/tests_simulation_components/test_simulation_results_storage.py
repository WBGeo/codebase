"""
SimulationResults as the Workbench stores it, and the CSV table helpers.

The Workbench saves a result as JSON of its declared fields and rebuilds
it with SimulationResults(**json); everything an export or inspector needs
must survive that.
"""
import json
import logging

import numpy as np
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
