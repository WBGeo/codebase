"""Focused regression tests; extract runner helpers without importing WBGeo."""
import ast
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import logging
import pytest

SOURCE = Path(__file__).with_name("golem_run.py").read_text()
TREE = ast.parse(SOURCE)

def helper(name):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"queue": queue, "shlex": shlex, "subprocess": subprocess, "threading": threading, "time": time, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), "golem_run.py", "exec"), namespace)
    return namespace[name]

def test_stream_silent_by_default(capsys):
    output = helper("_stream_process")(["python", "-c", "print('hello')"], heartbeat_seconds=0.1, stream_output=False)
    assert "hello" in output
    assert capsys.readouterr().out == ""

def test_stream_opt_in(capsys):
    output = helper("_stream_process")(["python", "-c", "print('hello')"], heartbeat_seconds=0.1, stream_output=True)
    assert "hello" in output
    assert "hello" in capsys.readouterr().out

def test_stream_failure_reports_output():
    with pytest.raises(RuntimeError, match="GOLEM exited with status 7") as err:
        helper("_stream_process")(["python", "-c", "print('failure'); raise SystemExit(7)"], stream_output=False)
    assert "failure" in str(err.value)

def test_runner_exposes_disabled_default():
    runner = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "run_simulation_golem")
    assert runner.args.args[-1].arg == "stream_output"
    assert isinstance(runner.args.defaults[-1], ast.Constant) and runner.args.defaults[-1].value is False

def test_local_failure_uses_single_readable_exception():
    runner = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "run_simulation_golem")
    raises = [n for n in ast.walk(runner) if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name)]
    assert not any(n.exc.func.id == "ValueError" and len(n.exc.args) > 1 for n in raises)
    assert "Working directory:" in SOURCE and "exit code" in SOURCE
