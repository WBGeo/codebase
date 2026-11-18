"""Numeric CSV tables <-> SimulationResults.solver_tables entries.

Solver CSV output (GOLEM/MOOSE postprocessors and vector postprocessors)
is a header row followed by numeric rows. It is stored in the result as
{column: array} so it survives the Workbench's JSON round trip, can be
used directly (e.g. plotted as a time series) and is written back as CSV
on export.
"""
import csv
import io
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_numeric_csv(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Read a CSV of a header row and numeric rows as {column: float array}.

    Returns None (with a warning naming the file) when the file is not
    such a table, instead of guessing at its content.
    """
    with open(path, newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if not rows:
        logger.warning("Solver CSV %s is empty; not stored in the result.", path.name)
        return None
    header, body = [name.strip() for name in rows[0]], rows[1:]
    if not all(header) or len(set(header)) != len(header):
        logger.warning("Solver CSV %s has empty or repeated column names; not stored in the result.", path.name)
        return None
    if any(len(row) != len(header) for row in body):
        logger.warning("Solver CSV %s has rows of different length; not stored in the result.", path.name)
        return None
    try:
        values = np.array([[float(cell) for cell in row] for row in body], dtype=np.float64).reshape(-1, len(header))
    except ValueError:
        logger.warning("Solver CSV %s has non-numeric values; not stored in the result.", path.name)
        return None
    return {name: values[:, k].copy() for k, name in enumerate(header)}


def numeric_csv_text(table: Dict[str, np.ndarray]) -> str:
    """CSV text of a {column: array} table (columns in stored order)."""
    columns = list(table)
    data = [np.asarray(table[name], dtype=np.float64).ravel() for name in columns]
    if data and len({len(column) for column in data}) != 1:
        raise ValueError("All columns of a table must have the same length.")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for row in zip(*data):
        writer.writerow([repr(float(value)) for value in row])
    return out.getvalue()
