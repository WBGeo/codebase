"""
Write a SimulationResults (any simulator) as one multi-time-step Exodus II file.

The result already holds what a solver's Exodus output contains: geometry,
nodal and element variables per output time, and each cell's block. This
writer turns it back into a file ParaView and MOOSE tools can read, so
results don't have to keep the solver's original file around (see
SimulationResults). Not written: side sets, node sets and QA/info records;
those belong to the input mesh and the input file (kept in
SimulationResults.input_files), not to the results.
"""
import logging
from typing import Dict, List, Tuple

import netCDF4 as nc
import numpy as np

from core.object_components import SimulationResults

logger = logging.getLogger(__name__)

_LEN_STRING = 33  # Exodus default name length (32 characters + terminator)

# VTK cell type -> (Exodus element type, nodes per element). Node order of
# these types is the same in VTK and Exodus.
_VTK_TO_EXODUS = {
    1: ("SPHERE", 1),
    3: ("BAR2", 2),
    5: ("TRI3", 3),
    9: ("QUAD4", 4),
    10: ("TETRA4", 4),
    12: ("HEX8", 8),
    13: ("WEDGE6", 6),
    14: ("PYRAMID5", 5),
}

# Cell arrays that identify blocks (used for grouping, not written as variables).
_BLOCK_ID_FIELDS = ("ObjectId", "mat_id")
# Reader-generated id arrays; never results.
_ID_FIELDS = {
    "globalnodeid", "pedigreenodeid", "globalelementid", "pedigreeelementid",
    "globalcellid", "pedigreecellid", "objectid", "vtkoriginalpointids",
    "vtkoriginalcellids", "vtkghosttype",
}


def _char_array(names: List[str]) -> np.ndarray:
    data = np.full((len(names), _LEN_STRING), b"\0", dtype="S1")
    for i, name in enumerate(names):
        encoded = name.encode("ascii", "replace")[:_LEN_STRING - 1]
        data[i, :len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return data


def _split_cells(cells: np.ndarray, celltypes: np.ndarray) -> List[np.ndarray]:
    """VTK flat connectivity [n, ids..., n, ids...] -> one id array per cell."""
    result = []
    i = 0
    for _ in range(len(celltypes)):
        n = int(cells[i])
        result.append(cells[i + 1:i + 1 + n])
        i += n + 1
    if i != len(cells):
        raise ValueError("Cell connectivity does not match the number of cell types.")
    return result


def _scalar_fields(data: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    """Result arrays of length n as scalar components.

    Vectors are split into <name>x/<name>y/<name>z, the names Exodus readers
    (VTK, ParaView) join back into the vector <name>.
    """
    fields = {}
    for name, values in data.items():
        if name.lower() in _ID_FIELDS or name in _BLOCK_ID_FIELDS:
            continue
        array = np.asarray(values)
        if array.shape[0] != n or not np.issubdtype(array.dtype, np.number):
            continue
        if array.ndim == 1:
            fields[name] = array.astype(np.float64)
        elif array.ndim == 2:
            suffixes = ("x", "y", "z") if array.shape[1] == 3 else [f"_{k}" for k in range(array.shape[1])]
            for k, suffix in enumerate(suffixes):
                fields[f"{name}{suffix}"] = array[:, k].astype(np.float64)
    return fields


def _blocks(sim: SimulationResults, time: float, celltypes: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """(block id, VTK cell type, cell indices) per Exodus block, in block-id order."""
    cell_data = sim.cell_data_by_time.get(time, {})
    block_ids = None
    for field in _BLOCK_ID_FIELDS:
        if field in cell_data and len(cell_data[field]) == len(celltypes):
            block_ids = np.asarray(cell_data[field]).astype(np.int64).ravel()
            break
    if block_ids is None:
        # No block information: one block per cell type.
        block_ids = np.zeros(len(celltypes), dtype=np.int64)
    blocks = []
    for block_id in np.unique(block_ids):
        in_block = block_ids == block_id
        for celltype in np.unique(celltypes[in_block]):
            blocks.append((int(block_id), int(celltype), np.flatnonzero(in_block & (celltypes == celltype))))
    return blocks


def write_exodus_results(sim: SimulationResults, path: str) -> str:
    """Write every saved time step of `sim` into one Exodus II file at `path`.

    Requires the same geometry at every time step (true for SfePy and GOLEM
    results). Cells are grouped into element blocks by their ObjectId (GOLEM)
    or mat_id (SfePy) and cell type; block names come from
    sim.block_names where known. Vector fields are written as
    x/y/z components, which readers join back into the vector.

    :return: path
    """
    times = sorted(sim.nodes_by_time)
    if not times:
        raise ValueError("SimulationResults has no time steps to export.")
    t0 = times[0]
    nodes = np.asarray(sim.nodes_by_time[t0], dtype=np.float64)
    cells = np.asarray(sim.cells_by_time.get(t0, []), dtype=np.int64)
    celltypes = np.asarray(sim.celltypes_by_time.get(t0, []), dtype=np.int64)
    for t in times[1:]:
        if (not np.array_equal(sim.nodes_by_time[t], nodes)
                or not np.array_equal(sim.cells_by_time.get(t, []), cells)
                or not np.array_equal(sim.celltypes_by_time.get(t, []), celltypes)):
            raise ValueError(f"The mesh changes at time {t}; Exodus export needs the same mesh at every time step.")
    if len(celltypes) == 0:
        raise ValueError("SimulationResults has no cells to export.")
    unsupported = sorted(set(celltypes.tolist()) - set(_VTK_TO_EXODUS))
    if unsupported:
        raise ValueError(f"Cell types {unsupported} (VTK ids) are not supported by the Exodus export.")

    cell_nodes = _split_cells(cells, celltypes)
    blocks = _blocks(sim, t0, celltypes)
    block_names = dict(sim.block_names or {})

    node_vars = list(_scalar_fields(sim.node_data_by_time.get(t0, {}), len(nodes)))
    elem_vars = list(_scalar_fields(sim.cell_data_by_time.get(t0, {}), len(celltypes)))

    ds = nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET")
    try:
        ds.title = "WBGeo simulation results"
        ds.api_version = np.float32(7.22)
        ds.version = np.float32(7.22)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(1)

        ds.createDimension("len_string", _LEN_STRING)
        ds.createDimension("len_name", _LEN_STRING)
        ds.createDimension("len_line", 81)
        ds.createDimension("four", 4)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", len(nodes))
        ds.createDimension("num_elem", len(celltypes))
        ds.createDimension("num_el_blk", len(blocks))

        ds.createVariable("time_whole", "f8", ("time_step",))
        # file_size=1 ("large model") layout: one array per coordinate.
        for axis, name in enumerate(("coordx", "coordy", "coordz")):
            ds.createVariable(name, "f8", ("num_nodes",))[:] = nodes[:, axis]
        ds.createVariable("coor_names", "S1", ("num_dim", "len_name"))[:] = _char_array(["X", "Y", "Z"])

        ds.createVariable("eb_status", "i4", ("num_el_blk",))[:] = np.ones(len(blocks), dtype=np.int32)
        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        exodus_ids = []
        names = []
        for block_id, _, _ in blocks:
            # Exodus ids must be positive and unique; keep the source id when possible.
            exodus_id = block_id if block_id > 0 and block_id not in exodus_ids else (max(exodus_ids, default=0) + 1)
            exodus_ids.append(exodus_id)
            names.append(block_names.get(block_id, f"block_{exodus_id}"))
        eb_prop1[:] = np.asarray(exodus_ids, dtype=np.int32)
        ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))[:] = _char_array(names)

        ds.createVariable("elem_num_map", "i4", ("num_elem",))[:] = np.arange(1, len(celltypes) + 1, dtype=np.int32)
        ds.createVariable("node_num_map", "i4", ("num_nodes",))[:] = np.arange(1, len(nodes) + 1, dtype=np.int32)

        for k, (block_id, celltype, indices) in enumerate(blocks, start=1):
            exo_type, n_per = _VTK_TO_EXODUS[celltype]
            ds.createDimension(f"num_el_in_blk{k}", len(indices))
            ds.createDimension(f"num_nod_per_el{k}", n_per)
            connect = ds.createVariable(f"connect{k}", "i4", (f"num_el_in_blk{k}", f"num_nod_per_el{k}"))
            connect.elem_type = exo_type
            connect[:] = np.stack([cell_nodes[i] for i in indices]).astype(np.int32) + 1

        if node_vars:
            ds.createDimension("num_nod_var", len(node_vars))
            ds.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _char_array(node_vars)
            for k in range(len(node_vars)):
                ds.createVariable(f"vals_nod_var{k + 1}", "f8", ("time_step", "num_nodes"))
        if elem_vars:
            ds.createDimension("num_elem_var", len(elem_vars))
            ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))[:] = _char_array(elem_vars)
            ds.createVariable("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"))[:] = np.ones(
                (len(blocks), len(elem_vars)), dtype=np.int32)
            for v in range(len(elem_vars)):
                for k in range(1, len(blocks) + 1):
                    ds.createVariable(f"vals_elem_var{v + 1}eb{k}", "f8", ("time_step", f"num_el_in_blk{k}"))

        for step, t in enumerate(times):
            ds.variables["time_whole"][step] = float(t)
            node_values = _scalar_fields(sim.node_data_by_time.get(t, {}), len(nodes))
            for k, name in enumerate(node_vars):
                if name not in node_values:
                    raise ValueError(f"Node variable {name!r} is missing at time {t}.")
                ds.variables[f"vals_nod_var{k + 1}"][step, :] = node_values[name]
            elem_values = _scalar_fields(sim.cell_data_by_time.get(t, {}), len(celltypes))
            for v, name in enumerate(elem_vars):
                if name not in elem_values:
                    raise ValueError(f"Element variable {name!r} is missing at time {t}.")
                for k, (_, _, indices) in enumerate(blocks, start=1):
                    ds.variables[f"vals_elem_var{v + 1}eb{k}"][step, :] = elem_values[name][indices]
        logger.debug("Wrote %d time step(s), %d block(s) to %s", len(times), len(blocks), path)
    finally:
        ds.close()
    return path
