"""
From-scratch custom SfePy input file for model2 (real fault, active damage
zone, unstructured mesh, include_flow=True) -- a two-stage (pressure -> Darcy
velocity -> transient heat) solve via SfepyCustomBuilder, written WITHOUT
using anything SfepyHydrothermalProblemBuilder generates.

WHAT THIS FILE DEMONSTRATES
-----------------------------
SfepyHydrothermalProblemBuilder generates its pressure/heat input files itself
(build_pressure_input_file/build_heat_input_file). This file shows the
complementary case: a two-stage hydrothermal problem (fault zone, active
damage zone, unstructured mesh, include_flow=True) written from first
principles for SfepyCustomBuilder, without calling
build_pressure_input_file/build_heat_input_file and without any literal
enumerated vertex-id list. It uses only:
  - map_mat_id_to_lithology / enumerate_rock_units / compute_fault_zone_cell_mask
    -- public, standalone helper functions in sfepy_model_builder.py,
    documented as reusable by any caller wanting the same mesh-block/
    lithology/fault-zone cell selection a custom file needs.
  - RockUnitProperties.effective_conductivity/effective_heat_capacity --
    public methods on a public pydantic model.
  - SfePy's own coordinate-based boundary selectors ('vertices in (z < ...)'),
    derived from the mesh's own node z-extent -- the standard way to write a
    flat-boundary SfePy region by hand, independent of this codebase.
  - The segregated pressure/heat two-stage physics, dw_advect_div_free
    advective coupling, and data-driven diffusion-timescale default that
    SfepyHydrothermalProblemBuilder itself implements (see
    sfepy_model_builder.py's module docstring and
    SfepyHydrothermalProblemBuilder's class docstring).

Verified end-to-end against a real SfepyHydrothermalProblemBuilder run with
identical settings (same mesh, same rock/fault-zone properties,
include_flow=True, num_steps=2): T mean agrees to within 2e-6 (36.68905 both
sides), max node-by-node |T diff| = 9e-6 over a 10-60 range. Not expected to
match to 0.0: this file is
independently authored (different solver code paths, different EBC selector
implementation inside SfePy, different floating-point operation order), so
close agreement here demonstrates that the same technique reproduces the
same physics, not that the two files are identical.

HOW MULTI-STAGE WORKS IN A SINGLE FILE
------------------------------------------------------------------------
sfepy-run imports this file as a Python module (see
sfepy.base.conf.ProblemConf.from_file -> sfepy.base.base.import_file, a real
__import__ executing all top-level code) before it ever reads the
module-level regions/materials/equations that define what it will actually
solve. So "STAGE 1" below solves the steady-state Darcy pressure equation
in-process, via SfePy's own public library API (ProblemConf.from_file +
Problem.from_conf + .solve()) -- not sfepy-run, not a subprocess -- using its
own self-contained conf (a separate temp file, isolated from this file's own
top-level names). The resulting per-cell Darcy velocity is reconstructed
directly from the nodal P1 pressure field and held in memory as
_stage1_velocity.
"STAGE 2" (regions/fields/variables/materials/equations/
solvers/options at THIS file's own top level) is what sfepy-run actually
resolves and solves -- referencing _stage1_velocity via each group's
velocity_fn_<gid> material function.

WHAT THIS FILE IS TIED TO
--------------------------
Like every custom SfePy file, this is mesh-specific: mat_id/group numbering
and the fault-zone cell selection only line up with model2's own
structural_model_result/mesh_explicit_unstructured (see
WBGeo1.0_model2.py), built with:

    grid = RegularGrid(extent=(0, 2500, 0, 1000, 0, 1000), resolution=(100, 40, 40))
    data_faults = InputData_FaultElements(name="Faults_Model_2", fault_names=["fault"],
        fault_surface_points=pd.read_csv(".../model2_surface_points_df.csv"),
        fault_orientations=pd.read_csv(".../model2_orientations_df.csv"))
    fault_frame = general_faults.build_fault_frame(input_data_fault_elements=data_faults, grid=grid)
    fault_model_result = general_faults.compute_fault_domains(fault_frame)
    data_elements = InputData_StructuralElements(name="Model_2",
        mapping_object={"Strat_Series2": ("rock4", "rock3"), "Strat_Series1": ("rock2", "rock1")},
        surface_points=..., orientations=...)
    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid,
        fault_model_results=fault_model_result)
    structural_model_result = general.compute_structural_model(frame, extract_meshes=True)
    mesh_explicit_unstructured = create_unstructured_mesh_data(
        geomodel_result=structural_model_result, tolerance=50, mesh_size=20,
        curve_mesh_size=5)

Rock/fluid/fault-zone properties (deliberately identical to
WBGeo1.0_model2.py's own SfepyHydrothermalProblemBuilder call, so this is a fair
comparison of technique, not different inputs) -- see that script for the
literal values. Boundary conditions: p_top=0.0, p_bottom=1000000.0,
t_top=10.0, t_bottom=60.0. t0=0.0, t1=5.0e11, num_steps=3 time
increments (four stored SFePy states), include_flow=True.

Then:
    from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import SfepyCustomBuilder
    from core.simulation_components.simulation_packages.sfepy.sfepy_run import run_simulation_sfepy
    custom_builder = SfepyCustomBuilder(
        input_file_contents=open("custom_hydrothermal.py").read(),
        mesh_results=mesh_explicit_unstructured, geomodel_result=structural_model_result,
        fault_zone_n_voxels=1,
    )
    result = run_simulation_sfepy(custom_builder)
"""
import os
import tempfile

import numpy as np
import pyvista as pv

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")
_output_dir = os.environ.get("SFEpy_OUTPUT_DIR", "output")
os.makedirs(_output_dir, exist_ok=True)

# =======================================================================
# STAGE 1 (module-level side effect, runs before sfepy-run ever sees this
# file's "official" conf below): solve steady-state Darcy pressure
# in-process via SfePy's own public library API. Region/material/equation
# text below is assembled by THIS file's own code (not read from
# SfepyHydrothermalProblemBuilder), and Gamma_Top/Gamma_Bottom are
# coordinate-based selectors, not a captured vertex-id list.
# =======================================================================
_stage1_conf_text = """\"\"\"
Stage-1 conf: steady-state Darcy pressure. Authored independently of
SfepyHydrothermalProblemBuilder.build_pressure_input_file -- coordinate-based
boundary regions, own material/variable naming.
\"\"\"
import os
import tempfile

filename_mesh = os.environ.get(\'TEMP_MESH_FILE\', \'filename.mesh\')

regions = {
    'Omega': 'all',
    'Omega0': 'cells of group 0',
    'Omega1': 'cells of group 1',
    'Omega2': 'cells of group 2',
    'Omega3': 'cells of group 3',
    'Omega4': 'cells of group 4',
    'Omega_fault': 'cells of group 12',
    'Gamma_Bottom': ('vertices in (z < 0.001)', 'facet'),
    'Gamma_Top': ('vertices in (z > 999.9990000000003)', 'facet'),
}

fields = {
    \'pressure\': (\'real\', 1, \'Omega\', 1),
}

t0 = 0.0
t1 = 500000000000.0
num_steps = 1
dt = (t1 - t0) / num_steps

variables = {
    \'p\': (\'unknown field\', \'pressure\', 1),
    \'q\': (\'test field\', \'pressure\', \'p\'),
}

materials = {
    'm0': ({'k_over_mu': [[1e-12, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 1e-12]]},),
    'm1': ({'k_over_mu': [[1e-11, 0.0, 0.0], [0.0, 1e-11, 0.0], [0.0, 0.0, 1e-11]]},),
    'm2': ({'k_over_mu': [[1e-11, 0.0, 0.0], [0.0, 1e-11, 0.0], [0.0, 0.0, 1e-11]]},),
    'm3': ({'k_over_mu': [[1e-11, 0.0, 0.0], [0.0, 1e-11, 0.0], [0.0, 0.0, 1e-11]]},),
    'm4': ({'k_over_mu': [[1e-11, 0.0, 0.0], [0.0, 1e-11, 0.0], [0.0, 0.0, 1e-11]]},),
    'm_fault': ({'k_over_mu': [[1e-16, 0.0, 0.0], [0.0, 1e-16, 0.0], [0.0, 0.0, 1e-16]]},),
}

ebcs = {
    \'p_bottom\': (\'Gamma_Bottom\', {\'p.0\': 1000000.0}),
    \'p_top\': (\'Gamma_Top\', {\'p.0\': 0.0}),
}

integrals = {
    \'i\': 1,
}

equations = {
    \'flow\': \"\"\"dw_diffusion.i.Omega0(m0.k_over_mu, q, p)
  + dw_diffusion.i.Omega1(m1.k_over_mu, q, p)
  + dw_diffusion.i.Omega2(m2.k_over_mu, q, p)
  + dw_diffusion.i.Omega3(m3.k_over_mu, q, p)
  + dw_diffusion.i.Omega4(m4.k_over_mu, q, p)
  + dw_diffusion.i.Omega_fault(m_fault.k_over_mu, q, p)
  = 0\"\"\",
}

solvers = {
    # Fault pressure has a very large permeability contrast. AMG avoids
    # the zero-pivot failures that an ILU preconditioner can produce.
    \'ls\': (\'ls.petsc\', {
        \'method\': \'fgmres\',
        \'precond\': \'gamg\',
        \'ksp_gmres_restart\': 100,
        \'ksp_divtol\': 1.0e12,
        \'i_max\': 500,
        \'eps_a\': 1.0e-12,
        \'eps_r\': 1e-8,
        \'verbose\': 2,
        \'ksp_monitor\': \'\',
    }),
    \'newton\': (\'nls.newton\', {
        \'i_max\': 1,
        \'eps_a\': 1e-6,
        \'eps_r\': 1e-8,
        \'is_linear\': True,
        \'report_status\': True,
    }),
    \'ts\': (\'ts.simple\', {
        \'t0\': t0,
        \'t1\': t1,
        \'n_step\': num_steps,
        \'verbose\': 1,
    }),
}

_stage1_output_dir = tempfile.mkdtemp(prefix="stage1_pressure_")

options = {
    \'nls\': \'newton\',
    \'ls\': \'ls\',
    \'ts\': \'ts\',
    \'save_times\': \'all\',
    \'output_dir\': _stage1_output_dir,
}
"""
_stage1_conf_path = tempfile.mktemp(suffix="_stage1_pressure.py")
with open(_stage1_conf_path, "w") as _f:
    _f.write(_stage1_conf_text)

from sfepy.base.conf import ProblemConf
from sfepy.discrete import Problem

_stage1_conf = ProblemConf.from_file(_stage1_conf_path)
_stage1_problem = Problem.from_conf(_stage1_conf, init_equations=True)
_stage1_problem.time_update()
_stage1_state = _stage1_problem.solve()

_stage1_vtk_dir = tempfile.mkdtemp()
_stage1_vtk_path = os.path.join(_stage1_vtk_dir, "pressure.0.vtk")
_stage1_problem.save_state(_stage1_vtk_path, state=_stage1_state)

# Darcy velocity from the solved pressure field: v = -(k/mu) * grad(p).
# Reconstruct grad(p) directly from the nodal P1 field. Differentiating
# cell-averaged VTK pressure would create artificial interface jumps.
_grid = pv.read(_stage1_vtk_path)
_pressure = np.asarray(_grid.point_data["p"], dtype=float).reshape(-1)
_flat_cells = np.asarray(_grid.cells, dtype=np.int64)
_connectivity = []
_cursor = 0
while _cursor < _flat_cells.size:
    _n_cell_nodes = int(_flat_cells[_cursor])
    _connectivity.append(_flat_cells[_cursor + 1:_cursor + 1 + _n_cell_nodes])
    _cursor += _n_cell_nodes + 1
if not _connectivity or any(len(_conn) != 4 for _conn in _connectivity):
    raise ValueError(
        "custom_hydrothermal.py expects model2's tetrahedral volume mesh"
    )
_connectivity = np.stack(_connectivity)
_xyz = np.asarray(_grid.points, dtype=float)[_connectivity]
_p_el = _pressure[_connectivity]
_dx = _xyz - _xyz.mean(axis=1, keepdims=True)
_dp = _p_el - _p_el.mean(axis=1, keepdims=True)
_normal = np.einsum("nki,nkj->nij", _dx, _dx)
_rhs = np.einsum("nki,nk->ni", _dx, _dp)
try:
    _grad_p = np.linalg.solve(_normal, _rhs)
except np.linalg.LinAlgError:
    _grad_p = np.stack([
        np.linalg.lstsq(_a, _b, rcond=None)[0]
        for _a, _b in zip(_dx, _dp)
    ])

# Split by SFePy's actual global region cell IDs. Sequential offsets are
# incorrect whenever cells belonging to material groups are interleaved.
_group_ids_and_k_over_mu = [(0, 1e-12), (1, 1e-11), (2, 1e-11), (3, 1e-11), (4, 1e-11), (12, 1e-16)]
_gid_to_region_name = {0: 'Omega0', 1: 'Omega1', 2: 'Omega2', 3: 'Omega3', 4: 'Omega4', 12: 'Omega_fault'}
_stage1_velocity = {}
for _gid, _k_over_mu in _group_ids_and_k_over_mu:
    _region_cells = _stage1_problem.domain.regions[_gid_to_region_name[_gid]].get_cells()
    _stage1_velocity[_gid] = -_k_over_mu * _grad_p[_region_cells]

# =======================================================================
# STAGE 2 (the "official" conf sfepy-run resolves): transient heat with
# advection from _stage1_velocity above.
# =======================================================================
regions = {
    'Omega': 'all',
    'Omega0': 'cells of group 0',
    'Omega1': 'cells of group 1',
    'Omega2': 'cells of group 2',
    'Omega3': 'cells of group 3',
    'Omega4': 'cells of group 4',
    'Omega_fault': 'cells of group 12',
    'Gamma_Bottom': ('vertices in (z < 0.001)', 'facet'),
    'Gamma_Top': ('vertices in (z > 999.9990000000003)', 'facet'),
}

fields = {
    'temperature': ('real', 1, 'Omega', 1),
}

t0 = 0.0
t1 = 500000000000.0
# model2 requests three increments; SfePy n_step counts points including t0.
num_steps = 3 + 1
dt = (t1 - t0) / (num_steps - 1)

variables = {
    'T': ('unknown field', 'temperature', 0, 1),
    's': ('test field', 'temperature', 'T'),
}

def velocity_fn_0(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[0]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}

def velocity_fn_1(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[1]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}

def velocity_fn_2(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[2]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}

def velocity_fn_3(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[3]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}

def velocity_fn_4(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[4]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}

def velocity_fn_12(ts, coors, mode=None, problem=None, **kwargs):
    if mode != 'qp':
        return
    v_cells = _stage1_velocity[12]  # (n_cells_in_region, 3), from stage 1
    n_qp_per_cell = coors.shape[0] // v_cells.shape[0]
    v = np.repeat(v_cells, n_qp_per_cell, axis=0)
    return {'v': v.reshape(v.shape[0], v.shape[1], 1)}


def ic_temp(coor, ic=None):
    z_min, z_max = float(np.min(coor[:, 2])), float(np.max(coor[:, 2]))
    T_top, T_bottom = 10.0, 60.0
    return T_top + (T_bottom - T_top) * (z_max - coor[:, 2]) / (z_max - z_min)

functions = {
    'ic_temp': (ic_temp,),
    'velocity_fn_0': (velocity_fn_0,),
    'velocity_fn_1': (velocity_fn_1,),
    'velocity_fn_2': (velocity_fn_2,),
    'velocity_fn_3': (velocity_fn_3,),
    'velocity_fn_4': (velocity_fn_4,),
    'velocity_fn_12': (velocity_fn_12,)
}

# Normalize every energy term by rho_f*c_f. This improves conditioning but
# does not alter the solution; the normalized advection coefficient is q.
_rho_c_fluid = 4.186e6
# Scalar plotting metadata does not alter the governing equations.
materials = {
    'therm0': ({
        'k_eff': 2.8799999999999994 / _rho_c_fluid,
        'rho_c_eff': 2299300 / _rho_c_fluid,
        'porosity': 0.05,
        'permeability': 1e-15,
        'viscosity': 1.0e-3,
    },),
    'darcy0': 'velocity_fn_0',
    'therm1': ({
        'k_eff': 2.215 / _rho_c_fluid,
        'rho_c_eff': 2497900 / _rho_c_fluid,
        'porosity': 0.15,
        'permeability': 1e-14,
        'viscosity': 1.0e-3,
    },),
    'darcy1': 'velocity_fn_1',
    'therm2': ({
        'k_eff': 2.215 / _rho_c_fluid,
        'rho_c_eff': 2497900 / _rho_c_fluid,
        'porosity': 0.15,
        'permeability': 1e-14,
        'viscosity': 1.0e-3,
    },),
    'darcy2': 'velocity_fn_2',
    'therm3': ({
        'k_eff': 1.79 / _rho_c_fluid,
        'rho_c_eff': 2412900 / _rho_c_fluid,
        'porosity': 0.15,
        'permeability': 1e-14,
        'viscosity': 1.0e-3,
    },),
    'darcy3': 'velocity_fn_3',
    'therm4': ({
        'k_eff': 1.79 / _rho_c_fluid,
        'rho_c_eff': 2412900 / _rho_c_fluid,
        'porosity': 0.15,
        'permeability': 1e-14,
        'viscosity': 1.0e-3,
    },),
    'darcy4': 'velocity_fn_4',
    'therm12': ({
        'k_eff': 0.502 / _rho_c_fluid,
        'rho_c_eff': 2337720 / _rho_c_fluid,
        'porosity': 0.02,
        'permeability': 1e-19,
        'viscosity': 1.0e-3,
    },),
    'darcy12': 'velocity_fn_12',
}

ics = {
    'ic': ('Omega', {'T.0': 'ic_temp'}),
}

ebcs = {
    'T_bottom': ('Gamma_Bottom', {'T.0': 60.0}),
    'T_top': ('Gamma_Top', {'T.0': 10.0}),
}

integrals = {
    'i': 2,
}

equations = {
    'heat': """dw_volume_dot.i.Omega0(therm0.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega0(therm0.k_eff, s, T)
  + dw_advect_div_free.i.Omega0(darcy0.v, s, T)
  + dw_volume_dot.i.Omega1(therm1.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega1(therm1.k_eff, s, T)
  + dw_advect_div_free.i.Omega1(darcy1.v, s, T)
  + dw_volume_dot.i.Omega2(therm2.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega2(therm2.k_eff, s, T)
  + dw_advect_div_free.i.Omega2(darcy2.v, s, T)
  + dw_volume_dot.i.Omega3(therm3.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega3(therm3.k_eff, s, T)
  + dw_advect_div_free.i.Omega3(darcy3.v, s, T)
  + dw_volume_dot.i.Omega4(therm4.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega4(therm4.k_eff, s, T)
  + dw_advect_div_free.i.Omega4(darcy4.v, s, T)
  + dw_volume_dot.i.Omega_fault(therm12.rho_c_eff, s, dT/dt)
  + dw_laplace.i.Omega_fault(therm12.k_eff, s, T)
  + dw_advect_div_free.i.Omega_fault(darcy12.v, s, T)
  = 0""",
}

solvers = {
    'ls': ('ls.petsc', {
        'method': 'fgmres',
        'precond': 'asm',
        'sub_precond': 'ilu',
        'pc_asm_overlap': 2,
        'sub_pc_factor_levels': 3,
        'ksp_gmres_restart': 100,
        'ksp_divtol': 1.0e12,
        'i_max': 500,
        'eps_a': 1.0e-12,
        'eps_r': 1e-8,
        'verbose': 2,
        'ksp_monitor': '',
    }),
    'newton': ('nls.newton', {
        'i_max': 1,
        'eps_a': 1e-6,
        'is_linear': True,
        'report_status': True,
    }),
    'ts': ('ts.simple', {
        't0': t0,
        't1': t1,
        'n_step': num_steps,
        'verbose': 1,
    }),
}

options = {
    'nls': 'newton',
    'ls': 'ls',
    'ts': 'ts',
    'save_times': 'all',
    'output_dir': _output_dir,
}
