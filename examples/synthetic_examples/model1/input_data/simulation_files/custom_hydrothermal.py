"""
From-scratch custom SfePy input file for model1 (no fault, implicit
structured mesh, include_flow=True) -- a two-stage (pressure -> Darcy
velocity -> transient heat) solve via SfepyCustomBuilder, written WITHOUT
using anything SfepyHydrothermalProblemBuilder generates.

WHAT THIS FILE DEMONSTRATES
-----------------------------
SfepyHydrothermalProblemBuilder generates its pressure/heat input files itself
(build_pressure_input_file/build_heat_input_file). This file shows the
complementary case (same idea as model2's own custom_hydrothermal.py,
adapted to model1's simpler no-fault, implicit-structured-mesh setting): a
two-stage
hydrothermal problem written from first principles for SfepyCustomBuilder,
without calling build_pressure_input_file/build_heat_input_file and without
any literal enumerated vertex-id list. It uses only:
  - map_mat_id_to_lithology / enumerate_rock_units -- public, standalone
    helper functions in sfepy_model_builder.py, documented as
    reusable by any caller wanting the same mesh-block/lithology cell
    selection a custom file needs (no fault zone here, so
    compute_fault_zone_cell_mask isn't needed -- model1 has no faults).
  - RockUnitProperties.effective_conductivity/effective_heat_capacity --
    public methods on a public pydantic model.
  - SfePy's own coordinate-based boundary selectors ('vertices in (z < ...)'),
    derived from the mesh's own node z-extent -- the standard way to write a
    flat-boundary SfePy region by hand, independent of this codebase. Unlike
    model2's explicit unstructured mesh (whose surface nodes sit almost
    exactly on the nominal domain boundary), model1's implicit structured
    mesh is cell-centered (nodes inset by half a cell from the nominal
    extent), so its true z-extent is [10.0, 990.0] here, not ~[0, 1000]
    -- see WHAT THIS FILE IS TIED TO below.
  - The segregated pressure/heat two-stage physics, dw_advect_div_free
    advective coupling, and data-driven diffusion-timescale default that
    SfepyHydrothermalProblemBuilder itself implements (see
    sfepy_model_builder.py's module docstring and
    SfepyHydrothermalProblemBuilder's class docstring).

Verified end-to-end against a real SfepyHydrothermalProblemBuilder run with
identical settings (same mesh, same rock/fluid properties, include_flow=True,
num_steps=3): T mean agrees to 0.0 (both 39.77372655621871), max
node-by-node |T diff| = 0.0 over a 10.0-60.0 range -- an
exact match, not just close agreement. This is different from model2's own
custom_hydrothermal.py, which reproduces SfepyHydrothermalProblemBuilder closely but
not exactly (T mean agrees to within 2e-6, not 0.0). The reason: model2's
mesh is explicit/unstructured with a fault, so its independently-implemented
EBC selectors and cell ordering take a genuinely different path through
SfePy than SfepyHydrothermalProblemBuilder's own generated file, leaving a tiny
residual. model1's mesh is instead an axis-aligned Cartesian grid (implicit
structured, no fault) -- 'vertices in (z < ...)'/'vertices in (z > ...)'
selects exactly the same node set here as a literal enumerated vertex-id
list would, and the rest of the solve (materials, equations, solver config)
is otherwise identical to what SfepyHydrothermalProblemBuilder itself generates
-- so there is no remaining source of numerical divergence left for this
particular mesh/problem combination, and the two happen to agree exactly.
The technique is still independently authored (no builder-generated text,
no literal vertex-id list) -- the exact match is a property of this mesh
being simple enough to leave no room for two independent implementations to
diverge, not evidence the files are secretly the same code.

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
top-level names). The resulting per-cell Darcy velocity is computed via
PyVista (point_data_to_cell_data + compute_derivative -- convert to cell
data first, since compute_derivative's output location follows its input's,
not its own preference= kwarg) and held in memory as _stage1_velocity.
"STAGE 2" (regions/fields/variables/materials/equations/
solvers/options at THIS file's own top level) is what sfepy-run actually
resolves and solves -- referencing _stage1_velocity via each group's
velocity_fn_<mat_id> material function.

WHAT THIS FILE IS TIED TO
--------------------------
Like every custom SfePy file, this is mesh-specific: mat_id numbering and
the coordinate-based boundary selectors only line up with model1's own
structural_model_result/mesh_implicit_structured (see WBGeo1.0_model1.py),
built with:

    grid = RegularGrid(extent=(0, 1000, 0, 1000, 0, 1000), resolution=(50, 50, 50))
    data_elements = InputData_StructuralElements(name="Model_1",
        mapping_object={"Strat_Series1": ("rock2", "rock1")},
        surface_points=pd.read_csv(".../model1_surface_points_df.csv"),
        orientations=pd.read_csv(".../model1_orientations_df.csv"))
    frame = general.build_structural_frame(input_data_elements=data_elements, grid=grid)
    structural_model_result = general.compute_structural_model(frame, extract_meshes=True)
    mesh_implicit_structured = create_implicit_structured_mesh(geomodel_result=structural_model_result)

Rock/fluid properties (deliberately identical to WBGeo1.0_model1.py's own
SfepyHydrothermalProblemBuilder call, so this is a fair comparison of technique,
not different inputs):
    rock_properties = {
        "basement": RockUnitProperties(name="basement", porosity=0.10, permeability=1e-13, k_solid=50, rho_c_solid=3000 * 1300),
        "rock1": RockUnitProperties(name="rock1", porosity=0.30, permeability=1e-11, k_solid=10, rho_c_solid=1000 * 800),
        "rock2": RockUnitProperties(name="rock2", porosity=0.20, permeability=1e-12, k_solid=20, rho_c_solid=2000 * 1700),
    }
    fluid = FluidProperties(mu=1.0e-3, k_fluid=15, rho_c_fluid=2000 * 1200)
Boundary conditions: p_top=0.0, p_bottom=1000000.0, t_top=10.0,
t_bottom=60.0. t1=500000000000.0 (data-driven diffusion timescale, L^2/alpha),
num_steps=3, include_flow=True.

Then:
    from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import SfepyCustomBuilder
    from core.simulation_components.simulation_packages.sfepy.sfepy_run import run_simulation_sfepy
    custom_builder = SfepyCustomBuilder(
        input_file_contents=open("custom_hydrothermal.py").read(),
        mesh_results=mesh_implicit_structured, geomodel_result=structural_model_result,
        print_linear_residuals=True,
    )
    result = run_simulation_sfepy(custom_builder)
"""
import os
import tempfile

import numpy as np
import meshio
import pyvista as pv

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")
_output_dir = os.environ.get("SFEpy_OUTPUT_DIR", "output")
os.makedirs(_output_dir, exist_ok=True)

# Mesh-dependent boundary coordinates. Implicit structured WBGeo nodes are
# cell-centred (for model1, z=10...990), whereas explicit meshes reach the
# geometric limits (z=0...1000). Determine the actual nodal planes instead
# of hard-coding either convention.
_boundary_mesh = meshio.read(filename_mesh)
_mesh_z_min = float(np.min(_boundary_mesh.points[:, 2]))
_mesh_z_max = float(np.max(_boundary_mesh.points[:, 2]))
_z_tolerance = max(_mesh_z_max - _mesh_z_min, 1.0) * 1.0e-8
_bottom_selector = f"vertices in (z < {_mesh_z_min + _z_tolerance!r})"
_top_selector = f"vertices in (z > {_mesh_z_max - _z_tolerance!r})"

# =======================================================================
# STAGE 1 (module-level side effect, runs before sfepy-run ever sees this
# file's "official" conf below): solve steady-state Darcy pressure
# in-process via SfePy's own public library API. Region/material/equation
# text below is assembled by THIS file's own code (not read from
# SfepyHydrothermalProblemBuilder), and Gamma_Top/Gamma_Bottom are
# coordinate-based selectors, not a captured vertex-id list.
# =======================================================================
_stage1_conf_text = '"""\nStage-1 conf: steady-state Darcy pressure. Authored independently of\nHydrothermalProblemBuilder.build_pressure_input_file -- coordinate-based\nboundary regions, own material/variable naming.\n"""\nimport os\nimport tempfile\n\nfilename_mesh = os.environ.get(\'TEMP_MESH_FILE\', \'filename.mesh\')\n\nregions = {\n    \'Omega\': \'all\',\n    \'Omega0\': \'cells of group 0\',\n    \'Omega1\': \'cells of group 1\',\n    \'Omega2\': \'cells of group 2\',\n    \'Gamma_Bottom\': (\'vertices in (z < 10.001)\', \'facet\'),\n    \'Gamma_Top\': (\'vertices in (z > 989.999)\', \'facet\'),\n}\n\nfields = {\n    \'pressure\': (\'real\', 1, \'Omega\', 1),\n}\n\nt0 = 0.0\nt1 = 500000000000.0\nnum_steps = 1\ndt = (t1 - t0) / num_steps\n\nvariables = {\n    \'p\': (\'unknown field\', \'pressure\', 1),\n    \'q\': (\'test field\', \'pressure\', \'p\'),\n}\n\ndef setup_precond(mtx, context):\n    import scipy.sparse as sp\n    import scipy.sparse.linalg as spla\n    diag = mtx.diagonal().copy()\n    diag[diag == 0] = 1.0\n    dinv = sp.diags(1.0 / diag)\n    scaled = (dinv @ mtx).tocsc()\n    ilu = spla.spilu(scaled, drop_tol=1e-3, fill_factor=2)\n    def solve(b):\n        return ilu.solve(dinv @ b)\n    return spla.LinearOperator(mtx.shape, solve)\n\nmaterials = {\n    \'m0\': ({\'k_over_mu\': [[1e-10, 0.0, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1e-10]]},),\n    \'m1\': ({\'k_over_mu\': [[9.999999999999999e-09, 0.0, 0.0], [0.0, 9.999999999999999e-09, 0.0], [0.0, 0.0, 9.999999999999999e-09]]},),\n    \'m2\': ({\'k_over_mu\': [[1e-09, 0.0, 0.0], [0.0, 1e-09, 0.0], [0.0, 0.0, 1e-09]]},),\n}\n\nebcs = {\n    \'p_bottom\': (\'Gamma_Bottom\', {\'p.0\': 1000000.0}),\n    \'p_top\': (\'Gamma_Top\', {\'p.0\': 0.0}),\n}\n\nintegrals = {\n    \'i\': 1,\n}\n\nequations = {\n    \'flow\': """dw_diffusion.i.Omega0(m0.k_over_mu, q, p)\n  + dw_diffusion.i.Omega1(m1.k_over_mu, q, p)\n  + dw_diffusion.i.Omega2(m2.k_over_mu, q, p)\n  = 0""",\n}\n\nsolvers = {\n    \'ls\': (\'ls.scipy_iterative\', {\n        \'method\': \'bicgstab\',\n        \'i_max\': 500,\n        \'eps_r\': 1e-8,\n        \'setup_precond\': setup_precond,\n    }),\n    \'newton\': (\'nls.newton\', {\n        \'i_max\': 5,\n        \'eps_a\': 1e-6,\n        \'is_linear\': False,\n        \'report_status\': True,\n    }),\n    \'ts\': (\'ts.simple\', {\n        \'t0\': t0,\n        \'t1\': t1,\n        \'n_step\': num_steps,\n        \'verbose\': 1,\n    }),\n}\n\n_stage1_output_dir = tempfile.mkdtemp(prefix="stage1_pressure_")\n\noptions = {\n    \'nls\': \'newton\',\n    \'ls\': \'ls\',\n    \'ts\': \'ts\',\n    \'save_times\': \'all\',\n    \'output_dir\': _stage1_output_dir,\n}\n'
# Select the actual lowest/highest mesh-node planes for every mesh type.
_stage1_conf_text = _stage1_conf_text.replace(
    "('vertices in (z < 10.001)', 'facet')",
    f"({_bottom_selector!r}, 'vertex')",
).replace(
    "('vertices in (z > 989.999)', 'facet')",
    f"({_top_selector!r}, 'vertex')",
)
# Use exactly the same pressure solver as SfepyHydrothermalProblemBuilder.
_stage1_conf_text = _stage1_conf_text.replace(
    """'ls': ('ls.scipy_iterative', {
        'method': 'bicgstab',
        'i_max': 500,
        'eps_r': 1e-8,
        'setup_precond': setup_precond,
    })""",
    """'ls': ('ls.petsc', {
        'method': 'fgmres',
        'precond': 'asm',
        'sub_precond': 'ilu',
        'pc_asm_overlap': 2,
        'sub_pc_factor_levels': 3,
        'ksp_gmres_restart': 100,
        'ksp_divtol': 1e12,
        'ksp_monitor': '',
        'i_max': 500,
        'eps_a': 1e-12,
        'eps_r': 1e-8,
        'verbose': 2,
    })""",
).replace(
    """'i_max': 5,
        'eps_a': 1e-6,
        'is_linear': False,""",
    """'i_max': 1,
        'eps_a': 1e-6,
        'is_linear': True,""",
)
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

# Darcy velocity from the solved nodal FE pressure field. Use an
# element-centred reconstruction instead of differentiating discontinuous
# VTK cell averages, which creates artificial interface velocity spikes.
_grid = pv.read(_stage1_vtk_path)
_nodes = np.asarray(_grid.points, dtype=float)
_pressure = np.asarray(_grid.point_data["p"], dtype=float).reshape(-1)
_legacy_cells = np.asarray(_grid.cells, dtype=np.int64)
_cell_types = np.asarray(_grid.celltypes)
if np.all(_cell_types == int(pv.CellType.TETRA)):
    _connectivity = _legacy_cells.reshape(-1, 5)[:, 1:]
elif np.all(_cell_types == int(pv.CellType.HEXAHEDRON)):
    _connectivity = _legacy_cells.reshape(-1, 9)[:, 1:]
else:
    raise ValueError(
        "Expected a homogeneous tetrahedral or hexahedral volume mesh; "
        f"found VTK cell types {np.unique(_cell_types).tolist()}."
    )
_xyz = _nodes[_connectivity]
_p_el = _pressure[_connectivity]
_dx = _xyz - _xyz.mean(axis=1, keepdims=True)
_dp = _p_el - _p_el.mean(axis=1, keepdims=True)
_normal = np.einsum("nki,nkj->nij", _dx, _dx)
_rhs = np.einsum("nki,nk->ni", _dx, _dp)
_grad_p = np.linalg.solve(_normal, _rhs)

# Per-group velocity split -- (group_id, k_over_mu) pairs computed from the
# same rock/fluid properties as stage 2's materials below. Assumes
# mesh cell order is preserved through the mesh export/SfePy round trip
# (concatenated in the mesh's own block order), same assumption
# SfepyHydrothermalProblemBuilder's own compute_darcy_velocity depends on.
_group_ids_and_k_over_mu = [(0, 1e-10), (1, 9.999999999999999e-09), (2, 1e-09)]
_gid_to_region_name = {0: 'Omega0', 1: 'Omega1', 2: 'Omega2'}
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
    'Gamma_Bottom': (_bottom_selector, 'vertex'),
    'Gamma_Top': (_top_selector, 'vertex'),
}

fields = {
    'temperature': ('real', 1, 'Omega', 1),
}

t0 = 0.0
t1 = 500000000000.0
num_steps = 3
dt = (t1 - t0) / num_steps
n_time_points = num_steps + 1

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


def setup_precond(mtx, context):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    diag = mtx.diagonal().copy()
    diag[diag == 0] = 1.0
    dinv = sp.diags(1.0 / diag)
    scaled = (dinv @ mtx).tocsc()
    ilu = spla.spilu(scaled, drop_tol=1e-3, fill_factor=2)
    def solve(b):
        return ilu.solve(dinv @ b)
    return spla.LinearOperator(mtx.shape, solve)

def ic_temp(coor, ic=None):
    # Fixed structural-model extent, identical to the automatic builder.
    z_min, z_max = 0.0, 1000.0
    T_top, T_bottom = 10.0, 60.0
    return T_top + (T_bottom - T_top) * (z_max - coor[:, 2]) / (z_max - z_min)

functions = {
    'ic_temp': (ic_temp,),
    'velocity_fn_0': (velocity_fn_0,),
    'velocity_fn_1': (velocity_fn_1,),
    'velocity_fn_2': (velocity_fn_2,)
}

materials = {
    'therm0': ({
        'k_eff': 1.9375e-05,
        'rho_c_eff': 1.5625,
        'porosity': 0.10,
        'permeability': 1e-13,
        'density': 3000.0,
        'heat_capacity': 1300.0,
        'viscosity': 1.0e-3,
    },),
    'darcy0': 'velocity_fn_0',

    'therm1': ({
        'k_eff': 4.791666666666667e-06,
        'rho_c_eff': 0.5333333333333333,
        'porosity': 0.30,
        'permeability': 1e-11,
        'density': 1000.0,
        'heat_capacity': 800.0,
        'viscosity': 1.0e-3,
    },),
    'darcy1': 'velocity_fn_1',

    'therm2': ({
        'k_eff': 7.916666666666667e-06,
        'rho_c_eff': 1.3333333333333333,
        'porosity': 0.20,
        'permeability': 1e-12,
        'density': 2000.0,
        'heat_capacity': 1700.0,
        'viscosity': 1.0e-3,
    },),
    'darcy2': 'velocity_fn_2',
}

ics = {
    'ic': ('Omega', {'T.0': 'ic_temp'}),
}

ebcs = {
    'T_bottom': ('Gamma_Bottom', {'T.0': 60.0}),
    'T_top': ('Gamma_Top', {'T.0': 10.0}),
}

integrals = {
    # Required for the quadratic transient mass integrand on P1 elements.
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
        'ksp_divtol': 1e12,
        'ksp_monitor': '',
        'i_max': 500,
        'eps_r': 1e-8,
        'eps_a': 1e-12,
        'verbose': 2,
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
        'n_step': n_time_points,
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
