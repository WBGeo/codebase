"""
Transient heat conduction in SfePy with multiple blocks.
- Blocks from meshio mat_id (Gmsh physical groups)
- Non-constant IC: T = grad*(z - z0) + T0
- Top/bottom Dirichlet BCs
- Transient solver
"""

import numpy as nm
from sfepy.discrete.fem import FEDomain, Field
from sfepy.base.base import Struct
from sfepy import data_dir
import meshio
import os
from sfepy.discrete.fem import Mesh, FEDomain, Field



# -------------------------------
# Mesh
# -------------------------------
import os

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")

# -----------------------------
# Regions
# -----------------------------
regions = {
    'Omega': 'all',
    'Omega1': 'cells of group 0',
    'Omega2': 'cells of group 1',
    'Omega3': 'cells of group 2',
    'Omega4': 'cells of group 3',
    'Omega5': 'cells of group 4',
    'Gamma_Bottom': ('vertices in (z < 50)', 'facet'),
    'Gamma_Top':    ('vertices in (z > 980)', 'facet'),
}


# -----------------------------
# Time stepping
# -----------------------------
t0 = 0.0
t1 = 1.0
num_steps = 1
dt = (t1 - t0)/num_steps



# -----------------------------
# Materials
# -----------------------------
k_blocks = [16, 1, 27, 2, 31]  # same as PETSc
materials = {
    'mat': ({'val': {
        'Omega1': 16.0,
        'Omega2': 1.0,
        'Omega3': 27.0,
        'Omega4': 2.0,
        'Omega5': 31.0,
    }},),
}
# -----------------------------
# Field and variables
# -----------------------------
fields = {
    'temperature': ('real', 1, 'Omega', 1),
}

variables = {
    'T': ('unknown field', 'temperature'),
    's': ('test field', 'temperature', 'T'),
}

# -----------------------------
# Boundary conditions (Dirichlet)
# -----------------------------
ebcs = {
    'T_top': ('Gamma_Top', {'T.0': 10.0}),
    'T_bottom': ('Gamma_Bottom', {'T.0': 40.0}),
}

# -----------------------------
# Initial condition
# -----------------------------
def get_ic(coor, ic):
    grad, z0, T0 = -0.03, 1000.0, 10.0
    return grad * (coor[:,2] - z0) + T0

functions = {
    'get_ic': (get_ic,),
}

ics = {
    'ic': ('Omega', {'T.0': 'get_ic'}),
}

# -----------------------------
# Equations (one per block)
# -----------------------------

equations = {
    'Temperature': """
        dw_laplace.i.Omega(mat.val, s, T)
      = 0
    """
}

# -----------------------------
# Integral
# -----------------------------
integrals = {'i': 1}

# -------------------------------------------------
# Solvers
# -------------------------------------------------
solvers = {
    'ls': ('ls.petsc', {
        'method' : 'bcgsl',
        'precond': 'bjacobi',
        'sub_precond': 'ilu',
        'eps_a' : 0.0,
        'eps_r' : 1e-10,
        'eps_d' : 1e10,
        'i_max' : 200,
    }),
    'newton': ('nls.newton', {
        'i_max': 1,
        'eps_a': 1e-10,
        'is_linear': True,
    }),
    'ts': ('ts.simple', {
        't0': t0,
        't1': t1,
        'n_step': num_steps,
        'verbose': 1,
    }),
}

# -------------------------------------------------
# Output folder
# -------------------------------------------------
output_dir = os.environ.get("SFEpy_OUTPUT_DIR", "output")
os.makedirs(output_dir, exist_ok=True)
print(output_dir,'output_dir')
# -------------------------------------------------
# Options
# -------------------------------------------------
options = {
    'nls': 'newton',
    'ls': 'ls',
    'ts': 'ts',
    'save_times': 'all',
        'output_dir': output_dir,

}


print('output_dir:', output_dir)

