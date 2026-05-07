"""
GOLEM-style fully coupled HT problem in SfePy
3 blocks, transient heat + steady Darcy flow
"""
import numpy as np
import os
import meshio

# -------------------------------
# Mesh
# -------------------------------
import os

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")
#----------------
regions = {
    'Omega': 'all',
    'Omega0': 'cells of group 0',
    'Omega1': 'cells of group 1',
    'Gamma_Bottom': ('vertices in (z < 50)', 'facet'),
    'Gamma_Top': ('vertices in (z > 980)', 'facet'),
}

# -------------------------------------------------
# Fields
# -------------------------------------------------
fields = {
    'temperature': ('real', 1, 'Omega', 1),
    'pressure': ('real', 1, 'Omega', 1),
}

# -------------------------------------------------
# Time stepping
# -------------------------------------------------
t0 = 0.0
t1 = 700.0
num_steps = 1
dt = (t1 - t0)/num_steps

# -------------------------------------------------
# Variables
# -------------------------------------------------
variables = {
    'T': ('unknown field', 'temperature', 0, 1),
    's': ('test field', 'temperature', 'T'),
    'p': ('unknown field', 'pressure', 1),
    'q': ('test field', 'pressure', 'p'),
}

# -------------------------------------------------
# Material properties
# -------------------------------------------------
dim = 3
materials = {
    'm1': ({
        'rho_c_solid': 1000 * 800,
        'rho_c_fluid': 1000 * 1000,
        'k_s': 10,
        'k_f': 27,
        'perm': ((1e-11) * np.eye(dim)).tolist(),
        'mu': 1.0e-3,
        'perm_over_mu': ((1e-11)/1.0e-3 * np.eye(dim)).tolist(),
    },),
    'm0': ({
        'rho_c_solid': 2000 * 1700,
        'rho_c_fluid': 1500 * 1500,
        'k_s': 20,
        'k_f': 10,
        'perm': ((1e-12) * np.eye(dim)).tolist(),
        'mu': 1.0e-3,
        'perm_over_mu': ((1e-12)/1.0e-3 * np.eye(dim)).tolist(),
    },),
}

# -------------------------------------------------
# Initial condition
# -------------------------------------------------
def ic_temp(coor, ic=None):
    grad, z0, T0 = -0.05, 1000.0, 10.0
    return grad * (coor[:, 2] - z0) + T0

functions = {
    'ic_temp': (ic_temp,),
}

ics = {
    'ic': ('Omega', {'T.0': 'ic_temp'}),
}

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
ebcs = {
    'p_bottom': ('Gamma_Bottom', {'p.0': 100e4}),
    'p_top': ('Gamma_Top', {'p.0': 0.0}),
    'T_bottom': ('Gamma_Bottom', {'T.0': 60.0}),
    'T_top': ('Gamma_Top', {'T.0': 10.0}),
}

# -------------------------------------------------
# Integrals
# -------------------------------------------------
integrals = {
    'i': 1,
}

# -------------------------------------------------
# Equations (GOLEM-style)
# -------------------------------------------------
equations = {

    # Steady Darcy flow
    'flow':
    """
    dw_diffusion.i.Omega0(m0.perm_over_mu, q, p)
  + dw_diffusion.i.Omega1(m1.perm_over_mu, q, p)
  = 0
    """,

    # Transient heat equation
    'heat':
    """
    dw_volume_dot.i.Omega0(m0.rho_c_solid, s, dT/dt)
  + dw_volume_dot.i.Omega0(m0.rho_c_fluid, s, dT/dt)
  + dw_laplace.i.Omega0(m0.k_s, s, T)
  + dw_laplace.i.Omega0(m0.k_f, s, T)

  + dw_volume_dot.i.Omega1(m1.rho_c_solid, s, dT/dt)
  + dw_volume_dot.i.Omega1(m1.rho_c_fluid, s, dT/dt)
  + dw_laplace.i.Omega1(m1.k_s, s, T)
  + dw_laplace.i.Omega1(m1.k_f, s, T)

  = 0
    """
}

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

