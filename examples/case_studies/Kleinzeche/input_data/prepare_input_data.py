"""
Kleinzeche — prepare input data for structural models
======================================================
Builds surface_points, orientations, and fault data for both the full and
simplified (simple) Kleinzeche models, then writes them as CSV files to
this directory (input_data/).

Run once (or after re-picking contacts) to regenerate the CSVs.
The generated files are committed to the repository so that the model
scripts work without running this script first.

Output files
------------
  fault1_surface_points.csv   — Fault-1 borehole intercepts (shared)
  fault1_orientations.csv     — Fault-1 plane normal via SVD (shared)
  full_surface_points.csv     — Full model: all formations + proxy Clay/Coal
  full_orientations.csv       — Full model: true-dip analytical orientations
  simple_surface_points.csv   — Simple model: Profile A only + 2.5D extension
  simple_orientations.csv     — Simple model: upward orientations at X extents
"""

import os
import sys

import numpy as np
import pandas as pd

# ── Path setup — shared modules are siblings in the same input_data/ directory ─
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Kleinzeche_shared import BOREHOLES, Z_MIN, CSV_PATH
from Kleinzeche_section_picking import picks_to_dataframes, FORMATIONS_A

# ══════════════════════════════════════════════════════════════════════════════
# Shared: Fault-1  (identical for both models)
# ══════════════════════════════════════════════════════════════════════════════

_csv = pd.read_csv(CSV_PATH)
fault1_sp = _csv[_csv['formation'] == 'Fault-1'].copy().reset_index(drop=True)

# Fit a plane to the borehole intercepts via SVD → upward-pointing normal
_pts      = fault1_sp[['X', 'Y', 'Z']].values
_centroid = _pts.mean(axis=0)
_, _, _Vt = np.linalg.svd(_pts - _centroid)
_fn       = _Vt[-1]
if _fn[2] < 0:
    _fn = -_fn
_fn /= np.linalg.norm(_fn)

fault1_ori = pd.DataFrame([{
    'X': float(_centroid[0]), 'Y': float(_centroid[1]), 'Z': float(_centroid[2]),
    'G_x': float(_fn[0]), 'G_y': float(_fn[1]), 'G_z': float(_fn[2]),
    'formation': 'Fault-1',
}])

fault1_sp.to_csv(os.path.join(_HERE, 'fault1_surface_points.csv'), index=False)
fault1_ori.to_csv(os.path.join(_HERE, 'fault1_orientations.csv'), index=False)
print(f"Fault-1 : {len(fault1_sp)} surface pts  "
      f"normal G=({_fn[0]:.3f}, {_fn[1]:.3f}, {_fn[2]:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# Full model
# ══════════════════════════════════════════════════════════════════════════════

# ── Surface points ─────────────────────────────────────────────────────────────
_sp_all, _ = picks_to_dataframes()

# True dip derived from apparent dip on Profile B
APPARENT_DIP_B = 65.0
_s  = BOREHOLES['MO1'][:2] - BOREHOLES['MP1'][:2];  _s  /= np.linalg.norm(_s)
_pB = BOREHOLES['O3'][:2]  - BOREHOLES['O4'][:2];   _pB /= np.linalg.norm(_pB)
_beta     = np.arccos(np.clip(abs(np.dot(_s, _pB)), 0.0, 1.0))
_true_dip = np.arctan(np.tan(np.radians(APPARENT_DIP_B)) / np.sin(_beta))
print(f"Full    : beta={np.degrees(_beta):.1f} deg  true dip={np.degrees(_true_dip):.1f} deg")

# Proxy Clay/Coal points (hanging-wall, Profile A domain)
_dz_clay1 = 4 / np.cos(_true_dip)
_dz_coal1 = 3 / np.cos(_true_dip)
_dz_clay2 = 4 / np.cos(_true_dip)
_proxy_base   = {'X': 380048.827256, 'Y': 5700894.0}
_z_sand3_base = 72.331387

full_sp = pd.concat([_sp_all, pd.DataFrame([
    {**_proxy_base, 'Z': _z_sand3_base - _dz_clay1,                         'formation': 'Clay-1'},
    {**_proxy_base, 'Z': _z_sand3_base - _dz_clay1 - _dz_coal1,             'formation': 'Coal-1'},
    {**_proxy_base, 'Z': _z_sand3_base - _dz_clay1 - _dz_coal1 - _dz_clay2, 'formation': 'Clay-2'},
])], ignore_index=True)

_Z_MIN_FULL = Z_MIN - 10   # 30 m — extended to capture Clay/Coal in hanging wall
full_sp = full_sp[full_sp['Z'] >= _Z_MIN_FULL].reset_index(drop=True)

full_sp.to_csv(os.path.join(_HERE, 'full_surface_points.csv'), index=False)
print(f"Full    : {len(full_sp)} surface pts  (Z_MIN = {_Z_MIN_FULL} m)")

# ── Orientations ───────────────────────────────────────────────────────────────
_d = np.array([-_s[1], _s[0], 0.0])
if np.dot(_d[:2], BOREHOLES['O4'][:2] - BOREHOLES['O3'][:2]) < 0:
    _d = -_d
_GX = float(np.sin(_true_dip) * _d[0])
_GY = float(np.sin(_true_dip) * _d[1])
_GZ = float(np.cos(_true_dip))

_available_full = set(full_sp['formation'].unique())
_cx = (380040 + 380132) / 2.0    # 380086 — model centre Easting
_cy = (5700884 + 5700938) / 2.0  # 5700911 — model centre Northing
_x_deep    = 380040.0
_x_shallow = _cx + 20.0

def _mz_full(fm):
    rows = full_sp[full_sp['formation'] == fm]['Z']
    return float(rows.mean()) if len(rows) else float((_Z_MIN_FULL + 130) / 2)

def _ori_full(x, y, fm, gx=None, gy=None, gz=None):
    return {'X': x, 'Y': y, 'Z': _mz_full(fm),
            'G_x': _GX if gx is None else gx,
            'G_y': _GY if gy is None else gy,
            'G_z': _GZ if gz is None else gz,
            'formation': fm}

_ori_rows = []
for _fm in ['Gravel-1', 'Loose-1']:
    if _fm in _available_full:
        _ori_rows.append(_ori_full(_cx, _cy, _fm, gx=0.0, gy=0.0, gz=1.0))
for _fm in ['Silt-1', 'Sand-1', 'Silt-2', 'Sand-2', 'Silt-3', 'Sand-3']:
    if _fm in _available_full:
        _ori_rows.append(_ori_full(_x_deep,    _cy, _fm))
        _ori_rows.append(_ori_full(_x_shallow, _cy, _fm))
for _fm in ['Clay-1', 'Coal-1', 'Clay-2']:
    if _fm in _available_full:
        _ori_rows.append(_ori_full(_cx, _cy, _fm))

full_ori = pd.DataFrame(_ori_rows)
full_ori.to_csv(os.path.join(_HERE, 'full_orientations.csv'), index=False)
print(f"Full    : {len(full_ori)} orientations")

# ══════════════════════════════════════════════════════════════════════════════
# Simple (2.5D) model
# ══════════════════════════════════════════════════════════════════════════════

# ── Surface points — Profile A only, duplicated at Y model edges ──────────────
_A_FMSET = set(FORMATIONS_A)
simple_sp = _sp_all[_sp_all['formation'].isin(_A_FMSET)].reset_index(drop=True)

_Y_MIN_GRID = round((BOREHOLES['MP1'][1] + BOREHOLES['MO1'][1]) / 2 - 25)  # 5700883
_Y_MAX_GRID = round((BOREHOLES['MP1'][1] + BOREHOLES['MO1'][1]) / 2 + 25)  # 5700933
_Y_NEAR = float(_Y_MIN_GRID + 2)
_Y_FAR  = float(_Y_MAX_GRID - 2)

_near = simple_sp.copy(); _near['Y'] = _Y_NEAR
_far  = simple_sp.copy(); _far['Y']  = _Y_FAR
simple_sp = pd.concat([simple_sp, _near, _far], ignore_index=True)

simple_sp.to_csv(os.path.join(_HERE, 'simple_surface_points.csv'), index=False)
print(f"Simple  : {len(simple_sp)} surface pts")

# ── Orientations — upward (0,0,1) at X_MIN and X_MAX per formation ────────────
_X_MIN    = round((BOREHOLES['MP1'][0] + BOREHOLES['MO1'][0]) / 2 - 25)  # 380046
_X_MAX    = round((BOREHOLES['MP1'][0] + BOREHOLES['MO1'][0]) / 2 + 25)  # 380096
_Y_CENTER = float(_Y_MIN_GRID + _Y_MAX_GRID) / 2

_available_simple = set(simple_sp['formation'].unique())

def _mz_simple(fm):
    rows = simple_sp[simple_sp['formation'] == fm]['Z']
    return float(rows.mean()) if len(rows) else float((Z_MIN + 130) / 2)

_ori_rows = []
for _fm in FORMATIONS_A:
    if _fm not in _available_simple:
        continue
    _z = _mz_simple(_fm)
    for _x in [float(_X_MIN), float(_X_MAX)]:
        _ori_rows.append({'X': _x, 'Y': _Y_CENTER, 'Z': _z,
                          'G_x': 0.0, 'G_y': 0.0, 'G_z': 1.0,
                          'formation': _fm})

simple_ori = pd.DataFrame(_ori_rows)
simple_ori.to_csv(os.path.join(_HERE, 'simple_orientations.csv'), index=False)
print(f"Simple  : {len(simple_ori)} orientations")

print("\nAll CSV files written to input_data/")
