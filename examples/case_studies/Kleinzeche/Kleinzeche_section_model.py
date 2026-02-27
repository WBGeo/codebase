"""
Kleinzeche — Structural model from cross-section picks + borehole fault data
=============================================================================
Builds a stratigraphic model for the Kleinzeche site using surface points
and orientations picked interactively from Profile A and Profile B.
Fault-1 surface points come from df1_filtered.csv; the fault orientation is
derived automatically by fitting a plane to those points.

Prerequisites
-------------
Run Kleinzeche_section_picking.py first to populate section_picks.json.
Already-picked formations are loaded automatically on each run.

Coordinate system
-----------------
UTM 32N, WGS84 — same as Kleinzeche_input_data.py.
Both section picks and df1_filtered.csv are in UTM; no transformation needed.
  Easting  X: 380040 – 380132
  Northing Y: 5700884 – 5700938
  Elevation Z: 40 – 130 m a.s.l.
"""

import os
import sys

import numpy as np
import pandas as pd

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D,
    plot_structural_model_3D,
    plot_fault_model_2D,
    plot_fault_model_3D,
)
from core.structural_modeling_components import general, general_faults

# ── Local Kleinzeche imports ──────────────────────────────────────────────────
try:
    _SECTION_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running as IPython cells (e.g. PyCharm console) — __file__ not defined
    _SECTION_DIR = os.path.join(os.getcwd(), "examples/case_studies/Kleinzeche")
if _SECTION_DIR not in sys.path:
    sys.path.insert(0, _SECTION_DIR)

from Kleinzeche_shared import (
    BOREHOLES, BH_COLORS, COLLARS, Z_MIN_A, Z_MIN_B, Z_MIN,
    FORMATION_COLORS, FAULT_FORMATIONS, FAULT_COLORS, TUBE_RADIUS,
    PATH_A, PATH_B, CSV_PATH, MARKER_CACHE,
    find_data_bbox, georeference_section, make_section_mesh,
)
from Kleinzeche_section_picking import picks_to_dataframes

# ── Dip configuration ─────────────────────────────────────────────────────────
# Profile A is assumed parallel to strike; apparent dip is read on Profile B.
APPARENT_DIP_B = 60.0  # apparent dip (°) as seen on Profile B

# Strike direction = Profile A direction (MP1 → MO1)
_s = BOREHOLES['MO1'][:2] - BOREHOLES['MP1'][:2]
_s = _s / np.linalg.norm(_s)

# Angle β between Profile B and the strike
_pB = BOREHOLES['O3'][:2] - BOREHOLES['O4'][:2]
_pB = _pB / np.linalg.norm(_pB)
_beta = np.arccos(np.clip(abs(np.dot(_s, _pB)), 0.0, 1.0))

# True dip: tan(δ) = tan(apparent) / sin(β)
_true_dip = np.arctan(np.tan(np.radians(APPARENT_DIP_B)) / np.sin(_beta))
print(f"Profile angle β = {np.degrees(_beta):.1f}°  →  true dip = {np.degrees(_true_dip):.1f}°")

# Dip direction: 90° CCW from strike, sign-checked against O3→O4 (deep side)
_d = np.array([-_s[1], _s[0], 0.0])
if np.dot(_d[:2], BOREHOLES['O4'][:2] - BOREHOLES['O3'][:2]) < 0:
    _d = -_d

# Upward-normal components shared by all dipping units
_GX = float(np.sin(_true_dip) * _d[0])
_GY = float(np.sin(_true_dip) * _d[1])
_GZ = float(np.cos(_true_dip))

# ── Stratigraphic picks (UTM from picks_to_dataframes) ───────────────────────
# Orientations are computed analytically below; discard those from the picker.
surface_points, _ = picks_to_dataframes()

surface_points = surface_points[surface_points['Z'] >= Z_MIN].reset_index(drop=True)

print(f"Surface points : {len(surface_points)} rows")

# ── Fault-1 data from borehole CSV (UTM) ─────────────────────────────────────
_csv = pd.read_csv(CSV_PATH)
fault_surface_points = _csv[_csv['formation'] == 'Fault-1'].copy().reset_index(drop=True)

# Derive fault orientation: fit a plane via SVD; last right singular vector = normal
_pts = fault_surface_points[['X', 'Y', 'Z']].values
_centroid = _pts.mean(axis=0)
_, _, _Vt = np.linalg.svd(_pts - _centroid)
_normal = _Vt[-1]
if _normal[2] < 0:
    _normal = -_normal
_normal /= np.linalg.norm(_normal)

fault_orientations = pd.DataFrame([{
    'X': float(_centroid[0]), 'Y': float(_centroid[1]), 'Z': float(_centroid[2]),
    'G_x': float(_normal[0]), 'G_y': float(_normal[1]), 'G_z': float(_normal[2]),
    'formation': 'Fault-1',
}])

print(f"\nFault-1 surface points:\n{fault_surface_points.to_string()}")
print(f"Fault-1 derived normal : G=({_normal[0]:.3f}, {_normal[1]:.3f}, {_normal[2]:.3f})")

# ── Grid (UTM extent) ─────────────────────────────────────────────────────────
_grid = RegularGrid(
    extent=(380040, 380132, 5700884, 5700938, Z_MIN, 130),
    resolution=(92, 54, 90),
)

# ── Fault frame ───────────────────────────────────────────────────────────────
data_faults = InputData_FaultElements(
    name='Kleinzeche',
    fault_surface_points=fault_surface_points,
    fault_orientations=fault_orientations,
    fault_names=['Fault-1'],
)

fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=_grid,
)

fault_frame.detailed_report()

#%%

plot_fault_model_2D(fault_frame)
plot_fault_model_3D(fault_frame)

#%%

fault_model_result = general_faults.compute_fault_domains(fault_frame)

plot_fault_model_2D(fault_model_result.fault_frame)
plot_fault_model_3D(fault_model_result.fault_frame)

# ── Formation mapping (youngest → oldest) ────────────────────────────────────
_available = set(surface_points['formation'].unique())
print(f"\nFormations with data: {sorted(_available)}")

_ALL_SERIES = [
    ("Strat_Series3", ("Gravel-1",)),
    ("Strat_Series2", ("Loose-1",)),
    ("Strat_Series1", ("Silt-1", "Sand-1", "Silt-2", "Sand-2", "Silt-3", "Sand-3")),
    # Clay/Coal only picked in Profile B → only present in the shallow fault domain.
    # Fault-1 activity applies to all groups older than Strat_Series1, so this group
    # is also faulted.  The codebase handles missing iso-surfaces gracefully (no mesh
    # extracted for domains where the formation is absent).
    ("Strat_Series0", ("Clay-1", "Coal-1", "Clay-2")),
]

mapping_object = {}
for _series, _fms in _ALL_SERIES:
    _present = tuple(f for f in _fms if f in _available)
    if _present:
        mapping_object[_series] = _present

# ── Analytical orientations ───────────────────────────────────────────────────
# Model centre and two X positions that fall on opposite sides of Fault-1.
_cx = (380040 + 380132) / 2.0    # 380086  — model centre Easting
_cy = (5700884 + 5700938) / 2.0  # 5700911 — model centre Northing
_x_deep    = 380040.0            # model left edge — safely inside the deep/hanging-wall domain
_x_shallow = _cx + 20.0          # toward O3 (shallow / foot-wall side)

def _mean_z(fm):
    """Representative Z for an orientation point: mean of picked surface points."""
    rows = surface_points[surface_points['formation'] == fm]['Z']
    return float(rows.mean()) if len(rows) else float((Z_MIN + 130) / 2)

def _ori(x, y, fm, gx=None, gy=None, gz=None):
    return {
        'X': x, 'Y': y, 'Z': _mean_z(fm),
        'G_x': _GX if gx is None else gx,
        'G_y': _GY if gy is None else gy,
        'G_z': _GZ if gz is None else gz,
        'formation': fm,
    }

_ori_rows = []

# Flat units — single upward orientation at model centre
for _fm in ['Gravel-1', 'Loose-1']:
    if _fm in _available:
        _ori_rows.append(_ori(_cx, _cy, _fm, gx=0.0, gy=0.0, gz=1.0))

# Faulted dipping units — one orientation per fault domain
for _fm in ['Silt-1', 'Sand-1', 'Silt-2', 'Sand-2', 'Silt-3', 'Sand-3']:
    if _fm in _available:
        _ori_rows.append(_ori(_x_deep,    _cy, _fm))
        _ori_rows.append(_ori(_x_shallow, _cy, _fm))

# Unfaulted dipping units — single orientation at model centre
for _fm in ['Clay-1', 'Coal-1', 'Clay-2']:
    if _fm in _available:
        _ori_rows.append(_ori(_cx, _cy, _fm))

orientations = pd.DataFrame(_ori_rows)
print(f"Orientations   : {len(orientations)} rows")

# ── Structural input data ─────────────────────────────────────────────────────
grid = RegularGrid(
    extent=(380040, 380132, 5700884, 5700938, Z_MIN, 130),
    resolution=(92, 54, 90),
)

data_elements = InputData_StructuralElements(
    name="Kleinzeche_sections",
    mapping_object=mapping_object,
    surface_points=surface_points,
    orientations=orientations,
)

# ── Build structural frame ────────────────────────────────────────────────────
frame = general.build_structural_frame(
    input_data_elements=data_elements,
    grid=grid,
    fault_model_results=fault_model_result,
)

# ── Formation colors ──────────────────────────────────────────────────────────
for _series, _fms in _ALL_SERIES:
    if _series not in mapping_object:
        continue
    for _fm in mapping_object[_series]:
        if _fm in FORMATION_COLORS:
            frame[_series][_fm].set_color(FORMATION_COLORS[_fm])

# Fault-1 displaces Strat_Series1 (Silt/Sand/Clay layers) only.
# Gravel-1 and Loose-1 are shallow post-fault sediments — treated as unfaulted.
frame.set_fault_activity_by_group(fault_name='Fault-1', group_name='Strat_Series1')

frame.detailed_report()

#%%

# ── Interpolation methods ─────────────────────────────────────────────────────
for _series in mapping_object:
    frame[_series].set_interpolation_method("Universal Co-Kriging")

# ── Plot input data ───────────────────────────────────────────────────────────
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# ── Compute structural model ──────────────────────────────────────────────────
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# ── Visualize results ─────────────────────────────────────────────────────────
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(
    structural_model_result.structural_frame,
    show_surface_meshes=True,
    show_orientations=True,
)

#%%

# ── Combined: model + cross sections + borehole trajectories ─────────────────
import json
import pyvista as pv
from PIL import Image as PILImage

with open(MARKER_CACHE) as _f:
    _markers = {k: tuple(v) for k, v in json.load(_f).items()}

_bbox_A    = find_data_bbox(PATH_A)
_bbox_B    = find_data_bbox(PATH_B)
_corners_A = georeference_section(BOREHOLES['MP1'], BOREHOLES['MO1'],
                                   _markers['MP1'], _markers['MO1'], _bbox_A, Z_MIN_A)
_corners_B = georeference_section(BOREHOLES['O4'],  BOREHOLES['O3'],
                                   _markers['O4'],  _markers['O3'],  _bbox_B, Z_MIN_B)

_tex_A = pv.read_texture(PATH_A)
_tex_B = pv.read_texture(PATH_B)

# Get the model plotter without displaying yet
p = plot_structural_model_3D(
    structural_model_result.structural_frame,
    show_surface_meshes=True,
    show_orientations=False,
    show=False,
)

# ── Cross sections (toggleable) ───────────────────────────────────────────────
_section_actors = [
    p.add_mesh(make_section_mesh(_corners_A, _bbox_A, PILImage.open(PATH_A).size),
               texture=_tex_A, opacity=0.9),
    p.add_mesh(make_section_mesh(_corners_B, _bbox_B, PILImage.open(PATH_B).size),
               texture=_tex_B, opacity=0.9),
]

# ── Boreholes (toggleable) ────────────────────────────────────────────────────
_borehole_actors = []

# Collar markers + name labels
for _name, _bh in BOREHOLES.items():
    _borehole_actors.append(
        p.add_mesh(pv.Sphere(radius=TUBE_RADIUS, center=(_bh[0], _bh[1], _bh[2])),
                   color=BH_COLORS[_name])
    )
    _borehole_actors.append(
        p.add_point_labels(
            np.array([[_bh[0], _bh[1], _bh[2]]]), [_name],
            font_size=14, bold=True, text_color=BH_COLORS[_name],
            always_visible=True, shape_opacity=0.0,
        )
    )

# Formation tubes
_df_fault = _csv[_csv['formation'].isin(FAULT_FORMATIONS)].copy()
_df_bh    = _csv[~_csv['formation'].isin(FAULT_FORMATIONS)].copy()
_df_bh['_key'] = _df_bh.apply(lambda r: (round(r['X'], 1), round(r['Y'], 1)), axis=1)

for _key, _grp in _df_bh.groupby('_key'):
    _grp   = _grp.sort_values('Z', ascending=False).reset_index(drop=True)
    _x, _y = float(_grp.iloc[0]['X']), float(_grp.iloc[0]['Y'])
    _z_top = COLLARS.get(_key, float(_grp['Z'].max()))
    for _, _row in _grp.iterrows():
        _z_bot = float(_row['Z'])
        _color = FORMATION_COLORS.get(_row['formation'], '#888888')
        if _z_top > _z_bot:
            _borehole_actors.append(
                p.add_mesh(
                    pv.Line((_x, _y, _z_top), (_x, _y, _z_bot)).tube(radius=TUBE_RADIUS),
                    color=_color,
                )
            )
        _z_top = _z_bot

# Fault intercept spheres
for _fn, _fg in _df_fault.groupby('formation'):
    _pts = np.column_stack([_fg['X'].values, _fg['Y'].values, _fg['Z'].values])
    _borehole_actors.append(
        p.add_mesh(pv.PolyData(_pts), color=FAULT_COLORS.get(_fn, 'black'),
                   point_size=8, render_points_as_spheres=True)
    )

# ── Toggle checkboxes ─────────────────────────────────────────────────────────
def _toggle_sections(flag):
    for _a in _section_actors:
        _a.SetVisibility(flag)

def _toggle_boreholes(flag):
    for _a in _borehole_actors:
        _a.SetVisibility(flag)

p.add_checkbox_button_widget(
    _toggle_sections, value=True, position=(10, 10), size=30,
    border_size=3, color_on='white', color_off='grey', background_color='grey',
)
p.add_text('Sections',  position=(0.065, 0.012), font_size=11, color='white', shadow=True)

p.add_checkbox_button_widget(
    _toggle_boreholes, value=True, position=(10, 50), size=30,
    border_size=3, color_on='white', color_off='grey', background_color='grey',
)
p.add_text('Boreholes', position=(0.065, 0.062), font_size=11, color='white', shadow=True)

p.show()
