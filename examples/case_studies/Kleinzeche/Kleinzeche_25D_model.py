"""
Kleinzeche — 2.5D structural model from Profile A picks
========================================================
Simplified model using Profile A picks only.  The geometry is extruded along
the strike direction (perpendicular to Profile A) to give 3-D continuity.

Formations (youngest → oldest):
  Strat_Series3  Gravel-1
  Strat_Series2  Loose-1
  Strat_Series1  Silt-1 / Sand-1 / Silt-2 / Sand-2 / Silt-3 / Sand-3

Fault-1 from borehole CSV (same data as the full model): active for Strat_Series1.
No Clay-1 / Coal-1 / Clay-2 (those are only visible in Profile B).

Grid
----
Square horizontal footprint centred on the Profile A mid-point (MP1 → MO1),
side length = Profile A length ≈ 49 m → rounded to 50 m.
  X : 380046 – 380096  (50 m, along section)
  Y : 5700883 – 5700933 (50 m, across section; Profile A runs through middle)
  Z : 40 – 130 m a.s.l.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image as PILImage

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D,
    plot_structural_model_3D,
    plot_fault_model_3D,
)
from core.structural_modeling_components import general, general_faults

# ── Local Kleinzeche imports ──────────────────────────────────────────────────
try:
    _SECTION_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SECTION_DIR = os.path.join(os.getcwd(), "examples/case_studies/Kleinzeche")
if _SECTION_DIR not in sys.path:
    sys.path.insert(0, _SECTION_DIR)

from Kleinzeche_shared import (
    BOREHOLES, BH_COLORS, COLLARS, Z_MIN_A, Z_MIN,
    FORMATION_COLORS, FAULT_FORMATIONS, FAULT_COLORS, TUBE_RADIUS,
    PATH_A, CSV_PATH, MARKER_CACHE,
    find_data_bbox, georeference_section, make_section_mesh,
)
from Kleinzeche_section_picking import picks_to_dataframes, FORMATIONS_A

# ── Profile A picks (surface points only) ────────────────────────────────────
_sp_raw, _ = picks_to_dataframes()

_A_FMSET = set(FORMATIONS_A)
surface_points = _sp_raw[_sp_raw['formation'].isin(_A_FMSET)].reset_index(drop=True)

print(f"Surface points (Profile A): {len(surface_points)} rows")

# ── 2.5D extension for surface points ────────────────────────────────────────
# Duplicate picks at the near and far Y model edges to constrain the
# interpolator across the full Y extent.
_Y_MIN_GRID = round((BOREHOLES['MP1'][1] + BOREHOLES['MO1'][1]) / 2 - 25)  # 5700883
_Y_MAX_GRID = round((BOREHOLES['MP1'][1] + BOREHOLES['MO1'][1]) / 2 + 25)  # 5700933
_Y_NEAR = float(_Y_MIN_GRID + 2)
_Y_FAR  = float(_Y_MAX_GRID - 2)

_sp_near = surface_points.copy(); _sp_near['Y'] = _Y_NEAR
_sp_far  = surface_points.copy(); _sp_far['Y']  = _Y_FAR
surface_points = pd.concat([surface_points, _sp_near, _sp_far], ignore_index=True)

print(f"Surface points (after 2.5D extension): {len(surface_points)} rows")

# ── Formation mapping (youngest → oldest) ────────────────────────────────────
_ALL_SERIES = [
    ('Strat_Series3', ('Gravel-1',)),
    ('Strat_Series2', ('Loose-1',)),
    ('Strat_Series1', ('Silt-1', 'Sand-1', 'Silt-2', 'Sand-2', 'Silt-3', 'Sand-3')),
]
_available = set(surface_points['formation'].unique())
mapping_object = {}
for _series, _fms in _ALL_SERIES:
    _present = tuple(f for f in _fms if f in _available)
    if _present:
        mapping_object[_series] = _present

# ── Orientations: one per formation per fault domain ─────────────────────────
# One upward (0,0,1) orientation at X_MIN (hanging wall) and X_MAX (footwall),
# placed at model-centre Y and the mean Z of each formation's surface points.
_X_MIN = round((BOREHOLES['MP1'][0] + BOREHOLES['MO1'][0]) / 2 - 25)  # 380046
_X_MAX = round((BOREHOLES['MP1'][0] + BOREHOLES['MO1'][0]) / 2 + 25)  # 380096
_Y_CENTER = float(_Y_MIN_GRID + _Y_MAX_GRID) / 2

def _mean_z(fm):
    rows = surface_points[surface_points['formation'] == fm]['Z']
    return float(rows.mean()) if len(rows) else float((Z_MIN + 130) / 2)

_ori_rows = []
for _series, _fms in _ALL_SERIES:
    for _fm in _fms:
        if _fm not in _available:
            continue
        _z = _mean_z(_fm)
        for _x in [float(_X_MIN), float(_X_MAX)]:
            _ori_rows.append({
                'X': _x, 'Y': _Y_CENTER, 'Z': _z,
                'G_x': 0.0, 'G_y': 0.0, 'G_z': 1.0,
                'formation': _fm,
            })

orientations = pd.DataFrame(_ori_rows)
print(f"Orientations: {len(orientations)} rows  (2 per formation × {len(orientations)//2} formations)")

# ── Grid ─────────────────────────────────────────────────────────────────────
grid = RegularGrid(
    extent=(_X_MIN, _X_MAX, _Y_MIN_GRID, _Y_MAX_GRID, Z_MIN, 130),
    resolution=(50, 50, 90),
)

# ── Fault-1 (same data as full model) ─────────────────────────────────────────
_csv_full = pd.read_csv(CSV_PATH)
fault_surface_points = _csv_full[_csv_full['formation'] == 'Fault-1'].copy().reset_index(drop=True)

# Fit a plane to the borehole intercepts via SVD → upward fault normal
_pts = fault_surface_points[['X', 'Y', 'Z']].values
_centroid = _pts.mean(axis=0)
_, _, _Vt = np.linalg.svd(_pts - _centroid)
_fn = _Vt[-1]
if _fn[2] < 0:
    _fn = -_fn
_fn /= np.linalg.norm(_fn)

fault_orientations = pd.DataFrame([{
    'X': float(_centroid[0]), 'Y': float(_centroid[1]), 'Z': float(_centroid[2]),
    'G_x': float(_fn[0]), 'G_y': float(_fn[1]), 'G_z': float(_fn[2]),
    'formation': 'Fault-1',
}])
print(f"Fault-1 normal: G=({_fn[0]:.3f}, {_fn[1]:.3f}, {_fn[2]:.3f})")

data_faults = InputData_FaultElements(
    name='Kleinzeche_25D',
    fault_surface_points=fault_surface_points,
    fault_orientations=fault_orientations,
    fault_names=['Fault-1'],
)

fault_frame = general_faults.build_fault_frame(
    input_data_fault_elements=data_faults,
    grid=grid,
)
fault_frame.detailed_report()

#%%

plot_fault_model_3D(fault_frame)

#%%

fault_model_result = general_faults.compute_fault_domains(fault_frame)
plot_fault_model_3D(fault_model_result.fault_frame)

# ── Structural input data ─────────────────────────────────────────────────────
data_elements = InputData_StructuralElements(
    name='Kleinzeche_25D',
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

# Fault-1 displaces Strat_Series1 (Silt/Sand) only; Gravel-1 and Loose-1 are post-fault.
frame.set_fault_activity_by_group(fault_name='Fault-1', group_name='Strat_Series2')

frame.detailed_report()

#%%

# ── Interpolation ─────────────────────────────────────────────────────────────
for _series in mapping_object:
    frame[_series].set_interpolation_method('Universal Co-Kriging')

# Preview input data
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

# ── Combined: model + Profile A section + borehole trajectories ───────────────
_csv = pd.read_csv(CSV_PATH)

with open(MARKER_CACHE) as _f:
    _markers = {k: tuple(v) for k, v in json.load(_f).items()}

_bbox_A    = find_data_bbox(PATH_A)
_corners_A = georeference_section(
    BOREHOLES['MP1'], BOREHOLES['MO1'],
    _markers['MP1'], _markers['MO1'],
    _bbox_A, Z_MIN_A,
)
_tex_A = pv.read_texture(PATH_A)

p = plot_structural_model_3D(
    structural_model_result.structural_frame,
    show_surface_meshes=True,
    show_orientations=False,
    show=False,
)

# ── Profile A section (toggleable) ────────────────────────────────────────────
_section_actors = [
    p.add_mesh(
        make_section_mesh(_corners_A, _bbox_A, PILImage.open(PATH_A).size),
        texture=_tex_A, opacity=0.9,
    ),
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
def _toggle_section(flag):
    for _a in _section_actors:
        _a.SetVisibility(flag)

def _toggle_boreholes(flag):
    for _a in _borehole_actors:
        _a.SetVisibility(flag)

p.add_checkbox_button_widget(
    _toggle_section, value=True, position=(10, 10), size=30,
    border_size=3, color_on='white', color_off='grey', background_color='grey',
)
p.add_text('Section A',  position=(0.065, 0.012), font_size=11, color='white', shadow=True)

p.add_checkbox_button_widget(
    _toggle_boreholes, value=True, position=(10, 50), size=30,
    border_size=3, color_on='white', color_off='grey', background_color='grey',
)
p.add_text('Boreholes', position=(0.065, 0.062), font_size=11, color='white', shadow=True)

p.show()