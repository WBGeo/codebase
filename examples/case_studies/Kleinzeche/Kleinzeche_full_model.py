"""
Kleinzeche — Full structural model
====================================
Stratigraphic model with Fault-1 using picks from Profile A and Profile B.
Input data loaded from pre-built CSV files in input_data/.
Run input_data/prepare_input_data.py to regenerate those files after re-picking.

Formations (youngest → oldest):
  Strat_Series3  Gravel-1
  Strat_Series2  Loose-1
  Strat_Series1  Silt-1 / Sand-1 / Silt-2 / Sand-2 / Silt-3 / Sand-3
                 Clay-1 / Coal-1 / Clay-2

Fault-1 active for Strat_Series1.  Gravel-1 and Loose-1 are post-fault.

Coordinate system: UTM 32N  X 380040–380132  Y 5700884–5700938  Z 30–130 m a.s.l.
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

# ── Local Kleinzeche imports ───────────────────────────────────────────────────
try:
    _SECTION_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SECTION_DIR = os.path.join(os.getcwd(), "examples/case_studies/Kleinzeche")

_INPUT_DATA_DIR = os.path.join(_SECTION_DIR, "input_data")

for _d in [_SECTION_DIR, _INPUT_DATA_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from input_data.Kleinzeche_shared import (
    BOREHOLES, BH_COLORS, COLLARS, Z_MIN, Z_MIN_A, Z_MIN_B,
    FORMATION_COLORS, FAULT_FORMATIONS, FAULT_COLORS, TUBE_RADIUS,
    PATH_A, PATH_B, CSV_PATH, MARKER_CACHE,
    find_data_bbox, georeference_section, make_section_mesh,
)

# ── Load input data ────────────────────────────────────────────────────────────
surface_points       = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'full_surface_points.csv'))
orientations         = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'full_orientations.csv'))
fault_surface_points = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'fault1_surface_points.csv'))
fault_orientations   = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'fault1_orientations.csv'))

# ── Grid ───────────────────────────────────────────────────────────────────────
# Z extended 10 m below Z_MIN to capture Clay/Coal in the hanging wall.
_Z_MIN_MODEL = Z_MIN - 10   # 30 m

grid = RegularGrid(
    extent=(380040, 380132, 5700884, 5700938, _Z_MIN_MODEL, 130),
    resolution=(92, 54, 90),
)

# ── Fault frame ────────────────────────────────────────────────────────────────
data_faults = InputData_FaultElements(
    name='Kleinzeche',
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

#%%

# ── Formation mapping (youngest → oldest) ─────────────────────────────────────
_available = set(surface_points['formation'].unique())
_ALL_SERIES = [
    ('Strat_Series3', ('Gravel-1',)),
    ('Strat_Series2', ('Loose-1',)),
    ('Strat_Series1', ('Silt-1', 'Sand-1', 'Silt-2', 'Sand-2', 'Silt-3', 'Sand-3',
                       'Clay-1', 'Coal-1', 'Clay-2')),
]
mapping_object = {}
for _series, _fms in _ALL_SERIES:
    _present = tuple(f for f in _fms if f in _available)
    if _present:
        mapping_object[_series] = _present

# ── Structural frame ───────────────────────────────────────────────────────────
data_elements = InputData_StructuralElements(
    name='Kleinzeche_full',
    mapping_object=mapping_object,
    surface_points=surface_points,
    orientations=orientations,
)

frame = general.build_structural_frame(
    input_data_elements=data_elements,
    grid=grid,
    fault_model_results=fault_model_result,
)

# ── Formation colors and fault activity ───────────────────────────────────────
for _series, _fms in _ALL_SERIES:
    if _series not in mapping_object:
        continue
    for _fm in mapping_object[_series]:
        if _fm in FORMATION_COLORS:
            frame[_series][_fm].set_color(FORMATION_COLORS[_fm])

# Fault-1 displaces Strat_Series1 (Silt/Sand/Clay); Gravel-1 and Loose-1 are post-fault.
frame.set_fault_activity_by_group(fault_name='Fault-1', group_name='Strat_Series1')

# ── Interpolation ──────────────────────────────────────────────────────────────
for _series in mapping_object:
    frame[_series].set_interpolation_method('Universal Co-Kriging')

# plot_structural_model_2D(frame, show_result=False)
frame.detailed_report()
plot_structural_model_3D(frame)

#%%

# ── Compute structural model ───────────────────────────────────────────────────
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# ── Visualize results ──────────────────────────────────────────────────────────
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(
    structural_model_result.structural_frame,
    show_surface_meshes=True,
    show_orientations=False,
)

#%%

# ── Combined: model + cross sections + borehole trajectories ──────────────────
_csv = pd.read_csv(CSV_PATH)

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

#%%

# TODO: Next step, meshing does not yet work for this model
# Explicit Unstructured meshing (Structured does not work with faults)
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d

# DISTANCE_THRESHOLD must be << model extent.  The default (50) removes ALL points
# from a 50×50 m model.  Use ~10 % of the minimum horizontal extent (here 5 m).
mesh_result = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    z_threshold=0.1,
    tolerance=1,
    DISTANCE_THRESHOLD=5,  # Minimum distance between points to be retained (after surface sampling and before meshing)
    mesh_size=10  # Target edge length for meshing (not a hard constraint, but smaller → finer mesh)
)
plot_mesh_3d(mesh_result, structural_model_result, show_plotter=True)
