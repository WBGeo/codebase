"""
Kleinzeche — Simple (2.5D) structural model
============================================
Simplified model using Profile A picks only, extruded along strike.
Input data loaded from pre-built CSV files in input_data/.
Run input_data/prepare_input_data.py to regenerate those files after re-picking.

Formations (youngest → oldest):
  Strat_Series3  Gravel-1
  Strat_Series2  Loose-1
  Strat_Series1  Silt-1 / Sand-1 / Silt-2 / Sand-2 / Silt-3 / Sand-3

Fault-1 active for Strat_Series2 upward.  Gravel-1 is post-fault.

Grid: 50 × 50 m square footprint centred on Profile A mid-point.
  X : 380046 – 380096   Y : 5700883 – 5700933   Z : 40 – 130 m a.s.l.
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
    BOREHOLES, BH_COLORS, COLLARS, Z_MIN, Z_MIN_A,
    FORMATION_COLORS, FAULT_FORMATIONS, FAULT_COLORS, TUBE_RADIUS,
    PATH_A, CSV_PATH, MARKER_CACHE,
    find_data_bbox, georeference_section, make_section_mesh,
)

# ── Load input data ────────────────────────────────────────────────────────────
surface_points       = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'simple_surface_points.csv'))
orientations         = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'simple_orientations.csv'))
fault_surface_points = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'fault1_surface_points.csv'))
fault_orientations   = pd.read_csv(os.path.join(_INPUT_DATA_DIR, 'fault1_orientations.csv'))

# ── Grid ───────────────────────────────────────────────────────────────────────
grid = RegularGrid(
    extent=(380030, 380110, 5700875, 5700945, Z_MIN, 130),
    resolution=(100, 100, 100),
)

# ── Fault frame ────────────────────────────────────────────────────────────────
data_faults = InputData_FaultElements(
    name='Kleinzeche_simple',
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
    ('Strat_Series1', ('Silt-1', 'Sand-1', 'Silt-2', 'Sand-2', 'Silt-3', 'Sand-3')),
]
mapping_object = {}
for _series, _fms in _ALL_SERIES:
    _present = tuple(f for f in _fms if f in _available)
    if _present:
        mapping_object[_series] = _present

# ── Structural frame ───────────────────────────────────────────────────────────
data_elements = InputData_StructuralElements(
    name='Kleinzeche_simple',
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

frame.set_fault_activity_by_group(fault_name='Fault-1', group_name='Strat_Series2')

# ── Interpolation ──────────────────────────────────────────────────────────────
for _series in mapping_object:
    frame[_series].set_interpolation_method('Universal Co-Kriging')

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

#%%

from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.mesh_format.exudos.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
cwd = os.getcwd()

#################################
# ImplicitStructured meshing.   #
#################################
mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)
#################################
# Explicit Unstructured meshing #
#################################
 #load wells
wells = load_wells_from_csv(cwd + "/examples/case_studies/Kleinzeche/input_data/engineering_objects/kleinzeche_wells.csv")
mesh_unstr = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=1,
    wells= wells,
    mesh_size=1,
    curve_mesh_size=2,
    DISTANCE_THRESHOLD = 3,
    PROJECTION_THRESHOLD = 4,
    EXTRUSION_FACTOR = 4.2,
    z_threshold = 1,
)
#plot_mesh_3d(mesh_unstr, structural_model_result, show_plotter=True)
###########################################################################################################################
#                                                    Exporting meshes
###########################################################################################################################
#########################
# Export mesh to exodus #
#########################

# Implicit structured mesh
#buf = export_mesh_results_to_exodus(mesh_implicit)
#with open("filename_implic_klein.exo", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_exodus(mesh_unstr)
#with open("filename_klein.exo", "wb") as f:
#    f.write(buf.getvalue())
