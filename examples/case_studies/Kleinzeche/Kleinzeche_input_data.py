import os
import sys
import json

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image as PILImage

# ── Local Kleinzeche imports ──────────────────────────────────────────────────
try:
    _SECTION_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running as IPython cells (e.g. PyCharm console) — __file__ not defined
    _SECTION_DIR = os.path.join(os.getcwd(), "examples/case_studies/Kleinzeche")
if _SECTION_DIR not in sys.path:
    sys.path.insert(0, _SECTION_DIR)

from Kleinzeche_shared import (
    BOREHOLES, BH_COLORS, COLLARS, Z_MIN_A, Z_MIN_B,
    FORMATION_COLORS, FAULT_FORMATIONS, FAULT_COLORS, TUBE_RADIUS,
    PATH_A, PATH_B, CSV_PATH, MARKER_CACHE, SAT_IMG_PATH, SAT_EXT_PATH,
    find_data_bbox, georeference_section, make_section_mesh,
)

#%%

# Coordinate system: UTM 32N, WGS84

#%%

# Cross sections stored in path
# examples/case_studies/Kleinzeche/Kleinzeche_profile_A.png
# examples/case_studies/Kleinzeche/Kleinzeche_profile_B.png

#%%

# Subtract a reference origin for numerical stability in PyVista.
# Z is kept as absolute elevation (metres above/below sea level).
origin = np.array([BOREHOLES['MP1'][0], BOREHOLES['MP1'][1], 0.0])

def to_local(utm_pt):
    return utm_pt - origin

#%%

REQUIRED_MARKERS = ['MP1', 'MO1', 'O4', 'O3']

def pick_marker_pixels(img_path, labels, colors):
    """
    Open the image in the system default viewer and collect pixel positions
    via console input() prompts.

    In Windows Paint: hover over the marker — coordinates appear at the bottom.
    In Windows Photos: right-click → 'Open with' → Paint for coordinate display.
    """
    try:
        if sys.platform == 'win32':
            os.startfile(img_path)
    except Exception:
        pass

    print(f"\nOpened: {os.path.basename(img_path)}")
    print("Hover over each marker in the viewer to read pixel x,y coordinates.")
    print("(Windows Paint shows coordinates at the bottom-left of the window)\n")

    result = {}
    for label, color in zip(labels, colors):
        while True:
            try:
                raw = input(f"  Pixel x,y of  {label} ({color}) marker: ")
                x, y = map(float, raw.strip().split(','))
                result[label] = (x, y)
                break
            except ValueError:
                print("  Please enter as  x,y  e.g.  152,28")
    return result

def load_or_pick_markers():
    """
    Load cached marker pixel positions from JSON.
    If the cache is missing or incomplete, run the interactive picker.
    Delete marker_pixels.json to force re-picking.
    """
    if os.path.exists(MARKER_CACHE):
        with open(MARKER_CACHE) as f:
            data = json.load(f)
        if all(k in data for k in REQUIRED_MARKERS):
            print("Loaded marker pixels from cache:", MARKER_CACHE)
            return {k: tuple(v) for k, v in data.items()}
        print("Cache incomplete — re-running picker and overwriting.")
        os.remove(MARKER_CACHE)

    print("Profile A: click MP1 (yellow), then MO1 (red)")
    markers_a = pick_marker_pixels(PATH_A, ['MP1', 'MO1'], ['yellow', 'red'])
    print("Profile B: click O4 (green), then O3 (blue)")
    markers_b = pick_marker_pixels(PATH_B, ['O4', 'O3'], ['green', 'blue'])

    all_markers = {**markers_a, **markers_b}
    if all(k in all_markers for k in REQUIRED_MARKERS):
        with open(MARKER_CACHE, 'w') as f:
            json.dump(all_markers, f, indent=2)
        print("Saved to", MARKER_CACHE)
    else:
        print("WARNING: not all markers were clicked — cache not saved.")
    return all_markers

marker_pixels = load_or_pick_markers()
px_MP1 = marker_pixels['MP1']
px_MO1 = marker_pixels['MO1']
px_O4  = marker_pixels['O4']
px_O3  = marker_pixels['O3']

bbox_A = find_data_bbox(PATH_A)
bbox_B = find_data_bbox(PATH_B)
print(f"Profile A  MP1={px_MP1}  MO1={px_MO1}  bbox={bbox_A}")
print(f"Profile B  O4={px_O4}   O3={px_O3}    bbox={bbox_B}")

#%%

# ── Georeferenced plane construction ─────────────────────────────────────────
corners_A = georeference_section(
    BOREHOLES['MP1'], BOREHOLES['MO1'], px_MP1, px_MO1, bbox_A, Z_MIN_A
)
corners_B = georeference_section(
    BOREHOLES['O4'], BOREHOLES['O3'], px_O4, px_O3, bbox_B, Z_MIN_B
)

print(f"Profile A: derived z_max = {corners_A[0][2]:.2f} m  (bh1 elevation = {BOREHOLES['MP1'][2]} m)")
print(f"Profile B: derived z_max = {corners_B[0][2]:.2f} m  (bh1 elevation = {BOREHOLES['O4'][2]} m)")

#%%

# ── Load images as PyVista textures ──────────────────────────────────────────
tex_A = pv.read_texture(PATH_A)
tex_B = pv.read_texture(PATH_B)

img_size_A = PILImage.open(PATH_A).size  # (width, height)
img_size_B = PILImage.open(PATH_B).size

plane_A = make_section_mesh([to_local(c) for c in corners_A], bbox_A, img_size_A)
plane_B = make_section_mesh([to_local(c) for c in corners_B], bbox_B, img_size_B)

#%%

# ── Satellite imagery (Esri World Imagery via contextily) ─────────────────────
# Requires:  pip install contextily pyproj
import contextily as cx
from pyproj import Transformer

# Bounding box from borehole extents + margin (metres, UTM 32N)
_bh_xy  = np.array([bh[:2] for bh in BOREHOLES.values()])
_margin = 50  # metres around the outermost boreholes
_xmin_utm = float(_bh_xy[:, 0].min()) - _margin
_xmax_utm = float(_bh_xy[:, 0].max()) + _margin
_ymin_utm = float(_bh_xy[:, 1].min()) - _margin
_ymax_utm = float(_bh_xy[:, 1].max()) + _margin

if os.path.exists(SAT_IMG_PATH) and os.path.exists(SAT_EXT_PATH):
    print("Loaded satellite image from cache.")
    sat_pil = PILImage.open(SAT_IMG_PATH)
    with open(SAT_EXT_PATH) as f:
        sat_ext_utm = json.load(f)   # keys: xmin, xmax, ymin, ymax (UTM 32N)
else:
    print("Fetching satellite tiles…")
    # Convert UTM 32N (EPSG:32632) → Web Mercator (EPSG:3857) for contextily
    _to_merc = Transformer.from_crs("EPSG:32632", "EPSG:3857", always_xy=True)
    _xmin_m, _ymin_m = _to_merc.transform(_xmin_utm, _ymin_utm)
    _xmax_m, _ymax_m = _to_merc.transform(_xmax_utm, _ymax_utm)

    sat_arr, sat_ext_merc = cx.bounds2img(
        _xmin_m, _ymin_m, _xmax_m, _ymax_m,
        source=cx.providers.Esri.WorldImagery,
    )
    # sat_ext_merc = (left, right, bottom, top) in EPSG:3857
    # sat_arr rows: row 0 = north, row -1 = south

    # Convert extent back to UTM 32N
    _to_utm = Transformer.from_crs("EPSG:3857", "EPSG:32632", always_xy=True)
    _x0, _y0 = _to_utm.transform(sat_ext_merc[0], sat_ext_merc[2])  # SW corner
    _x1, _y1 = _to_utm.transform(sat_ext_merc[1], sat_ext_merc[3])  # NE corner
    sat_ext_utm = {"xmin": _x0, "xmax": _x1, "ymin": _y0, "ymax": _y1}

    sat_pil = PILImage.fromarray(sat_arr[:, :, :3].astype(np.uint8))
    sat_pil.save(SAT_IMG_PATH)
    with open(SAT_EXT_PATH, 'w') as f:
        json.dump(sat_ext_utm, f, indent=2)
    print(f"Saved to {SAT_IMG_PATH}")

# Horizontal plane at mean borehole surface elevation
z_surface = float(np.mean([bh[2] for bh in BOREHOLES.values()]))

_sat_corners = [
    np.array([sat_ext_utm['xmin'], sat_ext_utm['ymax'], z_surface]),  # NW (top-left)
    np.array([sat_ext_utm['xmax'], sat_ext_utm['ymax'], z_surface]),  # NE (top-right)
    np.array([sat_ext_utm['xmax'], sat_ext_utm['ymin'], z_surface]),  # SE (bottom-right)
    np.array([sat_ext_utm['xmin'], sat_ext_utm['ymin'], z_surface]),  # SW (bottom-left)
]
sat_pts  = np.array([to_local(c) for c in _sat_corners])
sat_mesh = pv.PolyData(sat_pts, np.array([4, 0, 1, 2, 3]))
# UV: v=1 → row 0 of image (north); v=0 → last row (south)
sat_mesh.active_texture_coordinates = np.array([
    [0.0, 1.0],  # NW
    [1.0, 1.0],  # NE
    [1.0, 0.0],  # SE
    [0.0, 0.0],  # SW
])
sat_tex = pv.numpy_to_texture(np.array(sat_pil))

#%%

# ── 3D plot ───────────────────────────────────────────────────────────────────
plotter = pv.Plotter()

plotter.add_mesh(sat_mesh, texture=sat_tex, opacity=1.0)
plotter.add_mesh(plane_A, texture=tex_A, opacity=1.0)
plotter.add_mesh(plane_B, texture=tex_B, opacity=1.0)

for name, pt_utm in BOREHOLES.items():
    pt = to_local(pt_utm)
    plotter.add_mesh(pv.Sphere(radius=1.5, center=pt), color=BH_COLORS[name])
    plotter.add_point_labels(
        np.array([pt]),
        [name],
        font_size=14,
        bold=True,
        text_color=BH_COLORS[name],
        always_visible=True,
        shape_opacity=0.0,
    )

plotter.add_axes()
plotter.show_bounds(
    xtitle=f'Easting − {int(origin[0])} m',
    ytitle=f'Northing − {int(origin[1])} m',
    ztitle='Elevation (m a.s.l.)',
)

# ── Borehole trajectories from df1_filtered.csv ──────────────────────────────
# Each CSV point marks the BOTTOM of a formation unit.
# Segments are drawn from the borehole collar (or previous boundary) down to
# each point and coloured by formation.  Fault points are shown as black spheres.

_bh_csv = pd.read_csv(CSV_PATH)

_df_fault = _bh_csv[_bh_csv['formation'].isin(FAULT_FORMATIONS)].copy()
_df_bh    = _bh_csv[~_bh_csv['formation'].isin(FAULT_FORMATIONS)].copy()

_df_bh['_bh_key'] = _df_bh.apply(
    lambda r: (round(r['X'], 1), round(r['Y'], 1)), axis=1
)

for _bh_key, _group in _df_bh.groupby('_bh_key'):
    _group   = _group.sort_values('Z', ascending=False).reset_index(drop=True)
    _x_local = float(_group.iloc[0]['X']) - origin[0]
    _y_local = float(_group.iloc[0]['Y']) - origin[1]
    _z_top   = COLLARS.get(_bh_key, float(_group['Z'].max()))

    for _, _row in _group.iterrows():
        _z_bot = float(_row['Z'])
        _color = FORMATION_COLORS.get(_row['formation'], '#888888')
        if _z_top > _z_bot:
            _line = pv.Line((_x_local, _y_local, _z_top), (_x_local, _y_local, _z_bot))
            plotter.add_mesh(_line.tube(radius=TUBE_RADIUS), color=_color)
        _z_top = _z_bot

# Fault points — coloured spheres per fault
for _fname, _fgroup in _df_fault.groupby('formation'):
    _fcolor = FAULT_COLORS.get(_fname, 'black')
    _fault_pts = np.column_stack([
        _fgroup['X'].values - origin[0],
        _fgroup['Y'].values - origin[1],
        _fgroup['Z'].values,
    ])
    plotter.add_mesh(
        pv.PolyData(_fault_pts),
        color=_fcolor, point_size=8, render_points_as_spheres=True,
    )

plotter.show()
