"""
Kleinzeche shared constants and utility functions
=================================================
Imported by Kleinzeche_input_data.py, Kleinzeche_section_picking.py, and
Kleinzeche_section_model.py to avoid duplicating borehole coordinates,
formation colours, path definitions, and georeferencing helpers.
"""

import os

import numpy as np

# ── Paths (all __file__-relative) ─────────────────────────────────────────────

SECTION_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(SECTION_DIR, "input_data")
CACHE_DIR      = os.path.join(SECTION_DIR, "cache")

PATH_A       = os.path.join(INPUT_DATA_DIR, "Kleinzeche_profile_A.png")
PATH_B       = os.path.join(INPUT_DATA_DIR, "Kleinzeche_profile_B.png")
CSV_PATH     = os.path.join(INPUT_DATA_DIR, "df1_filtered.csv")
MARKER_CACHE = os.path.join(CACHE_DIR,      "marker_pixels.json")
PICKS_CACHE  = os.path.join(CACHE_DIR,      "section_picks.json")
SAT_IMG_PATH = os.path.join(CACHE_DIR,      "satellite.png")
SAT_EXT_PATH = os.path.join(CACHE_DIR,      "satellite_extent.json")

# ── Borehole surface coordinates (UTM 32N, WGS84: Easting, Northing, Elevation) ──

BOREHOLES = {
    'MP1': np.array([380050.00, 5700894.90, 125.44]),  # yellow – Profile A left
    'MO1': np.array([380091.60, 5700920.90, 125.12]),  # red    – Profile A right
    'O4':  np.array([380090.90, 5700926.60, 125.18]),  # green  – Profile B left
    'O3':  np.array([380121.70, 5700903.60, 124.61]),  # blue   – Profile B right
}

BH_COLORS = {'MP1': 'yellow', 'MO1': 'red', 'O4': 'green', 'O3': 'blue'}

# Collar elevations (m a.s.l.) keyed by rounded (Easting, Northing).
# MI1 is included for borehole trajectory plotting only.
COLLARS = {
    (380050.0, 5700894.9): 125.44,  # MP1
    (380091.6, 5700920.9): 125.12,  # MO1
    (380090.9, 5700926.6): 125.18,  # O4
    (380121.7, 5700903.6): 124.61,  # O3
    (380114.5, 5700917.1): 122.72,  # MI1 — collar estimated from Gravel-1 top
}

# ── Elevation bounds ──────────────────────────────────────────────────────────

Z_MIN_A =  55.0   # Profile A bottom
Z_MIN_B = -70.0   # Profile B bottom
Z_MIN   =  40.0   # Structural model grid floor

# ── Formation colours (complete set, used by all three scripts) ───────────────

FORMATION_COLORS = {
    'Gravel-1': '#959595',
    'Loose-1':  '#00c401',
    'Silt-1':   '#828701', 'Silt-2': '#828701', 'Silt-3': '#828701', 'Silt-5': '#828701',
    'Sand-1':   '#ec7a10', 'Sand-2': '#ec7a10', 'Sand-3': '#ec7a10', 'Sand-6': '#ec7a10',
    'Clay-1':   '#c07dcc', 'Clay-2': '#c07dcc',
    'Coal-1':   '#4d2900',
}

# ── Fault sets ────────────────────────────────────────────────────────────────

FAULT_FORMATIONS = {'Fault-1', 'Fault-2'}
FAULT_COLORS     = {'Fault-1': 'red', 'Fault-2': 'black'}

TUBE_RADIUS = 1.5

# ── Utility functions ─────────────────────────────────────────────────────────

def find_data_bbox(img_path, white_threshold=230):
    """Return (px_left, px_right, px_top, px_bottom) of the non-white data area."""
    from PIL import Image as PILImage
    arr = np.array(PILImage.open(img_path).convert('RGB'))
    is_data = ~(
        (arr[:, :, 0] > white_threshold) &
        (arr[:, :, 1] > white_threshold) &
        (arr[:, :, 2] > white_threshold)
    )
    rows, cols = np.where(is_data)
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def georeference_section(bh1_utm, bh2_utm, px_bh1, px_bh2, data_bbox, z_min):
    """
    Compute the 4 corner 3D coordinates (UTM) of a cross-section image.

    The image horizontal axis maps linearly to distance along the bh1→bh2 line.
    The vertical axis is calibrated using bh1's known surface elevation and its
    picked collar pixel row, with z_min anchored at the bottom of the data bbox.
    z_max is derived — do not pass it manually.

    Returns: [top-left, top-right, bottom-right, bottom-left] as (E, N, Z) arrays.
    """
    px_left, px_right, px_top, px_bottom = data_bbox

    d_bh    = np.linalg.norm(bh2_utm[:2] - bh1_utm[:2])
    h_scale = d_bh / (px_bh2[0] - px_bh1[0])

    s_left  = (px_left  - px_bh1[0]) * h_scale
    s_right = (px_right - px_bh1[0]) * h_scale

    direction = (bh2_utm[:2] - bh1_utm[:2]) / d_bh

    z_bh1  = bh1_utm[2]
    py_bh1 = px_bh1[1]
    v_scale = (z_bh1 - z_min) / (px_bottom - py_bh1)
    z_max   = z_bh1 + (py_bh1 - px_top) * v_scale

    def corner(s, z):
        xy = bh1_utm[:2] + s * direction
        return np.array([xy[0], xy[1], z])

    return [
        corner(s_left,  z_max),  # top-left
        corner(s_right, z_max),  # top-right
        corner(s_right, z_min),  # bottom-right
        corner(s_left,  z_min),  # bottom-left
    ]


def make_section_mesh(corners, data_bbox, img_size):
    """Create a textured PyVista quad from 4 corner points.

    UV coordinates are set so that only the data bounding box (non-white area)
    is mapped to the plane.  Corners are taken as-is; pass
    ``[to_local(c) for c in corners]`` when working in a local-coordinate
    PyVista scene.
    """
    import pyvista as pv
    pts  = np.array(corners)
    mesh = pv.PolyData(pts, np.array([4, 0, 1, 2, 3]))

    px_left, px_right, px_top, px_bottom = data_bbox
    img_w, img_h = img_size

    u_left   = px_left   / img_w
    u_right  = px_right  / img_w
    v_top    = 1.0 - px_top    / img_h
    v_bottom = 1.0 - px_bottom / img_h

    mesh.active_texture_coordinates = np.array([
        [u_left,  v_top   ],  # top-left
        [u_right, v_top   ],  # top-right
        [u_right, v_bottom],  # bottom-right
        [u_left,  v_bottom],  # bottom-left
    ])
    return mesh
