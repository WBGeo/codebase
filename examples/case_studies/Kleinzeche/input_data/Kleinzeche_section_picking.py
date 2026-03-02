"""
Kleinzeche — interactive cross-section contact picking
=======================================================
Picks formation contacts from Profile A and Profile B by clicking directly on
the images in a matplotlib window.  Results are cached in section_picks.json
so you can resume or re-pick individual formations at any time.

Workflow
--------
1. Run the script (or call run_picking() from a cell).
2. A matplotlib window opens for each formation in each section.
   - Click anywhere along the visible contact / formation boundary.
   - Right-click to undo the last point.
   - Press Enter to confirm and move to the next formation.
3. Already-picked formations are shown as overlays so you have context.
4. After picking, call picks_to_dataframes() to get surface_points and
   orientations DataFrames ready for InputData_StructuralElements.
5. Call preview_picks() at any time to review what has been picked so far.

Re-picking
----------
Delete individual formation entries from section_picks.json to re-pick them,
or delete the whole file to start fresh.
"""

import json
import os

import matplotlib
matplotlib.use('TkAgg')   # force a standalone interactive window (required for ginput)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Kleinzeche_shared import (
    SECTION_DIR, INPUT_DATA_DIR, CACHE_DIR,
    PATH_A, PATH_B, MARKER_CACHE, PICKS_CACHE,
    BOREHOLES, Z_MIN_A, Z_MIN_B, FORMATION_COLORS,
    find_data_bbox,
)

# ── Formation lists and display colours ───────────────────────────────────────
# Points mark the TOP of each formation (= bottom of the overlying unit).
# Orientations for HORIZONTAL_FORMATIONS are hardcoded as (0,0,1).
# All others get their 3D normal computed from the slope of the picked points.

HORIZONTAL_FORMATIONS = {'Gravel-1', 'Loose-1'}

FORMATIONS_A = [
    'Gravel-1',   # flat, grey cap
    'Loose-1',    # flat green; left side slightly lower due to Fault-1 offset
    'Silt-1',     # first muddy-green band (top)
    'Sand-1',     # first orange band
    'Silt-2',
    'Sand-2',
    'Silt-3',
    'Sand-3',     # deepest visible orange band in Profile A
]

FORMATIONS_B = [
    'Gravel-1',
    'Loose-1',
    'Silt-1',
    'Sand-1',
    'Silt-2',
    'Sand-2',
    'Silt-3',
    'Sand-3',
    'Clay-1',     # top purple band — pick its bottom contact
    'Coal-1',     # thin dark-brown band
    'Clay-2',     # lower purple band — pick its bottom contact
]

# ── Georeferencing ────────────────────────────────────────────────────────────

def _load_marker_pixels() -> dict:
    with open(MARKER_CACHE) as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items()}


class SectionParams:
    """
    Georeferencing parameters for one cross section.

    Converts pixel (px, py) → UTM 3D point and computes the 3D upward normal
    to a layer from a list of picked UTM points.
    """

    def __init__(self, bh1_utm, bh2_utm, px_bh1, px_bh2, z_min, data_bbox):
        px_left, px_right, px_top, px_bottom = data_bbox

        d_bh           = np.linalg.norm(bh2_utm[:2] - bh1_utm[:2])
        self.h_scale   = d_bh / (px_bh2[0] - px_bh1[0])       # m per pixel (horizontal)
        self.direction = (bh2_utm[:2] - bh1_utm[:2]) / d_bh    # unit vec bh1→bh2 (horizontal)

        self.bh1_utm  = bh1_utm
        self.px_bh1   = px_bh1
        self.z_bh1    = float(bh1_utm[2])
        self.py_bh1   = float(px_bh1[1])                        # pixel row of bh1 collar

        # m per pixel vertically (positive: higher pixel row = lower elevation)
        self.v_scale = (self.z_bh1 - z_min) / (px_bottom - self.py_bh1)

        # Along-strike unit vector: perpendicular to section direction in horizontal plane
        self.along_strike = np.array([-self.direction[1], self.direction[0], 0.0])

    def pixel_to_utm(self, px: float, py: float) -> np.ndarray:
        """Convert an image pixel position to a UTM 3D coordinate."""
        s  = (px - self.px_bh1[0]) * self.h_scale
        xy = self.bh1_utm[:2] + s * self.direction
        z  = self.z_bh1 + (self.py_bh1 - py) * self.v_scale
        return np.array([xy[0], xy[1], z])

    def upward_normal_from_points(self, utm_pts: list) -> np.ndarray:
        """
        Compute the mean upward-pointing unit normal to a layer from ≥2 UTM
        points picked along the contact.

        The normal is derived by crossing the mean dip vector (in the section
        plane) with the along-strike direction.  Returns (0,0,1) for a single
        point or degenerate geometry.
        """
        if len(utm_pts) < 2:
            return np.array([0.0, 0.0, 1.0])

        normals = []
        for i in range(len(utm_pts) - 1):
            dv = np.array(utm_pts[i + 1]) - np.array(utm_pts[i])
            mag = np.linalg.norm(dv)
            if mag < 1e-6:
                continue
            dv /= mag
            n = np.cross(dv, self.along_strike)
            if n[2] < 0:
                n = -n
            normals.append(n / np.linalg.norm(n))

        if not normals:
            return np.array([0.0, 0.0, 1.0])

        mean_n = np.mean(normals, axis=0)
        return mean_n / np.linalg.norm(mean_n)


def _build_section_params():
    """Build SectionParams for Profile A and Profile B."""
    markers = _load_marker_pixels()
    bbox_A  = find_data_bbox(PATH_A)
    bbox_B  = find_data_bbox(PATH_B)

    params_A = SectionParams(
        BOREHOLES['MP1'], BOREHOLES['MO1'],
        markers['MP1'], markers['MO1'],
        Z_MIN_A, bbox_A,
    )
    params_B = SectionParams(
        BOREHOLES['O4'], BOREHOLES['O3'],
        markers['O4'], markers['O3'],
        Z_MIN_B, bbox_B,
    )
    return params_A, params_B


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_picks() -> dict:
    if os.path.exists(PICKS_CACHE):
        with open(PICKS_CACHE) as f:
            return json.load(f)
    return {'A': {}, 'B': {}}


def _save_picks(picks: dict):
    with open(PICKS_CACHE, 'w') as f:
        json.dump(picks, f, indent=2)


# ── Interactive picking ───────────────────────────────────────────────────────

def _pick_section(section_label: str, img_path: str, formations: list,
                  picks: dict) -> dict:
    """
    Open the section image and pick contact points for each formation.
    Already-cached formations are shown as overlays and skipped automatically.
    Saves to cache after every formation so progress is never lost.
    """
    from PIL import Image as _PILImage
    img = np.array(_PILImage.open(img_path).convert('RGB'))
    section_picks = picks.get(section_label, {})

    for fm in formations:
        if fm in section_picks and len(section_picks[fm]) > 0:
            print(f"  [Profile {section_label}]  {fm:15s}  "
                  f"{len(section_picks[fm])} pts cached — skipping")
            continue

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.imshow(img, origin='upper')
        ax.set_title(
            f"Profile {section_label}  ·  Formation: {fm}\n"
            f"Click along the {fm} contact  —  right-click to undo  —  Enter to confirm",
            fontsize=11, pad=10,
        )

        # Overlay already-picked formations as context
        for cached_fm, cached_pts in section_picks.items():
            if not cached_pts:
                continue
            c  = FORMATION_COLORS.get(cached_fm, '#888888')
            xs = [p[0] for p in cached_pts]
            ys = [p[1] for p in cached_pts]
            ax.plot(xs, ys, 'o-', color=c, markersize=3, linewidth=1, alpha=0.5)
            ax.text(xs[0], ys[0], cached_fm, fontsize=7, color=c,
                    va='bottom', ha='left', clip_on=True)

        # Highlight the current formation in the legend
        patch = mpatches.Patch(color=FORMATION_COLORS.get(fm, '#333333'), label=fm)
        ax.legend(handles=[patch], loc='upper right', fontsize=10)
        plt.tight_layout()

        raw_pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close(fig)

        pixel_pts = [[float(p[0]), float(p[1])] for p in raw_pts]
        section_picks[fm] = pixel_pts
        picks[section_label] = section_picks
        _save_picks(picks)
        print(f"  [Profile {section_label}]  {fm:15s}  {len(pixel_pts)} pts saved")

    picks[section_label] = section_picks
    return picks


def run_picking():
    """
    Run the full interactive picking session for both sections.
    Already-cached formations are skipped automatically.
    """
    picks = _load_picks()

    print("=== Profile A ===")
    picks = _pick_section('A', PATH_A, FORMATIONS_A, picks)

    print("\n=== Profile B ===")
    picks = _pick_section('B', PATH_B, FORMATIONS_B, picks)

    print("\nPicking complete. Call picks_to_dataframes() to build the DataFrames.")
    return picks


# ── Convert picks → DataFrames ────────────────────────────────────────────────

def picks_to_dataframes() -> tuple:
    """
    Convert cached pixel picks to surface_points and orientations DataFrames.

    Returns
    -------
    surface_points : pd.DataFrame   columns: X, Y, Z, formation
    orientations   : pd.DataFrame   columns: X, Y, Z, G_x, G_y, G_z, formation
    """
    picks    = _load_picks()
    params_A, params_B = _build_section_params()

    section_map = {
        'A': (params_A, FORMATIONS_A),
        'B': (params_B, FORMATIONS_B),
    }

    sp_rows  = []
    ori_rows = []

    for section_label, (params, formations) in section_map.items():
        section_picks = picks.get(section_label, {})

        for fm in formations:
            pixel_pts = section_picks.get(fm, [])
            if not pixel_pts:
                continue

            # Pixel → UTM
            utm_pts = [params.pixel_to_utm(p[0], p[1]) for p in pixel_pts]

            # Surface points — one row per picked pixel
            for pt in utm_pts:
                sp_rows.append({
                    'X': float(pt[0]), 'Y': float(pt[1]),
                    'Z': float(pt[2]), 'formation': fm,
                })

            # One orientation at the centroid of the picked cluster
            centroid = np.mean(utm_pts, axis=0)
            if fm in HORIZONTAL_FORMATIONS:
                gx, gy, gz = 0.0, 0.0, 1.0
            else:
                n = params.upward_normal_from_points(utm_pts)
                gx, gy, gz = float(n[0]), float(n[1]), float(n[2])

            ori_rows.append({
                'X': float(centroid[0]), 'Y': float(centroid[1]),
                'Z': float(centroid[2]),
                'G_x': gx, 'G_y': gy, 'G_z': gz,
                'formation': fm,
            })

    surface_points = pd.DataFrame(sp_rows)
    orientations   = pd.DataFrame(ori_rows)
    return surface_points, orientations


# ── Preview ───────────────────────────────────────────────────────────────────

def preview_picks():
    """Show all cached picks overlaid on both sections side-by-side."""
    picks = _load_picks()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, section_label, img_path in [
        (axes[0], 'A', PATH_A),
        (axes[1], 'B', PATH_B),
    ]:
        from PIL import Image as _PILImage
        img = np.array(_PILImage.open(img_path).convert('RGB'))
        ax.imshow(img, origin='upper')
        ax.set_title(f'Profile {section_label} — picked contacts', fontsize=11)

        section_picks = picks.get(section_label, {})
        for fm, pts in section_picks.items():
            if not pts:
                continue
            c  = FORMATION_COLORS.get(fm, '#888888')
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, 'o-', color=c, markersize=5, linewidth=1.5, label=fm)
            ax.text(xs[0], ys[0], fm, fontsize=7, color=c, va='bottom', clip_on=True)

        ax.legend(fontsize=7, loc='lower right', ncol=2)

    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_picking()
    preview_picks()

    surface_points, orientations = picks_to_dataframes()
    print(f"\nSurface points : {len(surface_points)} rows")
    print(surface_points.to_string())
    print(f"\nOrientations   : {len(orientations)} rows")
    print(orientations.to_string())
