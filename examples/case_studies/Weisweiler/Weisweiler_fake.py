# Synthetic Weisweiler-inspired elongated syncline (no faults)
#
# Simplified from the Weisweiler geological model (screenshot reference).
# Six elements in one stratigraphic group (youngest → oldest):
#
#   Claystone          — combined claystone / siltstone            (youngest, top)
#   Carbonate_upper    — combined younger carbonates
#   Siltstone_upper    — combined younger siltstone / sandstone
#   Schist             — clay-schist unit (middle)
#   Carbonate_lower    — combined older carbonates
#   Siltstone_lower    — combined older siltstone / sandstone      (oldest, base)
#
# Structural setting: elongated periclinal syncline (no faults).
#   - Fold axis parallel to X; fold deepens toward the centre and pinches out
#     toward both ends (periclinal closure).
#   - Gaussian cross-section shape produces steep, narrow flanks (~40° dip).
#   - Slight regional tilt along X adds along-strike variety.
#   - Carbonate units are intentionally thinner than the clastic units.
#
# Domain: X ∈ [0, 8000] m  (along strike)
#         Y ∈ [0, 6000] m  (across strike)
#         Z ∈ [-2500, 500] m

import numpy as np
import pandas as pd
import os

from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.object_components import InputData_StructuralElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D,
    plot_structural_model_3D,
)
from core.structural_modeling_components import general

cwd = os.getcwd()

# ── Syncline geometry ─────────────────────────────────────────────────────────
#
# Gaussian cross-section (across strike, Y):
#   z(x, y) = Z_FLANK[fm]
#             - A(x) * exp(-0.5 * ((y - Y_CENTER) / SIGMA_Y)^2)   ← Gaussian syncline
#             - TILT_X * x                                          ← regional tilt
#
# Using a Gaussian rather than a parabola ensures:
#   (a) the fold approaches zero smoothly at the domain edges (no negative fold shape),
#   (b) the narrow sigma produces steep (~40°) flanks in the central zone.
#
# Periclinal closure along strike (X):
#   A(x) = A_MAX * max(0, 1 - ((x - X_CENTER) / X_HALF)^2)
#   → amplitude is zero at x = 0 and x = 8000, maximum at x = 4000.

Y_CENTER = 3000.0   # hinge Y position (m)
SIGMA_Y  =  900.0   # Gaussian half-width (m); smaller = narrower and steeper flanks

X_CENTER = 4000.0
X_HALF   = 4000.0
A_MAX    = 1200.0   # maximum fold amplitude at the hinge (m)

TILT_X   =  0.015   # regional dip along X (m per m)


def a_at_x(x: float) -> float:
    """Fold amplitude at along-strike position x (periclinal closure)."""
    t = (x - X_CENTER) / X_HALF
    return float(A_MAX * max(0.0, 1.0 - t ** 2))


def da_dx(x: float) -> float:
    """Derivative of fold amplitude with respect to x."""
    t = (x - X_CENTER) / X_HALF
    if abs(t) >= 1.0:
        return 0.0
    return float(-A_MAX * 2.0 * t / X_HALF)


# Top-surface elevation of each formation far from the fold (flank reference).
# Carbonate units are intentionally thin; clastic units are thicker.
#
#   Claystone:        200  →  -100   = 300 m thick
#   Carbonate_upper:  -100 →  -250   = 150 m thick  (thin)
#   Siltstone_upper:  -250 →  -600   = 350 m thick
#   Schist:           -600 →  -850   = 250 m thick
#   Carbonate_lower:  -850 → -1000   = 150 m thick  (thin)
#   Siltstone_lower: -1000 → model base              (thick)
Z_FLANK = {
    "Claystone":        200.0,
    "Carbonate_upper": -100.0,
    "Siltstone_upper": -250.0,
    "Schist":          -600.0,
    "Carbonate_lower": -850.0,
    "Siltstone_lower": -1000.0,
}


def gaussian(y: float) -> float:
    return float(np.exp(-0.5 * ((y - Y_CENTER) / SIGMA_Y) ** 2))


def z_interface(x: float, y: float, formation: str) -> float:
    """Elevation of a formation's top interface at position (x, y)."""
    return float(Z_FLANK[formation] - a_at_x(x) * gaussian(y) - TILT_X * x)


def upward_normal(x: float, y: float):
    """
    Analytically derived upward-pointing unit normal to the fold surface at (x, y).

        dz/dx = -dA/dx * G(y)  -  TILT_X
        dz/dy =  A(x) * (y - Y_CENTER) / SIGMA_Y^2  *  G(y)

    where G(y) = exp(-0.5 * ((y - Y_CENTER) / SIGMA_Y)^2)

    Normal (unnormalised) = (-dz/dx, -dz/dy, 1).
    """
    g = gaussian(y)
    dz_dx = -da_dx(x) * g - TILT_X
    dz_dy = a_at_x(x) * (y - Y_CENTER) / (SIGMA_Y ** 2) * g
    nx, ny, nz = -dz_dx, -dz_dy, 1.0
    mag = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return float(nx / mag), float(ny / mag), float(nz / mag)


# ── Grid ──────────────────────────────────────────────────────────────────────

grid = RegularGrid(
    extent=(0, 8000, 0, 6000, -2500, 500),
    resolution=(100, 75, 50),
)

# ── Formation stack (youngest → oldest) ───────────────────────────────────────

FORMATIONS = (
    "Claystone",        # youngest — top unit in image
    "Carbonate_upper",
    "Siltstone_upper",
    "Schist",           # clay-schist middle unit
    "Carbonate_lower",
    "Siltstone_lower",  # oldest — base unit
)

# ── Surface points ────────────────────────────────────────────────────────────
# Y positions vary with X to reproduce the periclinal plan-view shape:
#   - At the fold centre (x = 4000): points span Y ∈ [~1050, ~4950] (widest)
#   - At the fold ends  (x =  500 / 7500): points span Y ∈ [~1800, ~4200] (narrowest)
# Outside the point footprint the interpolator extrapolates into basement,
# so Siltstone_lower crops out at both Y extremes and at the fold ends.
#
# 4 along-strike (X) × 4 across-strike (Y) positions per formation
# → 16 pts per formation × 6 formations = 96 total

X_SP = [500.0, 2000.0, 5000.0, 7500.0]


def y_spread(x: float) -> float:
    """
    Half-span of the Y point footprint at a given X.
    Scales with fold amplitude: widest at the fold centre, narrowest at the ends.
      x = 4000 → spread = 1300 m  (Y range ~ [1050, 4950])
      x =  500 / 7500 → spread ~  800 m  (Y range ~ [1800, 4200])
    """
    fold_fraction = a_at_x(x) / A_MAX          # 1.0 at centre, ~0 at ends
    return 800.0 + 500.0 * fold_fraction


def y_positions(x: float) -> list:
    """Four Y positions symmetric about Y_CENTER for surface points at x."""
    ys = y_spread(x)
    return [
        Y_CENTER - 1.5 * ys,
        Y_CENTER - 0.5 * ys,
        Y_CENTER + 0.5 * ys,
        Y_CENTER + 1.5 * ys,
    ]


sp_rows = []
for fm in FORMATIONS:
    for x in X_SP:
        for y in y_positions(x):
            sp_rows.append({"X": x, "Y": y, "Z": z_interface(x, y, fm), "formation": fm})

surface_points = pd.DataFrame(sp_rows)   # 96 points total

# ── Orientations ──────────────────────────────────────────────────────────────
# 3 positions per formation, all inside the active fold footprint, capturing
# both periclinal plunge (X gradient) and across-fold dip (Y gradient):
#   (x=1000, y=2000) — near left closure, left-flank of fold
#   (x=4000, y=4500) — fold centre, right-flank (steep zone)
#   (x=7000, y=4000) — near right closure, right-flank of fold
# → 3 per formation × 6 formations = 18 total

ORI_POSITIONS = [(1000.0, 2000.0), (4000.0, 4500.0), (7000.0, 4000.0)]

ori_rows = []
for fm in FORMATIONS:
    for x, y in ORI_POSITIONS:
        gx, gy, gz = upward_normal(x, y)
        ori_rows.append({
            "X": x, "Y": y, "Z": z_interface(x, y, fm),
            "G_x": gx, "G_y": gy, "G_z": gz,
            "formation": fm,
        })

orientations = pd.DataFrame(ori_rows)   # 18 orientations total

# ── Input data ────────────────────────────────────────────────────────────────

data_elements = InputData_StructuralElements(
    name="Weisweiler_fake",
    mapping_object={
        "Strat_Series1": FORMATIONS,
    },
    surface_points=surface_points,
    orientations=orientations,
)

# ── Build structural frame ────────────────────────────────────────────────────

frame = general.build_structural_frame(
    input_data_elements=data_elements,
    grid=grid,
)

# ── Assign colors matching the Weisweiler screenshot legend ──────────────────

frame["Strat_Series1"]["Claystone"].set_color("#C8A42D")        # golden yellow
frame["Strat_Series1"]["Carbonate_upper"].set_color("#89C4E0")  # light blue
frame["Strat_Series1"]["Siltstone_upper"].set_color("#C4956A")  # warm tan
frame["Strat_Series1"]["Schist"].set_color("#5E6E7E")           # dark grey-blue
frame["Strat_Series1"]["Carbonate_lower"].set_color("#6BAED6")  # medium blue
frame["Strat_Series1"]["Siltstone_lower"].set_color("#8B6340")  # dark brown

frame.detailed_report()

#%%

# Plot input data
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation method
# frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed
# frame["Strat_Series1"].configure_interpolation_params()

#%%

# Compute structural model
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# Visualize results
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

#%%

# Derive refinement values from the known flank thicknesses.
# Layer order (bottom → top): Siltstone_lower | Carbonate_lower | Schist |
#   Siltstone_upper | Carbonate_upper | Claystone | model cap
# One refinement value per gap: n_surfaces + 1 = 7 values.
# Target ~50 m per cell; clamp to [3, 20] so the thick base unit stays manageable.
_min_z, _max_z = grid.extent[4], grid.extent[5]
_layer_tops = [
    Z_FLANK["Siltstone_lower"],
    Z_FLANK["Carbonate_lower"],
    Z_FLANK["Schist"],
    Z_FLANK["Siltstone_upper"],
    Z_FLANK["Carbonate_upper"],
    Z_FLANK["Claystone"],
    _max_z,
]
_layer_bottoms = [_min_z] + _layer_tops[:-1]
_TARGET_CELL_M = 50.0
_refinement_data = tuple(
    min(20, max(3, round((top - bot) / _TARGET_CELL_M)))
    for bot, top in zip(_layer_bottoms, _layer_tops)
)

print("Refinement values (bottom → top):", _refinement_data)

# Explicit Structured meshing
mesh_result = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=_refinement_data,
    z_threshold=0.1,
    tolerance=1
)

# Explicit Unstructured meshing
# mesh_result = create_unstructured_mesh_data(
#     geomodel_result=structural_model_result,
#     z_threshold=0.1,
#     tolerance=1
# )

#%%

# Plot the meshing results
plot_mesh_3d(mesh_result, structural_model_result, show_plotter=True)

#%%

# Export mesh to gmsh format
mesh_result.export_gmsh("C:/Users/vonha/Desktop/Weisweiler_fake.gmsh")

