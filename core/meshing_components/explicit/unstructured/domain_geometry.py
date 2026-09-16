"""Domain geometry helpers for axis-aligned and arbitrarily oriented boxes.

A domain can be supplied in either form:

1. Legacy axis-aligned extent::

       [xmin, xmax, ymin, ymax, zmin, zmax]

2. Eight corners of an oriented rectangular/parallelepiped box::

       [p0, p1, p2, p3, p4, p5, p6, p7]

   with the conventional ordering

       bottom: p0, p1, p2, p3
       top:    p4, p5, p6, p7

   and edges p0->p1, p0->p3, p0->p4 defining the three local axes.

The 8-corner form is intentionally restricted to a parallelepiped (including
rotated rectangular boxes). This keeps point-in-domain tests and boundary
classification well defined.
"""
from __future__ import annotations

from typing import Any, Tuple
import numpy as np


def normalize_domain(domain: Any) -> Tuple[str, np.ndarray]:
    if domain is None:
        raise ValueError("Domain/extent is None")
    arr = np.asarray(domain, dtype=float)
    if arr.ndim == 1 and arr.size == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = arr.tolist()
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("Extent minima must be smaller than maxima")
        return "extent", arr.reshape(6)
    if arr.size == 24:
        corners = arr.reshape(8, 3)
        _validate_corners(corners)
        return "corners", corners
    raise ValueError(
        "Domain must be either [xmin, xmax, ymin, ymax, zmin, zmax] "
        "or 8 corner coordinates with shape (8, 3) (a flat 24-value list is also accepted)."
    )


def _validate_corners(c: np.ndarray, tol: float = 1e-7) -> None:
    if c.shape != (8, 3) or not np.all(np.isfinite(c)):
        raise ValueError("8-corner domain must have shape (8, 3) and finite coordinates")
    u, v, w = c[1] - c[0], c[3] - c[0], c[4] - c[0]
    scale = max(np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w), 1.0)
    if abs(np.linalg.det(np.column_stack((u, v, w)))) <= tol * scale**3:
        raise ValueError("The 8 corners do not define a non-degenerate 3D box")
    expected = np.vstack([
        c[0],
        c[0] + u,
        c[0] + u + v,
        c[0] + v,
        c[0] + w,
        c[0] + u + w,
        c[0] + u + v + w,
        c[0] + v + w,
    ])
    if not np.allclose(c, expected, rtol=tol, atol=tol * scale):
        raise ValueError(
            "The 8 corners must describe a parallelepiped and use ordering "
            "[p0,p1,p2,p3] on the bottom and [p4,p5,p6,p7] on the top."
        )


def domain_aabb(domain: Any) -> np.ndarray:
    kind, data = normalize_domain(domain)
    if kind == "extent":
        return data.copy()
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=float)


def domain_local_coordinates(points: Any, domain: Any) -> np.ndarray:
    """Return normalized local coordinates; inside means each coordinate is in [0, 1]."""
    pts = np.asarray(points, dtype=float)
    one = pts.ndim == 1
    pts = np.atleast_2d(pts)
    kind, data = normalize_domain(domain)
    if kind == "extent":
        xmin, xmax, ymin, ymax, zmin, zmax = data
        origin = np.array([xmin, ymin, zmin])
        basis = np.diag([xmax - xmin, ymax - ymin, zmax - zmin])
    else:
        origin = data[0]
        basis = np.column_stack((data[1] - data[0], data[3] - data[0], data[4] - data[0]))
    local = np.linalg.solve(basis, (pts - origin).T).T
    return local[0] if one else local


def points_inside_domain(points: Any, domain: Any, tol: float = 1e-9) -> np.ndarray:
    local = np.atleast_2d(domain_local_coordinates(points, domain))
    return np.all((local >= -tol) & (local <= 1.0 + tol), axis=1)


def point_inside_domain(point: Any, domain: Any, tol: float = 1e-9) -> bool:
    return bool(points_inside_domain(np.asarray(point, dtype=float).reshape(1, 3), domain, tol)[0])


def classify_local_face(point: Any, domain: Any, tol_ratio: float = 1e-4):
    q = np.asarray(domain_local_coordinates(point, domain), dtype=float)
    names = ((0.0, "XMIN"), (1.0, "XMAX"), (0.0, "YMIN"), (1.0, "YMAX"), (0.0, "ZMIN"), (1.0, "ZMAX"))
    vals = (q[0], q[0], q[1], q[1], q[2], q[2])
    return [name for value, (target, name) in zip(vals, names) if abs(value - target) <= tol_ratio]


def add_gmsh_domain_volume(gmsh, domain: Any) -> int:
    """Create the requested domain volume and return its OCC volume tag."""
    kind, data = normalize_domain(domain)
    if kind == "extent":
        xmin, xmax, ymin, ymax, zmin, zmax = data
        return gmsh.model.occ.addBox(xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin)

    c = data
    p = [gmsh.model.occ.addPoint(*xyz) for xyz in c]
    edges = {
        "01": gmsh.model.occ.addLine(p[0], p[1]), "12": gmsh.model.occ.addLine(p[1], p[2]),
        "23": gmsh.model.occ.addLine(p[2], p[3]), "30": gmsh.model.occ.addLine(p[3], p[0]),
        "45": gmsh.model.occ.addLine(p[4], p[5]), "56": gmsh.model.occ.addLine(p[5], p[6]),
        "67": gmsh.model.occ.addLine(p[6], p[7]), "74": gmsh.model.occ.addLine(p[7], p[4]),
        "04": gmsh.model.occ.addLine(p[0], p[4]), "15": gmsh.model.occ.addLine(p[1], p[5]),
        "26": gmsh.model.occ.addLine(p[2], p[6]), "37": gmsh.model.occ.addLine(p[3], p[7]),
    }
    loops = [
        [edges["01"], edges["12"], edges["23"], edges["30"]],
        [edges["45"], edges["56"], edges["67"], edges["74"]],
        [edges["01"], edges["15"], -edges["45"], -edges["04"]],
        [edges["12"], edges["26"], -edges["56"], -edges["15"]],
        [edges["23"], edges["37"], -edges["67"], -edges["26"]],
        [edges["30"], edges["04"], -edges["74"], -edges["37"]],
    ]
    surfaces = [gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(loop)]) for loop in loops]
    shell = gmsh.model.occ.addSurfaceLoop(surfaces)
    return gmsh.model.occ.addVolume([shell])
