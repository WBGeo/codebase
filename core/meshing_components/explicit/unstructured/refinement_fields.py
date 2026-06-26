from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement,
    LinearWellRefinement,
    FunctionWellRefinement,
    LinearSourceRefinement,
    FunctionSourceRefinement,
)
import gmsh
import numpy as np
from scipy.spatial import cKDTree

class FieldIDGenerator:
    """
    Generates unique Gmsh field IDs.
    This utility ensures that every mesh refinement field receives
    a unique identifier when multiple refinement strategies are
    combined in the same model.
    """

    def __init__(self):
        self._id = 0

    def next(self):
        self._id += 1
        return self._id


def build_well_refinement(zip_info, well_lines, refinement, field_gen):
    """
    Creates mesh refinement fields around well trajectories.
    Supports both:
    - LinearWellRefinement (threshold-based refinement)
    - FunctionWellRefinement (user-defined mathematical expressions)

    The refinement is applied to all child curves generated after
    geometry fragmentation.
    """
    if not refinement or refinement.wells is None:
        return None

    w = refinement.wells

    # Flatten original well tags
    well_tags = set()
    for wl in well_lines:
        well_tags.update(wl)

    well_child_curves = []

    for parent, children in zip_info:
        dim, tag = parent

        # Keep only original well lines
        if dim != 1 or tag not in well_tags:
            continue

        for cdim, ctag in children:
            if cdim == 1:
                well_child_curves.append(ctag)

    well_child_curves = list(set(well_child_curves))

    if not well_child_curves:
        return None

    # Distance field
    fid_dist = field_gen.next()
    gmsh.model.mesh.field.add("Distance", fid_dist)
    gmsh.model.mesh.field.setNumbers(
        fid_dist,
        "CurvesList",
        well_child_curves,
    )
    gmsh.model.mesh.field.setNumber(
        fid_dist,
        "Sampling",
        2000,
    )

    # Linear refinement
    if isinstance(w, LinearWellRefinement):
        fid = field_gen.next()

        gmsh.model.mesh.field.add("Threshold", fid)
        gmsh.model.mesh.field.setNumber(fid, "InField", fid_dist)
        gmsh.model.mesh.field.setNumber(fid, "SizeMin", w.SizeMin)
        gmsh.model.mesh.field.setNumber(fid, "SizeMax", w.SizeMax)
        gmsh.model.mesh.field.setNumber(fid, "DistMin", w.DistMin)
        gmsh.model.mesh.field.setNumber(fid, "DistMax", w.DistMax)

        return fid

    # Function refinement
    elif isinstance(w, FunctionWellRefinement):
        fid = field_gen.next()

        gmsh.model.mesh.field.add("MathEval", fid)

        expr = w.expression.replace(
            "DIST",
            f"F{fid_dist}",
        )
        gmsh.model.mesh.field.setString(fid, "F", expr)

        return fid

    else:
        raise TypeError(type(w))


def build_source_refinement(source_points, refinement, field_gen):
    """
    Creates mesh refinement fields around source points.

    Supports:
    - LinearSourceRefinement
    - FunctionSourceRefinement

    Uses a distance field to control mesh size near source locations.
    """
    if not refinement or refinement.sources is None:
        return None

    if not source_points:
        return None

    s = refinement.sources

    fid_dist = field_gen.next()
    gmsh.model.mesh.field.add("Distance", fid_dist)
    gmsh.model.mesh.field.setNumbers(fid_dist, "NodesList", source_points)
    gmsh.model.mesh.field.setNumber(fid_dist, "Sampling", 2000)

    # Linear refinement
    if isinstance(s, LinearSourceRefinement):
        fid = field_gen.next()

        gmsh.model.mesh.field.add("Threshold", fid)
        gmsh.model.mesh.field.setNumber(fid, "InField", fid_dist)
        gmsh.model.mesh.field.setNumber(fid, "SizeMin", s.SizeMin)
        gmsh.model.mesh.field.setNumber(fid, "SizeMax", s.SizeMax)
        gmsh.model.mesh.field.setNumber(fid, "DistMin", s.DistMin)
        gmsh.model.mesh.field.setNumber(fid, "DistMax", s.DistMax)

        return fid

    # Function refinement
    elif isinstance(s, FunctionSourceRefinement):
        fid = field_gen.next()

        gmsh.model.mesh.field.add("MathEval", fid)

        expr = s.expression.replace("DIST", f"F{fid_dist}")
        gmsh.model.mesh.field.setString(fid, "F", expr)

        return fid

    else:
        raise TypeError(type(s))


def build_triangulation_mesh_callback(triangulations, hmin=3.0, hmax=30.0, d1=50.0, d2=100.0,):
    """
    Creates a mesh size callback based on triangulation points.
    A KD-tree is used to compute the distance from any mesh node
    to the nearest triangulation point and adjust mesh size
    according to distance thresholds.
    """
    triangulations_arr = np.asarray(triangulations)

    # collect points
    if triangulations_arr.shape[1] == 3:
        all_points = triangulations_arr
    else:
        tri_groups = {}

        for row in triangulations_arr:
            x, y, z, pid = row
            tri_groups.setdefault(int(pid), []).append([x, y, z])

        all_points = np.vstack(
            [np.asarray(v, dtype=float) for v in tri_groups.values()]
        )

    # KDTree
    tree = cKDTree(all_points)

    # callback
    def mesh_size_callback(dim, tag, x, y, z, lc):
        dist, _ = tree.query([x, y, z])

        if dist < d1:
            return hmin
        elif dist < d2:
            t = (dist - d1) / (d2 - d1)
            return hmin * (1 - t) + hmax * t
        else:
            return lc

    return mesh_size_callback


def build_ellipse_mesh_callback(ellipses, hmin=3.0, hmax=30.0, d1=30.0, d2=80.0,):
    """
    Creates a mesh size callback around elliptical regions.
    Mesh size is controlled by distance from ellipse centers,
    enabling refinement near features and coarsening away.
    """
    if not ellipses:
        return None

    centers = []
    scales = []

    for e in ellipses:
        cx, cy, cz = e["center"]
        r1, r2 = e["radii"]

        centers.append([cx, cy, cz])
        scales.append(max(r1, r2))

    centers = np.asarray(centers, dtype=float)
    scales = np.asarray(scales, dtype=float)

    tree = cKDTree(centers)

    def mesh_size_callback(dim, tag, x, y, z, lc):
        dist_center, idx = tree.query([x, y, z])

        dist = max(0.0, dist_center - scales[idx])

        if dist < d1:
            return hmin
        elif dist < d2:
            t = (dist - d1) / (d2 - d1)
            return hmin * (1.0 - t) + hmax * t
        else:
            return lc

    return mesh_size_callback


def build_fault_mesh_callback_from_fragments(fault_fragments, hmin=3.0, hmax=30.0, d1=20.0, d2=60.0,):
    """
    Creates a fault-based mesh refinement callback using OCC geometry.
    Surface points are sampled from fault fragments and interpolated
    using RBF interpolation. A KD-tree is then used to refine the mesh
    near fault surfaces without relying on an existing mesh.
    """
    if not fault_fragments:
        return None

    fault_points = []

    for dim, tag in fault_fragments:
        if dim != 2:
            continue

        try:
            from scipy.interpolate import RBFInterpolator

            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)

            bounds = gmsh.model.getParametrizationBounds(2, tag)
            umin, umax = bounds[0]
            vmin, vmax = bounds[1]

            surface_pts = []

            for u in np.linspace(umin, umax, 20):
                for v in np.linspace(vmin, vmax, 20):
                    surface_pts.append(
                        gmsh.model.getValue(2, tag, [u, v])
                    )

            surface_pts = np.asarray(surface_pts)

            rbf = RBFInterpolator(
                surface_pts[:, :2],
                surface_pts[:, 2],
                kernel="thin_plate_spline",
            )

            xs = np.linspace(xmin, xmax, 40)
            ys = np.linspace(ymin, ymax, 40)

            XX, YY = np.meshgrid(xs, ys)
            XY = np.column_stack([XX.ravel(), YY.ravel()])
            ZZ = rbf(XY)

            fault_points.extend(
                np.column_stack([XY, ZZ]).tolist()
            )

        except Exception as e:
            print(f"Skipping fault surface {tag}: {e}")
            continue

    fault_points = np.asarray(fault_points, dtype=float)

    if fault_points.shape[0] == 0:
        return None

    tree = cKDTree(fault_points)

    def mesh_size_callback(dim, tag, x, y, z, lc):
        dist, _ = tree.query([x, y, z])

        if dist < d1:
            return hmin
        elif dist < d2:
            t = (dist - d1) / (d2 - d1)
            return hmin * (1.0 - t) + hmax * t
        else:
            return lc

    return mesh_size_callback


def build_combined_mesh_callback(triangulations=None, ellipses=None, fault_surfaces=None, fault_fragments=None,
    tri_cfg=None, ell_cfg=None, fault_cfg=None,):
    """
    Combines triangulation, ellipse, and fault-based mesh refinements.
    Multiple refinement callbacks can be activated simultaneously.
    The minimum mesh size returned by any active refinement strategy
    is used as the final mesh size.
    """
    tri_cb = None
    ell_cb = None
    fault_cb = None

    # fault callback (USE FRAGMENTS)
    if (
        fault_fragments is not None
        and len(fault_fragments) > 0
        and fault_cfg is not None
        and fault_cfg.enabled
    ):
        fault_cb = build_fault_mesh_callback_from_fragments(
            fault_fragments,
            hmin=fault_cfg.hmin,
            hmax=fault_cfg.hmax,
            d1=fault_cfg.d1,
            d2=fault_cfg.d2,
        )

    # triangulation callback
    if (
        triangulations is not None
        and len(triangulations) > 0
        and tri_cfg is not None
        and tri_cfg.enabled
    ):
        tri_cb = build_triangulation_mesh_callback(
            triangulations,
            hmin=tri_cfg.hmin,
            hmax=tri_cfg.hmax,
            d1=tri_cfg.d1,
            d2=tri_cfg.d2,
        )

    # ellipse callback
    if (
        ellipses is not None
        and len(ellipses) > 0
        and ell_cfg is not None
        and ell_cfg.enabled
    ):
        ell_cb = build_ellipse_mesh_callback(
            ellipses,
            hmin=ell_cfg.hmin,
            hmax=ell_cfg.hmax,
            d1=ell_cfg.d1,
            d2=ell_cfg.d2,
        )

    if tri_cb is None and ell_cb is None and fault_cb is None:
        return None

    def mesh_size_callback(dim, tag, x, y, z, lc):
        values = [lc]

        if tri_cb is not None:
            values.append(tri_cb(dim, tag, x, y, z, lc))

        if ell_cb is not None:
            values.append(ell_cb(dim, tag, x, y, z, lc))

        if fault_cb is not None:
            values.append(fault_cb(dim, tag, x, y, z, lc))

        return min(values)

    return mesh_size_callback


def build_refinement_fields(zip_info, well_lines, source_points, triangulations=None, refinement=None,):
    """
    Builds all active Gmsh background refinement fields.
    Generates refinement fields for:
    - Wells
    - Sources

    Returns a list of active field IDs to be used as background
    mesh controls.
    """
    field_gen = FieldIDGenerator()
    active_fields = []

    well_field = build_well_refinement(
        zip_info,
        well_lines,
        refinement,
        field_gen,
    )

    source_field = build_source_refinement(
        source_points,
        refinement,
        field_gen,
    )

    if well_field is not None:
        active_fields.append(well_field)

    if source_field is not None:
        active_fields.append(source_field)

    return active_fields


def apply_background_fields(active_fields):
    """
    Applies refinement fields as the Gmsh background mesh.

    If multiple fields are active, they are combined using a
    'Min' field so that the smallest mesh size requirement
    dominates locally.
    """
    if not active_fields:
        return

    if len(active_fields) == 1:
        gmsh.model.mesh.field.setAsBackgroundMesh(active_fields[0])
        return

    fid = max(active_fields) + 1

    gmsh.model.mesh.field.add("Min", fid)
    gmsh.model.mesh.field.setNumbers(fid, "FieldsList", active_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(fid)
