import logging
from typing import Callable, List, Optional
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement,
    LinearWellRefinement,
    FunctionWellRefinement,
    LinearSourceRefinement,
    FunctionSourceRefinement,
    LinearShaftRefinement,
    FunctionShaftRefinement,
    GeneralRefinement,
)
import gmsh
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

MeshSizeCallback = Callable[[int, int, float, float, float, float], float]

class FieldIDGenerator:
    """
    Generates unique Gmsh field IDs.
    This utility ensures that every mesh refinement field receives
    a unique identifier when multiple refinement strategies are
    combined in the same model.
    """

    def __init__(self):
        self._id = 0

    def next(self) -> int:
        self._id += 1
        return self._id


def build_well_refinement(zip_info, well_lines, refinement, field_gen) -> Optional[int]:
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


def build_source_refinement(source_points, refinement, field_gen) -> Optional[int]:
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

def build_shaft_refinement(
    mine_shafts,
    refinement,
    field_gen,
) -> Optional[int]:
    """
    Refines according to perpendicular distance from the nearest
    infinite shaft axis.
    """
    if not refinement or refinement.shafts is None:
        return None

    if not mine_shafts:
        return None

    shaft_cfg = refinement.shafts

    def make_axis_distance_expression(shaft):
        cx, cy, cz = map(float, shaft["center"])
        ax, ay, az = map(float, shaft["axis"])

        axis_norm_squared = ax * ax + ay * ay + az * az

        if axis_norm_squared < 1.0e-20:
            return None

        # Components of (point - center) cross axis.
        cross_x = (
            f"((y-({cy}))*({az})"
            f"-(z-({cz}))*({ay}))"
        )
        cross_y = (
            f"((z-({cz}))*({ax})"
            f"-(x-({cx}))*({az}))"
        )
        cross_z = (
            f"((x-({cx}))*({ay})"
            f"-(y-({cy}))*({ax}))"
        )

        return (
            f"(sqrt("
            f"({cross_x})^2"
            f"+({cross_y})^2"
            f"+({cross_z})^2"
            f")/sqrt({axis_norm_squared}))"
        )

    # ---------------------------------------------------------
    # Function refinement
    # ---------------------------------------------------------
    if isinstance(shaft_cfg, FunctionShaftRefinement):
        shaft_size_fields = []

        for shaft_index, shaft in enumerate(mine_shafts):
            distance_expression = make_axis_distance_expression(shaft)

            if distance_expression is None:
                logger.warning( "Skipping degenerate shaft %d during refinement.", shaft_index, )
                continue

            # Embed the distance expression directly. Do not create
            # a separate MathEval distance field referenced as F#.
            expression = shaft_cfg.expression.replace(
                "DIST",
                distance_expression,
            )

            fid = field_gen.next()
            gmsh.model.mesh.field.add("MathEval", fid)
            gmsh.model.mesh.field.setString( fid, "F", expression, )

            logger.info( "Shaft %d function field %d: %s", shaft_index, fid, expression, )

            shaft_size_fields.append(fid)

        if not shaft_size_fields:
            return None

        if len(shaft_size_fields) == 1:
            return shaft_size_fields[0]

        # The closest shaft normally requests the smallest mesh size.
        fid_min = field_gen.next()
        gmsh.model.mesh.field.add("Min", fid_min)
        gmsh.model.mesh.field.setNumbers( fid_min, "FieldsList", shaft_size_fields, )

        return fid_min

    # ---------------------------------------------------------
    # Linear refinement
    # ---------------------------------------------------------
    if isinstance(shaft_cfg, LinearShaftRefinement):
        distance_fields = []

        for shaft_index, shaft in enumerate(mine_shafts):
            distance_expression = make_axis_distance_expression(shaft)

            if distance_expression is None:
                logger.warning( "Skipping degenerate shaft %d during refinement.", shaft_index, )
                continue

            fid_distance = field_gen.next()
            gmsh.model.mesh.field.add( "MathEval", fid_distance, )
            gmsh.model.mesh.field.setString( fid_distance, "F", distance_expression, )

            distance_fields.append(fid_distance)

        if not distance_fields:
            return None

        if len(distance_fields) == 1:
            fid_nearest_distance = distance_fields[0]
        else:
            fid_nearest_distance = field_gen.next()
            gmsh.model.mesh.field.add( "Min", fid_nearest_distance, )
            gmsh.model.mesh.field.setNumbers( fid_nearest_distance, "FieldsList", distance_fields, )

        fid = field_gen.next()
        gmsh.model.mesh.field.add("Threshold", fid)

        gmsh.model.mesh.field.setNumber( fid, "InField", fid_nearest_distance, )
        gmsh.model.mesh.field.setNumber( fid, "SizeMin", shaft_cfg.SizeMin, )
        gmsh.model.mesh.field.setNumber( fid, "SizeMax", shaft_cfg.SizeMax, )
        gmsh.model.mesh.field.setNumber( fid, "DistMin", shaft_cfg.DistMin, )
        gmsh.model.mesh.field.setNumber( fid, "DistMax", shaft_cfg.DistMax, )

        return fid

    raise TypeError(type(shaft_cfg))


def build_triangulation_mesh_callback(
    triangulations,
    SizeMin=3.0,
    SizeMax=30.0,
    DistMin=50.0,
    DistMax=100.0,
) -> MeshSizeCallback:

    triangulations_arr = np.asarray(triangulations, dtype=float)

    # Collect XYZ coordinates
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

    # KD-tree containing ALL triangulation vertices
    tree = cKDTree(all_points)

    def mesh_size_callback(dim, tag, x, y, z, lc):
        point = np.array([x, y, z])

        dist, _ = tree.query(point)

        # Exactly at / very close to triangulation point
        if dist <= DistMin:
            return SizeMin

        # Transition region
        if dist < DistMax:
            t = (dist - DistMin) / (DistMax - DistMin)

            return SizeMin + t * (SizeMax - SizeMin)

        # Outside refinement region
        return SizeMax

    return mesh_size_callback


def build_ellipse_mesh_callback(
    ellipses,
    SizeMin=3.0,
    SizeMax=30.0,
    DistMin=50.0,
    DistMax=100.0,
) -> Optional[MeshSizeCallback]:
    """
    Creates a mesh size callback around elliptical regions.

    Mesh size is controlled by the distance from the ellipse boundary:
      - <= DistMin from the ellipse: SizeMin
      - DistMin .. DistMax: linear transition
      - > DistMax: SizeMax

    Each ellipse is approximated using its center and its largest radius.
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

    # KD-tree containing all ellipse centers
    tree = cKDTree(centers)

    def mesh_size_callback(dim, tag, x, y, z, lc):
        point = np.array([x, y, z], dtype=float)

        # Find nearest ellipse center
        dist_center, idx = tree.query(point)

        # Approximate distance from the ellipse boundary
        dist = max(0.0, dist_center - scales[idx])

        # Exactly at / very close to ellipse
        if dist <= DistMin:
            return SizeMin

        # Transition region
        if dist < DistMax:
            t = (dist - DistMin) / (DistMax - DistMin)
            return SizeMin + t * (SizeMax - SizeMin)

        # Outside refinement region
        return SizeMax

    return mesh_size_callback


def build_fault_mesh_callback_from_fragments(
    fault_fragments,
    SizeMin=3.0,
    SizeMax=30.0,
    DistMin=20.0,
    DistMax=60.0,
) -> Optional[MeshSizeCallback]:
    """
    Creates a fault-based mesh refinement callback using OCC geometry.

    Fault surface points are sampled from the OCC fault fragments and
    stored in a KD-tree. Mesh size is controlled by the distance to the
    nearest sampled fault point:

        dist <= DistMin:
            SizeMin

        DistMin < dist < DistMax:
            Linear transition from SizeMin to SizeMax

        dist >= DistMax:
            SizeMax
    """

    if not fault_fragments:
        return None

    fault_points = []

    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        logger.warning(
            "scipy.interpolate.RBFInterpolator is not available; "
            "skipping fault mesh refinement."
        )
        return None

    for dim, tag in fault_fragments:
        if dim != 2:
            continue

        try:
            # ---------------------------------------------------------
            # Surface bounding box
            # ---------------------------------------------------------
            xmin, ymin, zmin, xmax, ymax, zmax = (
                gmsh.model.getBoundingBox(2, tag)
            )

            # ---------------------------------------------------------
            # Parametric bounds of the OCC surface
            # ---------------------------------------------------------
            bounds = gmsh.model.getParametrizationBounds(2, tag)

            umin, umax = bounds[0]
            vmin, vmax = bounds[1]

            # ---------------------------------------------------------
            # Sample points directly from the OCC surface
            # ---------------------------------------------------------
            surface_pts = []

            for u in np.linspace(umin, umax, 20):
                for v in np.linspace(vmin, vmax, 20):
                    try:
                        point = gmsh.model.getValue(2, tag, [u, v])
                        surface_pts.append(point)
                    except Exception:
                        continue

            surface_pts = np.asarray(surface_pts, dtype=float)

            if surface_pts.shape[0] < 3:
                logger.warning( "Not enough sample points for fault surface %s", tag, )
                continue

            # ---------------------------------------------------------
            # Interpolate Z = f(X, Y)
            # ---------------------------------------------------------
            rbf = RBFInterpolator(
                surface_pts[:, :2],
                surface_pts[:, 2],
                kernel="thin_plate_spline",
            )

            # ---------------------------------------------------------
            # Generate regular XY samples over the fault bounding box
            # ---------------------------------------------------------
            xs = np.linspace(xmin, xmax, 40)
            ys = np.linspace(ymin, ymax, 40)

            XX, YY = np.meshgrid(xs, ys)

            XY = np.column_stack(
                [XX.ravel(), YY.ravel()]
            )

            # ---------------------------------------------------------
            # Interpolated Z values
            # ---------------------------------------------------------
            ZZ = rbf(XY)

            sampled_fault_points = np.column_stack(
                [XY, ZZ]
            )

            fault_points.extend( sampled_fault_points.tolist() )

        except Exception as e:
            logger.warning( "Skipping fault surface %s: %s", tag, e, )
            continue

    # -------------------------------------------------------------
    # Validate sampled fault points
    # -------------------------------------------------------------
    if not fault_points:
        return None

    fault_points = np.asarray(
        fault_points,
        dtype=float,
    )

    if fault_points.ndim != 2 or fault_points.shape[1] != 3:
        return None

    # -------------------------------------------------------------
    # KD-tree containing ALL fault surface sample points
    # -------------------------------------------------------------
    tree = cKDTree(fault_points)

    # -------------------------------------------------------------
    # Mesh-size callback
    # -------------------------------------------------------------
    def mesh_size_callback(dim, tag, x, y, z, lc):
        point = np.array(
            [x, y, z],
            dtype=float,
        )

        # Distance to nearest sampled fault point
        dist, _ = tree.query(point)

        # Exactly at / very close to fault
        if dist <= DistMin:
            return SizeMin

        # Transition region
        if dist < DistMax:
            t = (
                (dist - DistMin)
                / (DistMax - DistMin)
            )

            return (
                SizeMin
                + t * (SizeMax - SizeMin)
            )

        # Outside refinement region
        return SizeMax

    return mesh_size_callback

def build_extra_plane_mesh_callback(
    extra_planes,
    SizeMin=3.0,
    SizeMax=30.0,
    DistMin=20.0,
    DistMax=60.0,
) -> Optional[MeshSizeCallback]:
    """
    Creates a mesh-size callback around extra planar regions.

    Each quadrilateral is split into two triangles. A centroid KD-tree
    prunes candidates using triangle bounding radii, while exact
    point-to-triangle distances preserve the original refinement behavior.
    Mesh size is controlled by the minimum distance to any triangle:

        distance <= DistMin:
            SizeMin

        DistMin < distance < DistMax:
            Linear transition from SizeMin to SizeMax

        distance >= DistMax:
            SizeMax
    """

    if not extra_planes:
        return None

    triangles = []

    for plane in extra_planes:
        if len(plane) != 12:
            logger.warning( "Skipping extra plane: expected 12 coordinates, got %d", len(plane), )
            continue

        p = np.asarray(plane, dtype=float).reshape(4, 3)
        p1, p2, p3, p4 = p

        # Split quadrilateral into two triangles
        triangles.append((p1, p2, p3))
        triangles.append((p1, p3, p4))

    if not triangles:
        return None

    # Build the spatial index once, not during Gmsh callback evaluations.
    # A centroid alone is not sufficient to identify the nearest triangle:
    # the closest point may lie on a large triangle far from its centroid.
    triangle_array = np.asarray(triangles, dtype=float)
    centroids = triangle_array.mean(axis=1)
    radii = np.linalg.norm(triangle_array - centroids[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centroids)
    max_radius = float(radii.max())

    def point_triangle_distance(point, a, b, c):
        """
        Exact closest Euclidean distance from a point to a triangle.
        """

        ab = b - a
        ac = c - a
        ap = point - a

        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)

        # Closest to vertex A
        if d1 <= 0.0 and d2 <= 0.0:
            return np.linalg.norm(point - a)

        bp = point - b

        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)

        # Closest to vertex B
        if d3 >= 0.0 and d4 <= d3:
            return np.linalg.norm(point - b)

        vc = d1 * d4 - d3 * d2

        # Closest to edge AB
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            projection = a + v * ab

            return np.linalg.norm(point - projection)

        cp = point - c

        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)

        # Closest to vertex C
        if d6 >= 0.0 and d5 <= d6:
            return np.linalg.norm(point - c)

        vb = d5 * d2 - d1 * d6

        # Closest to edge AC
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            projection = a + w * ac

            return np.linalg.norm(point - projection)

        va = d3 * d6 - d5 * d4

        # Closest to edge BC
        if (
            va <= 0.0
            and (d4 - d3) >= 0.0
            and (d5 - d6) >= 0.0
        ):
            w = (d4 - d3) / (
                (d4 - d3) + (d5 - d6)
            )

            projection = b + w * (c - b)

            return np.linalg.norm(point - projection)

        # Closest point is inside triangle
        denom = 1.0 / (va + vb + vc)

        v = vb * denom
        w = vc * denom

        projection = a + ab * v + ac * w

        return np.linalg.norm(point - projection)

    def mesh_size_callback(dim, tag, x, y, z, lc):
        point = np.array(
            [x, y, z],
            dtype=float,
        )

        # Start with the nearest centroid to obtain an exact upper bound.
        # For any triangle T with centroid C and bounding radius R,
        # distance(point, T) >= distance(point, C) - R.  The KD-tree
        # radius search therefore cannot omit the truly closest triangle.
        _, nearest_idx = tree.query(point)
        nearest_idx = int(nearest_idx)
        distance = point_triangle_distance(point, *triangle_array[nearest_idx])

        candidate_ids = tree.query_ball_point(point, r=distance + max_radius)
        for idx in candidate_ids:
            if idx == nearest_idx:
                continue
            # Per-triangle lower bound avoids unnecessary exact calculations.
            if np.linalg.norm(point - centroids[idx]) - radii[idx] >= distance:
                continue
            distance = min(
                distance,
                point_triangle_distance(point, *triangle_array[idx]),
            )

        # Exactly at / very close to extra plane
        if distance <= DistMin:
            return SizeMin

        # Transition region
        if distance < DistMax:
            t = (
                (distance - DistMin)
                / (DistMax - DistMin)
            )

            return (
                SizeMin
                + t * (SizeMax - SizeMin)
            )

        # Outside refinement region
        return SizeMax

    return mesh_size_callback

def build_general_refinement(
    refinement,
    field_gen,
    mesh_size: float,
) -> List[int]:
    """
    Builds general point- or line-based refinement fields.

    Linear configuration:
        SizeMin and DistMax must be provided.

        SizeMin is applied at DIST = 0 and increases linearly to
        mesh_size at DIST = DistMax.

    Function configuration:
        expression must be provided and should contain DIST.

    Returns:
        List of completed mesh-size field IDs.
    """
    if not refinement or refinement.general is None:
        return []

    configurations = refinement.general

    if isinstance(configurations, GeneralRefinement):
        configurations = [configurations]

    completed_fields = []

    for config_index, config in enumerate(configurations):
        coordinates = config.coordinates

        if len(coordinates) not in (1, 2):
            raise ValueError(
                "GeneralRefinement.coordinates must contain exactly "
                "one coordinate or two coordinates. "
                f"Configuration {config_index} contains "
                f"{len(coordinates)} coordinates."
            )

        # Exactly one refinement method must be selected.
        uses_expression = (
            config.expression is not None
            and config.expression.strip() != ""
        )

        uses_linear = (
            config.SizeMin is not None
            or config.DistMax is not None
        )

        if uses_expression and uses_linear:
            raise ValueError(
                "GeneralRefinement must use either 'expression' or "
                "'SizeMin' with 'DistMax', not both."
            )

        if not uses_expression:
            if config.SizeMin is None or config.DistMax is None:
                raise ValueError(
                    "GeneralRefinement requires either an expression "
                    "or both SizeMin and DistMax."
                )

            if config.SizeMin <= 0:
                raise ValueError(
                    "GeneralRefinement.SizeMin must be greater than zero."
                )

            if config.DistMax <= 0:
                raise ValueError(
                    "GeneralRefinement.DistMax must be greater than zero."
                )

        # -----------------------------------------------------
        # Create point or line geometry
        # -----------------------------------------------------
        if len(coordinates) == 1:
            x, y, z = map(float, coordinates[0])

            point_tag = gmsh.model.occ.addPoint(x, y, z)
            gmsh.model.occ.synchronize()

            fid_distance = field_gen.next()
            gmsh.model.mesh.field.add( "Distance", fid_distance, )
            gmsh.model.mesh.field.setNumbers( fid_distance, "NodesList", [point_tag], )

        else:
            x1, y1, z1 = map(float, coordinates[0])
            x2, y2, z2 = map(float, coordinates[1])

            segment_length = np.linalg.norm(
                [
                    x2 - x1,
                    y2 - y1,
                    z2 - z1,
                ]
            )

            if segment_length < 1.0e-10:
                raise ValueError(
                    "The two GeneralRefinement coordinates must "
                    "define a nonzero line segment."
                )

            start_tag = gmsh.model.occ.addPoint(
                x1,
                y1,
                z1,
            )
            end_tag = gmsh.model.occ.addPoint(
                x2,
                y2,
                z2,
            )
            line_tag = gmsh.model.occ.addLine(
                start_tag,
                end_tag,
            )

            gmsh.model.occ.synchronize()

            fid_distance = field_gen.next()
            gmsh.model.mesh.field.add( "Distance", fid_distance, )
            gmsh.model.mesh.field.setNumbers( fid_distance, "CurvesList", [line_tag], )
            gmsh.model.mesh.field.setNumber( fid_distance, "Sampling", 100, )

        # -----------------------------------------------------
        # Function refinement
        # -----------------------------------------------------
        if uses_expression:
            fid_result = field_gen.next()
            gmsh.model.mesh.field.add( "MathEval", fid_result, )

            expression = config.expression.replace(
                "DIST",
                f"F{fid_distance}",
            )

            gmsh.model.mesh.field.setString( fid_result, "F", expression, )

            logger.info( "General refinement %d expression: %s", config_index, expression, )

            completed_fields.append(fid_result)
            continue

        # -----------------------------------------------------
        # Linear refinement
        # -----------------------------------------------------
        fid_result = field_gen.next()
        gmsh.model.mesh.field.add( "Threshold", fid_result, )
        gmsh.model.mesh.field.setNumber( fid_result, "InField", fid_distance, )
        gmsh.model.mesh.field.setNumber( fid_result, "SizeMin", config.SizeMin, )
        gmsh.model.mesh.field.setNumber( fid_result, "SizeMax", mesh_size, )
        gmsh.model.mesh.field.setNumber( fid_result, "DistMin", 0.0, )
        gmsh.model.mesh.field.setNumber( fid_result, "DistMax", config.DistMax, )

        completed_fields.append(fid_result)

    return completed_fields


def build_combined_mesh_callback(
    triangulations=None,
    ellipses=None,
    fault_surfaces=None,
    fault_fragments=None,
    extra_planes=None,
    tri_cfg=None,
    ell_cfg=None,
    fault_cfg=None,
    extra_plane_cfg=None,
) -> Optional[MeshSizeCallback]:
    """
    Combines triangulation, ellipse, fault, and extra-plane
    mesh refinement callbacks.

    All refinement strategies use the same configuration:

        SizeMin -> minimum mesh size
        SizeMax -> maximum mesh size
        DistMin   -> distance for minimum mesh size
        DistMax   -> distance where mesh reaches maximum size

    The smallest mesh size requested by any active refinement
    strategy is returned.
    """

    tri_cb = None
    ell_cb = None
    fault_cb = None
    extra_plane_cb = None

    # ---------------------------------------------------------
    # Triangulation
    # ---------------------------------------------------------
    if (
        triangulations is not None
        and len(triangulations) > 0
        and tri_cfg is not None
    ):
        tri_cb = build_triangulation_mesh_callback(
            triangulations,
            SizeMin=tri_cfg.SizeMin,
            SizeMax=tri_cfg.SizeMax,
            DistMin=tri_cfg.DistMin,
            DistMax=tri_cfg.DistMax,
        )

    # ---------------------------------------------------------
    # Ellipse
    # ---------------------------------------------------------
    if (
        ellipses is not None
        and len(ellipses) > 0
        and ell_cfg is not None
    ):
        ell_cb = build_ellipse_mesh_callback(
            ellipses,
            SizeMin=ell_cfg.SizeMin,
            SizeMax=ell_cfg.SizeMax,
            DistMin=ell_cfg.DistMin,
            DistMax=ell_cfg.DistMax,
        )

    # ---------------------------------------------------------
    # Fault
    # ---------------------------------------------------------
    if (
        fault_fragments is not None
        and len(fault_fragments) > 0
        and fault_cfg is not None
    ):
        fault_cb = build_fault_mesh_callback_from_fragments(
            fault_fragments,
            SizeMin=fault_cfg.SizeMin,
            SizeMax=fault_cfg.SizeMax,
            DistMin=fault_cfg.DistMin,
            DistMax=fault_cfg.DistMax,
        )

    # ---------------------------------------------------------
    # Extra plane
    # ---------------------------------------------------------
    if (
        extra_planes is not None
        and len(extra_planes) > 0
        and extra_plane_cfg is not None
    ):
        extra_plane_cb = build_extra_plane_mesh_callback(
            extra_planes,
            SizeMin=extra_plane_cfg.SizeMin,
            SizeMax=extra_plane_cfg.SizeMax,
            DistMin=extra_plane_cfg.DistMin,
            DistMax=extra_plane_cfg.DistMax,
        )

    # ---------------------------------------------------------
    # No refinement enabled
    # ---------------------------------------------------------
    callbacks = [
        tri_cb,
        ell_cb,
        fault_cb,
        extra_plane_cb,
    ]

    if all(cb is None for cb in callbacks):
        return None

    # ---------------------------------------------------------
    # Combined callback
    # ---------------------------------------------------------
    def mesh_size_callback(dim, tag, x, y, z, lc):

        values = [lc]

        for callback in callbacks:
            if callback is not None:
                values.append( callback(dim, tag, x, y, z, lc) )

        return min(values)

    return mesh_size_callback


def build_refinement_fields(
    zip_info,
    well_lines,
    source_points,
    mine_shafts=None,
    triangulations=None,
    refinement=None,
    mesh_size: float = 20.0,
) -> List[int]:
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

    shaft_field = build_shaft_refinement(
        mine_shafts,
        refinement,
        field_gen,
    )

    general_fields = build_general_refinement(
        refinement=refinement,
        field_gen=field_gen,
        mesh_size=mesh_size,
    )

    if well_field is not None:
        active_fields.append(well_field)

    if source_field is not None:
        active_fields.append(source_field)

    if shaft_field is not None:
        active_fields.append(shaft_field)

    active_fields.extend(general_fields)

    return active_fields


def apply_background_fields(active_fields: List[int]) -> None:
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
