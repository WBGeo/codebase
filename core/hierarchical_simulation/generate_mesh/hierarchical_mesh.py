from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from py_api_wbgeo.nodesapi import wbgeo_component

from core.object_components import MeshResults, StructuralModelResults
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, meshit_coordinate_transform
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import (
    WellData, SourcesData, ShaftData, PlaneData, EllipseData, TriangulationData
)


def hierarchical_mesh(
    geomodel_result: Optional[StructuralModelResults] = None,
    mesh_type: str = "Un",
    surface_csv_dir: Optional[str] = None,

    extent1: Union[List[float], str] = "whole",
    extent2: Union[List[float], str] = None,
    extent_z: Optional[Tuple[float, float]] = None,

    # unstructured
    mesh_size1: Optional[float] = None,
    mesh_size2: Optional[float] = None,
    wells: Optional[WellData] = None,
    sources: Optional[SourcesData] = None,
    shafts: Optional[ShaftData] = None,
    extra_planes: Optional[PlaneData] = None,
    ellipses: Optional[EllipseData] = None,
    triangulations: Optional[TriangulationData] = None,
    # structured
    refinement1: object = None,
    refinement2: object = None,
    mesh_devision1: object = None,
    mesh_devision2: object = None,

    unstructured_params: dict = None,
    unstructured_params1: dict = None,
    unstructured_params2: dict = None,
    structured_params: dict = None,
    implicit_params: dict = None,
    inside_nodes_csv_dir: Optional[str] = None,

) -> tuple:

    mesh_type = mesh_type.lower()

    # Common unstructured parameters are applied to both meshes.
    # Level-specific dictionaries override only the values supplied for that
    # hierarchy level (for example, a different rotation_angle for mesh2).
    unstructured_params = unstructured_params or {}
    unstructured_params1 = {
        **unstructured_params,
        **(unstructured_params1 or {}),
    }
    unstructured_params2 = {
        **unstructured_params,
        **(unstructured_params2 or {}),
    }
    structured_params = structured_params or {}
    implicit_params = implicit_params or {}

    # CSV geological surfaces are supported for unstructured hierarchical
    # meshing. The same surface directory is passed to both hierarchy levels.
    if surface_csv_dir is not None and mesh_type != "un":
        raise ValueError(
            "surface_csv_dir is only supported for unstructured hierarchical meshes."
        )

    if geomodel_result is None and surface_csv_dir is None:
        raise ValueError(
            "Provide geomodel_result or surface_csv_dir for hierarchical meshing."
        )


    # --------------------------------------------------
    # Parameter checks
    # --------------------------------------------------

    if mesh_type == "str":

        # Structured meshes cannot use unstructured parameters
        if mesh_size1 is not None or mesh_size2 is not None:
            raise ValueError(
                "mesh_size1 and mesh_size2 are only valid for unstructured meshes."
            )

        if refinement1 is None or mesh_devision1 is None:
            raise ValueError(
                "refinement1 and mesh_devision1 must be provided for structured meshes."
            )

        if refinement2 is None:
            refinement2 = refinement1

        if mesh_devision2 is None:
            mesh_devision2 = mesh_devision1


    elif mesh_type == "un":

        # Unstructured meshes cannot use structured parameters
        if (
            refinement1 is not None
            or refinement2 is not None
            or mesh_devision1 is not None
            or mesh_devision2 is not None
        ):
            raise ValueError(
                "For unstructured meshes, use mesh_size1 and mesh_size2. "
                "refinement and mesh_devision are only valid for structured meshes."
            )

        if mesh_size1 is None:
            raise ValueError(
                "mesh_size1 must be provided for unstructured meshes."
            )

        if mesh_size2 is None:
            mesh_size2 = mesh_size1


    elif mesh_type == "im":

        # Implicit meshes do not use explicit mesh controls
        if (
            mesh_size1 is not None
            or mesh_size2 is not None
            or refinement1 is not None
            or refinement2 is not None
            or mesh_devision1 is not None
            or mesh_devision2 is not None
        ):
            raise ValueError(
                "Implicit meshes do not accept mesh_size, refinement, "
                "or mesh_devision parameters."
            )


    else:
        raise ValueError(
            f"Invalid mesh_type '{mesh_type}'. Use 'Un', 'Str', or 'Im'."
        )

    # ``extent_z`` optionally replaces the Z limits of an inferred/"whole"
    # outer extent: [z_min, z_max]. Values are absolute model coordinates.
    # An explicitly supplied extent1 always takes precedence.
    if extent_z is not None:
        if len(extent_z) != 2:
            raise ValueError(
                "extent_z must contain exactly two values: [z_min, z_max]."
            )
        extent_z_min, extent_z_max = (float(v) for v in extent_z)
        if not np.isfinite(extent_z_min) or not np.isfinite(extent_z_max):
            raise ValueError("extent_z values must be finite.")
        if extent_z_min >= extent_z_max:
            raise ValueError("extent_z requires z_min < z_max.")
    else:
        extent_z_min = extent_z_max = None

    # --------------------------------------------------
    # Resolve extent helper
    # --------------------------------------------------

    def _csv_surface_extent():
        """Return combined [xmin, xmax, ymin, ymax, zmin, zmax] of CSV surfaces."""
        csv_root = Path(surface_csv_dir).expanduser()
        if not csv_root.is_dir():
            raise ValueError(
                f"surface_csv_dir is not a directory: {surface_csv_dir}"
            )

        csv_files = sorted(
            p for p in csv_root.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        )
        if not csv_files:
            raise ValueError(
                f"No CSV surface files found in {surface_csv_dir}"
            )

        xyz_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        n_valid_points = 0

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            lower_to_original = {
                str(c).strip().lower(): c for c in df.columns
            }
            missing = [
                axis for axis in ("x", "y", "z")
                if axis not in lower_to_original
            ]
            if missing:
                raise ValueError(
                    f"{csv_path.name} must contain x, y, z columns; "
                    f"missing {missing}."
                )

            xyz = df[
                [
                    lower_to_original["x"],
                    lower_to_original["y"],
                    lower_to_original["z"],
                ]
            ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            if xyz.size == 0:
                continue

            xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
            n_valid_points += xyz.shape[0]

        if n_valid_points == 0:
            raise ValueError(
                "No finite x, y, z surface coordinates were found in "
                f"{surface_csv_dir}."
            )

        return np.array(
            [
                xyz_min[0], xyz_max[0],
                xyz_min[1], xyz_max[1],
                xyz_min[2], xyz_max[2],
            ],
            dtype=float,
        )

    def _resolve_extent(extent, apply_extent_z=False):

        model_extent = None
        if geomodel_result is not None:
            model_extent = np.asarray(
                geomodel_result.structural_frame.grid.extent,
                dtype=float
            )

        inferred = False
        if isinstance(extent, str) and extent.lower() == "whole":
            if model_extent is not None:
                resolved = model_extent.copy()
            elif surface_csv_dir is not None:
                resolved = _csv_surface_extent()
            else:
                raise ValueError(
                    "extent='whole' requires geomodel_result or surface_csv_dir."
                )
            inferred = True

        elif extent is None or (
            not isinstance(extent, str) and len(extent) == 0
        ):
            if model_extent is not None:
                resolved = model_extent.copy()
            elif surface_csv_dir is not None:
                resolved = _csv_surface_extent()
            else:
                raise ValueError(
                    "An extent requires geomodel_result or surface_csv_dir."
                )
            inferred = True
        else:
            # Explicit extents always take precedence and are not enlarged.
            return np.asarray(extent, dtype=float)

        if inferred and apply_extent_z and extent_z is not None:
            resolved = np.asarray(resolved, dtype=float).copy()
            resolved[4] = extent_z_min
            resolved[5] = extent_z_max

        return resolved

    extent1_resolved = _resolve_extent(
        extent1, apply_extent_z=True
    )

    if extent2 is None:
        raise ValueError(
            "extent2 must be provided as a smaller region inside extent1."
        )

    extent2_resolved = _resolve_extent(extent2)

    # --------------------------------------------------
    # Check extent containment
    # --------------------------------------------------
    # Both hierarchy extents may be either:
    #   * legacy [xmin, xmax, ymin, ymax, zmin, zmax], or
    #   * an oriented box given by 8 corner points (8, 3).
    #
    # IMPORTANT: this check does not replace/modify the original extents.
    # The original 8-corner extent is still passed unchanged to the mesher.

    def _extent_corners(extent):
        arr = np.asarray(extent, dtype=float)

        if arr.ndim == 1 and arr.size == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = arr
            return np.asarray([
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ], dtype=float)

        if arr.size == 24:
            return arr.reshape(8, 3)

        raise ValueError(
            "Extent must be either 6 values "
            "[xmin, xmax, ymin, ymax, zmin, zmax] "
            "or 8 three-dimensional corner points."
        )

    def _extent_contains(outer_extent, inner_extent, tol=1e-8):
        outer = np.asarray(outer_extent, dtype=float)
        inner_corners = _extent_corners(inner_extent)

        # Common case: the big model is an axis-aligned 6-value extent.
        if outer.ndim == 1 and outer.size == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = outer
            inside = (
                (inner_corners[:, 0] >= xmin - tol)
                & (inner_corners[:, 0] <= xmax + tol)
                & (inner_corners[:, 1] >= ymin - tol)
                & (inner_corners[:, 1] <= ymax + tol)
                & (inner_corners[:, 2] >= zmin - tol)
                & (inner_corners[:, 2] <= zmax + tol)
            )
            return bool(np.all(inside))

        # Oriented 8-corner outer box.  Corner convention is the same one used
        # by WBGeo's domain geometry: 0 is the origin and 1, 3, 4 define the
        # three local box directions.
        outer_corners = _extent_corners(outer)
        origin = outer_corners[0]
        basis = np.column_stack((
            outer_corners[1] - origin,
            outer_corners[3] - origin,
            outer_corners[4] - origin,
        ))

        try:
            local = np.linalg.solve(
                basis,
                (inner_corners - origin).T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("extent1 defines a degenerate 8-corner box.") from exc

        return bool(np.all((local >= -tol) & (local <= 1.0 + tol)))

    if not _extent_contains(extent1_resolved, extent2_resolved):
        raise ValueError("extent2 must be fully inside extent1.")

    # --------------------------------------------------
    # Select engineering objects for one hierarchy level
    # --------------------------------------------------

    def _for_mesh(data, mesh_id):
        """Return data in the *normal create_unstructured_mesh_data format*.

        Hierarchical loaders return {1: normal_data_for_mesh1,
        2: normal_data_for_mesh2}. Non-hierarchical data are applied to both
        meshes unchanged. A legacy two-item tuple is also supported.
        """
        if data is None:
            return None
        if isinstance(data, dict) and set(data.keys()).issubset({1, 2}):
            return data.get(mesh_id, [])
        if isinstance(data, tuple) and len(data) == 2:
            return data[mesh_id - 1]
        return data

    # --------------------------------------------------
    # Clip engineering objects to one mesh extent
    # --------------------------------------------------

    def _domain_frame(extent):
        """Return an affine frame for a 6-value AABB or an 8-corner box.

        World points are represented as p = origin + basis @ q, where q is in
        the unit cube [0, 1]^3.  For a legacy 6-value extent the basis is
        diagonal.  For an 8-corner domain the corner ordering follows WBGeo's
        domain convention: corner 0 is the origin and corners 1, 3, and 4
        define the three box edges.
        """
        arr = np.asarray(extent, dtype=float)
        if arr.ndim == 1 and arr.size == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = arr
            origin = np.array([xmin, ymin, zmin], dtype=float)
            basis = np.diag([xmax - xmin, ymax - ymin, zmax - zmin])
        elif arr.size == 24:
            corners = arr.reshape(8, 3)
            origin = corners[0].copy()
            basis = np.column_stack((
                corners[1] - corners[0],
                corners[3] - corners[0],
                corners[4] - corners[0],
            ))
        else:
            raise ValueError(
                "Extent must be either 6 values "
                "[xmin, xmax, ymin, ymax, zmin, zmax] or 8 corner points."
            )
        if abs(np.linalg.det(basis)) < 1e-15:
            raise ValueError("Extent is degenerate and cannot define a 3-D domain.")
        return origin, basis, np.linalg.inv(basis)

    def _to_domain_local(points, extent):
        origin, _, inv_basis = _domain_frame(extent)
        pts = np.asarray(points, dtype=float)
        return (pts - origin) @ inv_basis.T

    def _from_domain_local(points, extent):
        origin, basis, _ = _domain_frame(extent)
        q = np.asarray(points, dtype=float)
        return origin + q @ basis.T

    def _inside(point, extent, tol=1e-9):
        q = _to_domain_local(np.asarray(point, dtype=float), extent)
        return bool(np.all(q >= -tol) and np.all(q <= 1.0 + tol))

    def _clip_segment_to_extent(p0, p1, extent, tol=1e-12):
        """Clip a world-coordinate segment to a 6-value or 8-corner domain."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        # In domain-local coordinates every supported extent is the unit cube.
        q0 = _to_domain_local(p0, extent)
        q1 = _to_domain_local(p1, extent)
        d = q1 - q0
        t0, t1 = 0.0, 1.0

        for axis in range(3):
            if abs(d[axis]) <= tol:
                if q0[axis] < -tol or q0[axis] > 1.0 + tol:
                    return None
                continue
            a = (0.0 - q0[axis]) / d[axis]
            b = (1.0 - q0[axis]) / d[axis]
            enter, leave = min(float(a), float(b)), max(float(a), float(b))
            t0 = max(t0, enter)
            t1 = min(t1, leave)
            if t0 > t1 + tol:
                return None

        return p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)

    def _clip_wells(data, extent):
        if not data:
            return []
        result = []
        for well in data:
            pts = np.asarray(well, dtype=float).reshape(-1, 3)
            pieces = []
            current = []
            for a, b in zip(pts[:-1], pts[1:]):
                clipped = _clip_segment_to_extent(a, b, extent)
                if clipped is None:
                    if len(current) >= 2:
                        pieces.append(current)
                    current = []
                    continue
                c0, c1 = clipped
                if not current:
                    current = [c0, c1]
                elif np.allclose(current[-1], c0):
                    if not np.allclose(current[-1], c1):
                        current.append(c1)
                else:
                    if len(current) >= 2:
                        pieces.append(current)
                    current = [c0, c1]
            if len(current) >= 2:
                pieces.append(current)
            for piece in pieces:
                clean = [piece[0]]
                for pt in piece[1:]:
                    if not np.allclose(clean[-1], pt):
                        clean.append(pt)
                if len(clean) >= 2:
                    result.append(tuple(np.asarray(clean).reshape(-1).tolist()))
        return result

    def _clip_sources(data, extent):
        return [tuple(src) for src in (data or []) if _inside(src, extent)]

    def _clip_shafts(data, extent):
        """Clip each cylinder axis segment; radius is kept unchanged."""
        result = []
        for shaft in data or []:
            vals = np.asarray(shaft, dtype=float).reshape(-1)
            if vals.size != 7:
                result.append(shaft)
                continue
            center, axis, radius = vals[:3], vals[3:6], float(vals[6])
            clipped = _clip_segment_to_extent(center, center + axis, extent)
            if clipped is None:
                continue
            start, end = clipped
            new_axis = end - start
            if np.linalg.norm(new_axis) > 1e-12:
                result.append(tuple(np.r_[start, new_axis, radius].tolist()))
        return result

    def _clip_planes(data, extent):
        """Clamp legacy 4-corner planes in domain-local coordinates."""
        result = []
        for plane in data or []:
            pts = np.asarray(plane, dtype=float).reshape(-1, 3)
            if pts.shape[0] != 4:
                result.append(plane)
                continue
            q = _to_domain_local(pts, extent)
            q_clipped = np.clip(q, 0.0, 1.0)
            clipped = _from_domain_local(q_clipped, extent)
            if np.unique(np.round(clipped, 10), axis=0).shape[0] >= 3:
                result.append(tuple(clipped.reshape(-1).tolist()))
        return result

    def _clip_ellipses(data, extent):
        """Keep ellipses whose conservative bounding sphere fits the domain."""
        result = []
        origin, basis, inv_basis = _domain_frame(extent)
        for ellipse in data or []:
            center = np.asarray(ellipse['center'], dtype=float)
            radius = float(max(ellipse['radii']))
            q = (center - origin) @ inv_basis.T
            # A world-space radius-R sphere maps to these conservative local
            # half-widths along each local coordinate axis.
            dq = radius * np.linalg.norm(inv_basis, axis=1)
            if np.all(q - dq >= -1e-9) and np.all(q + dq <= 1.0 + 1e-9):
                result.append(ellipse)
        return result

    def _clip_triangulations(data, extent):
        if data is None:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr.reshape((0, 4)) if arr.ndim != 2 else arr
        q = _to_domain_local(arr[:, :3], extent)
        mask = np.all((q >= -1e-9) & (q <= 1.0 + 1e-9), axis=1)
        return arr[mask]

    def _clip_objects_for_extent(extent, wells_, sources_, shafts_, planes_, ellipses_, triangulations_):
        return (
            _clip_wells(wells_, extent),
            _clip_sources(sources_, extent),
            _clip_shafts(shafts_, extent),
            _clip_planes(planes_, extent),
            _clip_ellipses(ellipses_, extent),
            _clip_triangulations(triangulations_, extent),
        )

    # --------------------------------------------------
    # Mesh creator
    # --------------------------------------------------

    def _create_mesh(extent_array, mesh_id):

        if mesh_type == "un":

            mesh_size = mesh_size1 if mesh_id == 1 else mesh_size2
            # IMPORTANT: after this selection there is no hierarchical format
            # anymore. Each value is exactly the same kind of list/array that
            # create_unstructured_mesh_data() receives in the normal workflow.
            mesh_wells = _for_mesh(wells, mesh_id)
            mesh_sources = _for_mesh(sources, mesh_id)
            mesh_shafts = _for_mesh(shafts, mesh_id)
            mesh_extra_planes = _for_mesh(extra_planes, mesh_id)
            mesh_ellipses = _for_mesh(ellipses, mesh_id)
            mesh_triangulations = _for_mesh(triangulations, mesh_id)

            # Hierarchical objects describe the same physical geometry at both
            # levels. Clip the selected geometry to this mesh extent *before*
            # create_unstructured_mesh_data() validates it. This is essential
            # for mesh2, whose extent is smaller than mesh1.
            (
                mesh_wells,
                mesh_sources,
                mesh_shafts,
                mesh_extra_planes,
                mesh_ellipses,
                mesh_triangulations,
            ) = _clip_objects_for_extent(
                extent_array,
                mesh_wells,
                mesh_sources,
                mesh_shafts,
                mesh_extra_planes,
                mesh_ellipses,
                mesh_triangulations,
            )

            level_unstructured_params = dict(
                unstructured_params1 if mesh_id == 1 else unstructured_params2
            )

            # IMPORTANT FOR HIERARCHICAL MATCHING:
            # Generate both meshes in the original/world coordinate system.
            # Output rotation/translation is applied only AFTER the small-mesh
            # boundary nodes have been identified and matched to mesh1.
            for key in (
                "rotation_angle", "rotation_angle_translate", "xt", "yt", "zt"
            ):
                level_unstructured_params.pop(key, None)

            # Reuse create_unstructured_mesh_data() for both input modes:
            # structural-model surfaces or a directory of geological CSV surfaces.
            # Do not duplicate surface loading/interpolation logic here.
            if surface_csv_dir is not None:
                if "surface_csv_dir" in level_unstructured_params:
                    raise ValueError(
                        "surface_csv_dir was provided both to hierarchical_mesh() "
                        "and in unstructured parameters."
                    )
                level_unstructured_params = {
                    **level_unstructured_params,
                    "surface_csv_dir": surface_csv_dir,
                }

            return create_unstructured_mesh_data(
                geomodel_result,
                extent=extent_array,
                mesh_size=mesh_size,
                wells=mesh_wells,
                sources=mesh_sources,
                shafts=mesh_shafts,
                extra_planes=mesh_extra_planes,
                ellipses=mesh_ellipses,
                triangulations=mesh_triangulations,
                **level_unstructured_params
            )

        elif mesh_type == "str":

            refinement = refinement1 if mesh_id == 1 else refinement2
            mesh_devision = mesh_devision1 if mesh_id == 1 else mesh_devision2

            return create_structured_mesh_data(
                geomodel_result,
                extent=extent_array,
                refinement_data=refinement,
                mesh_devision=mesh_devision,
                **structured_params
            )

        elif mesh_type == "im":

            return create_implicit_structured_mesh(
                geomodel_result,
                extent=extent_array,
                **implicit_params
            )

        else:
            raise ValueError(
                f"Invalid mesh_type '{mesh_type}'. Use 'Un', 'Str', or 'Im'."
            )

    # --------------------------------------------------
    # Generate meshes
    # --------------------------------------------------

    mesh1 = _create_mesh(extent1_resolved, 1)
    mesh2 = _create_mesh(extent2_resolved, 2)

    # --------------------------------------------------
    # Get boundary nodes and match them BEFORE output transforms
    # --------------------------------------------------
    # Boundary matching is common to all hierarchical mesh types (Un, Str, Im).
    # mesh2.point_sets identifies the six physical faces of the inner mesh.
    # For unstructured meshes, output rotation/translation was intentionally
    # disabled during mesh creation, so both meshes are still in the common
    # original/world frame here. Structured and implicit meshes are generated
    # directly in that common frame as well. Boundary selection and nearest-node
    # matching therefore use the same procedure for all three mesh types.

    nodes_small_world = np.asarray(mesh2.nodes, dtype=float).copy()
    nodes_big_world = np.asarray(mesh1.nodes, dtype=float).copy()

    face_names = ("left", "right", "front", "back", "bottom", "top")

    if not getattr(mesh2, "point_sets", None):
        raise ValueError(
            "mesh2 does not contain boundary point_sets. "
            "Hierarchical boundary matching requires the boundary node sets "
            "created by the mesh generator."
        )

    boundary_faces_world = {}
    for face in face_names:
        node_ids = np.asarray(mesh2.point_sets.get(face, []), dtype=int)

        if node_ids.size == 0:
            boundary_faces_world[face] = np.empty((0, 3), dtype=float)
            continue

        if np.any(node_ids < 0) or np.any(node_ids >= len(nodes_small_world)):
            raise ValueError(
                f"mesh2.point_sets['{face}'] contains invalid node IDs."
            )

        boundary_faces_world[face] = nodes_small_world[node_ids]

    # Match the small-mesh physical boundary to mesh1 while BOTH meshes are
    # still unrotated/untranslated. The resulting indices are therefore
    # independent of any output coordinate transform.
    tree = cKDTree(nodes_big_world)
    boundary_node_ids_big = {}

    for face, face_nodes in boundary_faces_world.items():
        if face_nodes.size == 0:
            boundary_node_ids_big[face] = np.empty((0,), dtype=int)
            continue

        _, idx_closest = tree.query(face_nodes)
        boundary_node_ids_big[face] = np.asarray(idx_closest, dtype=int)

    # --------------------------------------------------
    # Apply requested output transforms AFTER boundary matching
    # --------------------------------------------------
    def _apply_output_transform(mesh, params):
        if mesh_type != "un":
            return

        rotation_angle = params.get("rotation_angle", None)
        rotation_angle_translate = params.get("rotation_angle_translate", None)
        xt = params.get("xt", None)
        yt = params.get("yt", None)
        zt = params.get("zt", None)

        if rotation_angle is not None and rotation_angle_translate is not None:
            raise ValueError(
                "Use either rotation_angle or rotation_angle_translate, not both."
            )

        if rotation_angle is not None:
            if any(v is not None for v in (xt, yt, zt)):
                raise ValueError(
                    "xt/yt/zt are only valid with rotation_angle_translate."
                )
            mesh.nodes = meshit_coordinate_transform(
                mesh.nodes,
                angle_deg=rotation_angle,
                translate=False,
            )

        elif rotation_angle_translate is not None:
            mesh.nodes = meshit_coordinate_transform(
                mesh.nodes,
                angle_deg=rotation_angle_translate,
                translate=True,
                xt=xt,
                yt=yt,
                zt=zt,
            )

        elif any(v is not None for v in (xt, yt, zt)):
            raise ValueError(
                "xt/yt/zt require rotation_angle_translate."
            )

    _apply_output_transform(mesh1, unstructured_params1)
    _apply_output_transform(mesh2, unstructured_params2)

    # --------------------------------------------------
    # Build the two boundary-node coordinate datasets
    # --------------------------------------------------
    # The node IDs always belong to mesh1 (the big mesh) and were selected
    # while both meshes were still in the original/world coordinate frame.
    #
    # 1) boundary_nodes_big:
    #       [big_node_id, x_big_final, y_big_final, z_big_final]
    #    Coordinates are those of the final big mesh, including any transform
    #    requested for mesh1.
    #
    # 2) boundary_nodes_small_frame:
    #       [big_node_id, x_small_frame, y_small_frame, z_small_frame]
    #    The SAME selected big-mesh world coordinates are transformed with the
    #    small mesh transform, using the centre of the FULL untransformed small
    #    mesh.  This puts the selected big nodes in exactly the coordinate frame
    #    used by the final small mesh.

    nodes_big_final = np.asarray(mesh1.nodes, dtype=float)

    def _transform_points_to_mesh_frame(points_world, reference_world, params):
        """Express selected mesh1 points in the final mesh2 coordinate frame.

        Structured and implicit meshes currently have no hierarchical output
        rotation/translation, so this is the identity transform for those mesh
        types. For unstructured meshes, use the full untransformed mesh2 to
        define xc/yc/(zc); using only the boundary subset would give a different
        centre from the transform applied to the complete mesh2.
        """
        if mesh_type != "un":
            return np.asarray(points_world, dtype=float).copy()
        points = np.asarray(points_world, dtype=float).copy()
        reference = np.asarray(reference_world, dtype=float)

        rotation_angle = params.get("rotation_angle", None)
        rotation_angle_translate = params.get("rotation_angle_translate", None)
        xt = params.get("xt", None)
        yt = params.get("yt", None)
        zt = params.get("zt", None)

        if rotation_angle is not None and rotation_angle_translate is not None:
            raise ValueError(
                "Use either rotation_angle or rotation_angle_translate, not both."
            )

        if rotation_angle is None and rotation_angle_translate is None:
            if any(v is not None for v in (xt, yt, zt)):
                raise ValueError("xt/yt/zt require rotation_angle_translate.")
            return points

        if rotation_angle is not None and any(v is not None for v in (xt, yt, zt)):
            raise ValueError(
                "xt/yt/zt are only valid with rotation_angle_translate."
            )

        xc = 0.5 * (reference[:, 0].min() + reference[:, 0].max())
        yc = 0.5 * (reference[:, 1].min() + reference[:, 1].max())

        angle = rotation_angle if rotation_angle is not None else rotation_angle_translate
        theta = np.deg2rad(float(angle))
        c = np.cos(theta)
        s = np.sin(theta)

        X = points[:, 0] - xc
        Y = points[:, 1] - yc

        result = points.copy()
        result[:, 0] = X * c - Y * s
        result[:, 1] = X * s + Y * c

        if rotation_angle_translate is None:
            # Same semantics as rotation_angle in mesh_data.py: XY is centred
            # and rotated, while the original Z coordinate is preserved.
            result[:, 2] = points[:, 2]
            return result

        if any(v is not None for v in (xt, yt, zt)) and not all(
            v is not None for v in (xt, yt, zt)
        ):
            raise ValueError("xt, yt, and zt must be provided together.")

        tx = 0.0 if xt is None else float(xt)
        ty = 0.0 if yt is None else float(yt)
        tz = 0.0 if zt is None else float(zt)
        zc = 0.5 * (reference[:, 2].min() + reference[:, 2].max())

        result[:, 0] += tx
        result[:, 1] += ty
        result[:, 2] = points[:, 2] - zc + tz
        return result

    boundary_nodes_big = {}
    boundary_nodes_small_frame = {}

    for face, idx_closest in boundary_node_ids_big.items():
        if idx_closest.size == 0:
            boundary_nodes_big[face] = np.empty((0, 4), dtype=float)
            boundary_nodes_small_frame[face] = np.empty((0, 4), dtype=float)
            continue

        selected_big_world = nodes_big_world[idx_closest]
        # Express the SAME selected mesh1 nodes in mesh2's final coordinate
        # frame. For Str/Im this is currently the same world coordinate frame;
        # for Un it includes mesh2's requested output transform.
        selected_big_small_frame = _transform_points_to_mesh_frame(
            selected_big_world,
            nodes_small_world,
            unstructured_params2,
        )

        boundary_nodes_big[face] = np.hstack(
            [
                idx_closest.reshape(-1, 1),
                nodes_big_final[idx_closest],
            ]
        )

        boundary_nodes_small_frame[face] = np.hstack(
            [
                idx_closest.reshape(-1, 1),
                selected_big_small_frame,
            ]
        )

    # Keep the public API backward-compatible: hierarchical_mesh() returns
    # exactly three values as before.  The third value contains both coordinate
    # representations of the SAME mesh1 node IDs.  Export code can therefore
    # write them into separate subdirectories without changing mesh generation.
    # --------------------------------------------------
    # Select ALL mesh1 nodes physically inside mesh2's extent
    # --------------------------------------------------
    # Selection is deliberately performed in the original/world frame, before
    # considering either mesh's output transform. This makes the selected node
    # IDs independent of rotation/translation. The same mesh1 node IDs are then
    # represented in two coordinate frames:
    #
    #   inside_nodes["big_frame"]   -> mesh1 final coordinates
    #   inside_nodes["small_frame"] -> those SAME physical mesh1 points
    #                                  expressed in mesh2 final coordinates
    #
    # This works for both 6-value and oriented 8-corner extent2 definitions.
    q_big_in_small = _to_domain_local(nodes_big_world, extent2_resolved)
    inside_mask = np.all(
        (q_big_in_small >= -1e-9) & (q_big_in_small <= 1.0 + 1e-9),
        axis=1,
    )
    inside_node_ids = np.flatnonzero(inside_mask).astype(int)
    selected_inside_world = nodes_big_world[inside_node_ids]

    selected_inside_small_frame = _transform_points_to_mesh_frame(
        selected_inside_world,
        nodes_small_world,
        unstructured_params2,
    )

    inside_nodes_big = np.hstack([
        inside_node_ids.reshape(-1, 1),
        nodes_big_final[inside_node_ids],
    ])
    inside_nodes_small_frame = np.hstack([
        inside_node_ids.reshape(-1, 1),
        selected_inside_small_frame,
    ])

    # Optional direct CSV output. Node IDs are mesh1's zero-based node indices,
    # matching the existing hierarchical boundary-node convention.
    if inside_nodes_csv_dir is not None:
        output_dir = Path(inside_nodes_csv_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        columns = ["NodeID", "X", "Y", "Z"]
        pd.DataFrame(inside_nodes_big, columns=columns).to_csv(
            output_dir / "mesh1_nodes_inside_mesh2_big_frame.csv",
            index=False,
        )
        pd.DataFrame(inside_nodes_small_frame, columns=columns).to_csv(
            output_dir / "mesh1_nodes_inside_mesh2_small_frame.csv",
            index=False,
        )

    # Keep the public API backward-compatible: hierarchical_mesh() still returns
    # exactly three values. Existing boundary exporters continue to use the
    # big_frame/small_frame entries. The additional inside_nodes entry contains
    # the two coordinate representations for every mesh1 node inside mesh2.
    # Store the exact resolved affine transformation between the two FINAL
    # mesh coordinate frames.  This is intentionally derived from the same
    # full untransformed mesh references used above, rather than estimated
    # later from boundary CSVs.  Homogeneous 4x4 matrices make the metadata
    # sufficient to transform arbitrary GOLEM sample coordinates.
    def _world_to_frame_matrix(reference_world, params):
        matrix = np.eye(4, dtype=float)
        if mesh_type != "un":
            return matrix

        reference = np.asarray(reference_world, dtype=float)
        rotation_angle = params.get("rotation_angle", None)
        rotation_angle_translate = params.get("rotation_angle_translate", None)
        xt = params.get("xt", None)
        yt = params.get("yt", None)
        zt = params.get("zt", None)

        if rotation_angle is not None and rotation_angle_translate is not None:
            raise ValueError(
                "Use either rotation_angle or rotation_angle_translate, not both."
            )
        if rotation_angle is None and rotation_angle_translate is None:
            return matrix

        angle = rotation_angle if rotation_angle is not None else rotation_angle_translate
        theta = np.deg2rad(float(angle))
        c, s = np.cos(theta), np.sin(theta)
        xc = 0.5 * (reference[:, 0].min() + reference[:, 0].max())
        yc = 0.5 * (reference[:, 1].min() + reference[:, 1].max())

        matrix[0, :] = [c, -s, 0.0, -c * xc + s * yc]
        matrix[1, :] = [s,  c, 0.0, -s * xc - c * yc]

        if rotation_angle_translate is not None:
            if any(v is not None for v in (xt, yt, zt)) and not all(
                v is not None for v in (xt, yt, zt)
            ):
                raise ValueError("xt, yt, and zt must be provided together.")
            tx = 0.0 if xt is None else float(xt)
            ty = 0.0 if yt is None else float(yt)
            tz = 0.0 if zt is None else float(zt)
            zc = 0.5 * (reference[:, 2].min() + reference[:, 2].max())
            matrix[0, 3] += tx
            matrix[1, 3] += ty
            matrix[2, :] = [0.0, 0.0, 1.0, -zc + tz]
        return matrix

    world_to_big = _world_to_frame_matrix(nodes_big_world, unstructured_params1)
    world_to_small = _world_to_frame_matrix(nodes_small_world, unstructured_params2)
    big_to_small = world_to_small @ np.linalg.inv(world_to_big)

    boundary_nodes = {
        "big_frame": boundary_nodes_big,
        "small_frame": boundary_nodes_small_frame,
        "inside_nodes": {
            "big_frame": inside_nodes_big,
            "small_frame": inside_nodes_small_frame,
        },
        "transformation_parameters": {
            "world_to_big_matrix": world_to_big.tolist(),
            "world_to_small_matrix": world_to_small.tolist(),
            "big_to_small_matrix": big_to_small.tolist(),
        },
    }

    return mesh1, mesh2, boundary_nodes
