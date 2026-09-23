from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
from scipy.spatial import cKDTree
from py_api_wbgeo.nodesapi import wbgeo_component

from core.object_components import MeshResults, StructuralModelResults
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import (
    WellData, SourcesData, ShaftData, PlaneData, EllipseData, TriangulationData
)


def hierarchical_mesh(
    geomodel_result: StructuralModelResults,
    mesh_type: str = "Un",

    extent1: Union[List[float], str] = "whole",
    extent2: Union[List[float], str] = None,

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

    # --------------------------------------------------
    # Resolve extent helper
    # --------------------------------------------------

    def _resolve_extent(extent):

        model_extent = np.asarray(
            geomodel_result.structural_frame.grid.extent,
            dtype=float
        )

        if isinstance(extent, str) and extent.lower() == "whole":
            return model_extent

        if extent is None or len(extent) == 0:
            return model_extent

        return np.asarray(extent, dtype=float)

    extent1_resolved = _resolve_extent(extent1)

    if extent2 is None:
        raise ValueError(
            "extent2 must be provided as a smaller region inside extent1."
        )

    extent2_resolved = _resolve_extent(extent2)

    # --------------------------------------------------
    # Check extent containment
    # --------------------------------------------------

    if not all([
        extent2_resolved[0] >= extent1_resolved[0],
        extent2_resolved[1] <= extent1_resolved[1],
        extent2_resolved[2] >= extent1_resolved[2],
        extent2_resolved[3] <= extent1_resolved[3],
        extent2_resolved[4] >= extent1_resolved[4],
        extent2_resolved[5] <= extent1_resolved[5],
    ]):
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

    def _inside(point, extent, tol=1e-9):
        x, y, z = map(float, point)
        return (
            extent[0] - tol <= x <= extent[1] + tol and
            extent[2] - tol <= y <= extent[3] + tol and
            extent[4] - tol <= z <= extent[5] + tol
        )

    def _clip_segment_to_extent(p0, p1, extent, tol=1e-12):
        """Liang-Barsky style clipping of a 3-D line segment to an AABB."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0
        lo = np.array([extent[0], extent[2], extent[4]], dtype=float)
        hi = np.array([extent[1], extent[3], extent[5]], dtype=float)
        t0, t1 = 0.0, 1.0
        for axis in range(3):
            if abs(d[axis]) <= tol:
                if p0[axis] < lo[axis] - tol or p0[axis] > hi[axis] + tol:
                    return None
                continue
            a = (lo[axis] - p0[axis]) / d[axis]
            b = (hi[axis] - p0[axis]) / d[axis]
            enter, leave = min(a, b), max(a, b)
            t0 = max(t0, enter)
            t1 = min(t1, leave)
            if t0 > t1 + tol:
                return None
        return p0 + t0 * d, p0 + t1 * d

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
                # Remove duplicate consecutive points before flattening.
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
        """Keep the 4-corner format expected downstream and clamp to the AABB.

        This mirrors the legacy coordinate-wise boundary clamping already used
        by the normal unstructured extra-plane creation code.
        """
        result = []
        lo = np.array([extent[0], extent[2], extent[4]], dtype=float)
        hi = np.array([extent[1], extent[3], extent[5]], dtype=float)
        for plane in data or []:
            pts = np.asarray(plane, dtype=float).reshape(-1, 3)
            if pts.shape[0] != 4:
                result.append(plane)
                continue
            clipped = np.minimum(np.maximum(pts, lo), hi)
            # A collapsed quadrilateral cannot define a surface.
            if np.unique(np.round(clipped, 10), axis=0).shape[0] >= 3:
                result.append(tuple(clipped.reshape(-1).tolist()))
        return result

    def _clip_ellipses(data, extent):
        """Keep ellipses that fit the extent. Partial ellipse clipping would
        change the object type, so it is intentionally left to a future surface
        clipping representation.
        """
        result = []
        for ellipse in data or []:
            center = np.asarray(ellipse['center'], dtype=float)
            radius = float(max(ellipse['radii']))
            lo = np.array([extent[0], extent[2], extent[4]], dtype=float)
            hi = np.array([extent[1], extent[3], extent[5]], dtype=float)
            if np.all(center - radius >= lo) and np.all(center + radius <= hi):
                result.append(ellipse)
        return result

    def _clip_triangulations(data, extent):
        if data is None:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr.reshape((0, 4)) if arr.ndim != 2 else arr
        xyz = arr[:, :3]
        mask = (
            (xyz[:, 0] >= extent[0]) & (xyz[:, 0] <= extent[1]) &
            (xyz[:, 1] >= extent[2]) & (xyz[:, 1] <= extent[3]) &
            (xyz[:, 2] >= extent[4]) & (xyz[:, 2] <= extent[5])
        )
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

            level_unstructured_params = (
                unstructured_params1 if mesh_id == 1 else unstructured_params2
            )

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
    # Get boundary nodes of the small mesh from point_sets
    # --------------------------------------------------
    #
    # The unstructured mesher already determines the physical boundary-node
    # sets when MeshResults is created. Reuse those sets here instead of
    # detecting faces again from min/max x, y and z. This is important for
    # rotated/oriented domains, whose physical faces need not be parallel to
    # the global coordinate axes.
    #
    # point_sets contains node IDs, e.g.
    #   {"left": [...], "right": [...], "front": [...],
    #    "back": [...], "bottom": [...], "top": [...]}
    #
    # The corresponding coordinates are obtained directly from mesh2.nodes.

    nodes_small = np.asarray(mesh2.nodes, dtype=float)
    nodes_big = np.asarray(mesh1.nodes, dtype=float)

    def _extent_xy_center(extent):
        # mesh_data.py rotates the output coordinates around the centre of the
        # meshed domain. Support both the normal six-value extent and an
        # eight-corner extent.
        arr = np.asarray(extent, dtype=float)
        if arr.ndim == 1 and arr.size == 6:
            return np.array([
                0.5 * (arr[0] + arr[1]),
                0.5 * (arr[2] + arr[3]),
            ])
        arr = arr.reshape(-1, 3)
        if arr.shape[0] != 8:
            raise ValueError(
                "Extent must contain six values or eight 3D corner points."
            )
        return arr[:, :2].mean(axis=0)

    def _undo_unstructured_output_transform(nodes, extent, params):
        # create_unstructured_mesh_data() stores rotated x/y coordinates after
        # translating the domain centre to the origin. Undo that operation so
        # mesh1 and mesh2 can be compared in the same original/world frame.
        result = np.asarray(nodes, dtype=float).copy()
        angle = params.get("rotation_angle", None)
        if angle is None:
            return result

        theta = np.deg2rad(float(angle))
        c = np.cos(theta)
        s = np.sin(theta)

        xr = result[:, 0].copy()
        yr = result[:, 1].copy()

        # Inverse of:
        #   xr = x*c - y*s
        #   yr = x*s + y*c
        x = xr * c + yr * s
        y = -xr * s + yr * c

        cx, cy = _extent_xy_center(extent)
        result[:, 0] = x + cx
        result[:, 1] = y + cy
        return result

    # point_sets identify WHICH mesh2 nodes are on each physical face.
    # The KD-tree search must use a common coordinate frame when the two
    # hierarchy levels have different rotations/translations.
    if mesh_type == "un":
        nodes_big_search = _undo_unstructured_output_transform(
            nodes_big, extent1_resolved, unstructured_params1
        )
        nodes_small_search = _undo_unstructured_output_transform(
            nodes_small, extent2_resolved, unstructured_params2
        )
    else:
        nodes_big_search = nodes_big
        nodes_small_search = nodes_small

    face_names = ("left", "right", "front", "back", "bottom", "top")

    if not getattr(mesh2, "point_sets", None):
        raise ValueError(
            "mesh2 does not contain boundary point_sets. "
            "Hierarchical boundary matching requires the boundary node sets "
            "created by the mesh generator."
        )

    boundary_faces = {}
    for face in face_names:
        node_ids = np.asarray(mesh2.point_sets.get(face, []), dtype=int)

        if node_ids.size == 0:
            boundary_faces[face] = np.empty((0, 3), dtype=float)
            continue

        if np.any(node_ids < 0) or np.any(node_ids >= len(nodes_small)):
            raise ValueError(
                f"mesh2.point_sets['{face}'] contains invalid node IDs."
            )

        boundary_faces[face] = nodes_small_search[node_ids]


    # --------------------------------------------------
    # Find closest nodes in big mesh for each face
    # --------------------------------------------------
    #
    # Search only uses the already identified physical boundary nodes from
    # mesh2. No boundary geometry is reconstructed from extent2.

    tree = cKDTree(nodes_big_search)
    boundary_nodes_big = {}

    for face, face_nodes in boundary_faces.items():

        if face_nodes.size == 0:
            boundary_nodes_big[face] = np.empty((0, 4), dtype=float)
            continue

        _, idx_closest = tree.query(face_nodes)
        idx_closest = np.asarray(idx_closest, dtype=int)

        closest_nodes_big = nodes_big[idx_closest]

        boundary_nodes_big[face] = np.hstack(
            [
                idx_closest.reshape(-1, 1),
                closest_nodes_big
            ]
        )


    return mesh1, mesh2, boundary_nodes_big
