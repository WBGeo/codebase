import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
import gmsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Tuple, Dict, Any, Union, Optional
from numpy.typing import NDArray
from scipy.interpolate import Rbf
from core.object_components import  ExtentData
from core.meshing_components.explicit.unstructured.refinement_mesh import (Refinement, LinearWellRefinement, FunctionWellRefinement,
                                                               LinearSourceRefinement,FunctionSourceRefinement,)
from core.meshing_components.explicit.unstructured.refinement_fields import (build_refinement_fields, apply_background_fields,build_triangulation_mesh_callback,build_combined_mesh_callback)


def plot_surfaces_individually(interpolated_surfaces: List[NDArray[np.float64]]) -> None:
    """
    Plot each interpolated surface in a separate 3D figure.
    Each surface is visualized using a triangular surface plot (`matplotlib.axes.Axes3D.plot_trisurf`) based on its (x, y, z) coordinates.

    Args:
        interpolated_surfaces: List of interpolated surfaces. Each surface is a NumPy array of shape (N, 3), where columns represent:
              - surface[:, 0] → x-coordinates
              - surface[:, 1] → y-coordinates
              - surface[:, 2] → z-coordinates

    Returns:
        None: The function produces matplotlib figures as a side effect.
    """
    for i, surface in enumerate(interpolated_surfaces, start=1):
        if surface is None or len(surface) == 0:
            continue
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(surface[:, 0], surface[:, 1], surface[:, 2], edgecolor="none")
        ax.set_title(f"Surface {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


def plot_interpolated_surfaces(interpolated_surfaces, cmap="plasma", alpha=0.9):
    """
    Plot a list of interpolated 3D surfaces.
    Each surface is assumed to be (N, 3): x, y, z.
    """

    for i, surface in enumerate(interpolated_surfaces, start=1):

        if surface is None or len(surface) == 0:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Extract coordinates
        x = surface[:, 0]
        y = surface[:, 1]
        z = surface[:, 2]

        # Plot triangulated surface
        ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor="none", alpha=alpha)

        # Labels
        ax.set_title(f"Interpolated Surface {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Improve visual consistency across plots
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

        plt.tight_layout()
        plt.show()


def plot_all_surfaces_together(interpolated_surfaces: List[NDArray[np.float64]]) -> None:
    """
    Plot all interpolated surfaces in a single 3D figure.
    Each surface is visualized using a triangular surface plot (plot_trisurf) with different colors.

    Args:
        interpolated_surfaces: List of interpolated surfaces. Each surface is a NumPy array of shape (N, 3).

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap("tab10")

    for i, surface in enumerate(interpolated_surfaces):

        if surface is None or len(surface) == 0:
            continue

        color = cmap(i % 10)

        ax.plot_trisurf(
            surface[:, 0],
            surface[:, 1],
            surface[:, 2],
            color=color,
            edgecolor="none",
            alpha=0.7
        )

    ax.set_title("All Interpolated Surfaces")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def create_surface_grid(cleaned_surfaces: List[Tuple[int, NDArray[np.float64]]], smooth: float = 1e-5) -> List[NDArray[np.float64]]:
    """
    Generate interpolated surface grids from cleaned geological surface points.
    This function processes a list of cleaned surfaces, where each surface consists
    of scattered (x, y, z) points. It removes duplicates, averages overlapping points,
    and applies Radial Basis Function (RBF) interpolation to generate a structured
    grid representation of each surface.
    The function automatically distinguishes between smooth and irregular surfaces:
        - Smooth surfaces are normalized before interpolation to improve numerical stability.
        - Irregular surfaces are interpolated directly in physical coordinates.
    Grid resolution is dynamically determined based on the number of unique x and y
    coordinates, with upper limits to control computational cost.

    Args:
        cleaned_surfaces (List[Tuple[int, NDArray[np.float64]]]):
        List of tuples containing:
                - surface ID (int)
                - array of shape (N, 3) with columns [x, y, z]
        smooth (float, optional):
            Smoothing parameter for RBF interpolation. Smaller values produce closer fits
            to the data, while larger values produce smoother surfaces. Default is 1e-5.

    Returns:
        List[NDArray[np.float64]]: List of interpolated surfaces, each represented as an array of shape (M, 3),
            where M = n_gx * n_gy (flattened grid points).
    """

    interpolated_surfaces: List[NDArray[np.float64]] = []

    max_n_gx = 250
    max_n_gy = 100

    for surf_id, points in cleaned_surfaces:

        # Detect surface orientation
        dx_raw = np.ptp(points[:, 0])
        dy_raw = np.ptp(points[:, 1])
        dz_raw = np.ptp(points[:, 2])

        tol = 1e-5
        is_vertical_yz = np.ptp(points[:, 0]) < tol

        # Grouping
        if not is_vertical_yz:
            # z = f(x,y)
            df = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2]}).drop_duplicates()
            df = df.groupby(["x","y"], as_index=False)["z"].mean()

            x_cleaned = df['x'].values
            y_cleaned = df['y'].values
            z_cleaned = df['z'].values

        else:
            df = pd.DataFrame({'y': points[:,1], 'z': points[:,2], 'x': points[:,0]}).drop_duplicates()
            df = df.groupby(["y","z"], as_index=False)["x"].mean()

            y_cleaned = df['y'].values
            z_cleaned = df['z'].values
            x_cleaned = df['x'].values

        # Bounds
        x_min, x_max = x_cleaned.min(), x_cleaned.max()
        y_min, y_max = y_cleaned.min(), y_cleaned.max()
        z_min, z_max = z_cleaned.min(), z_cleaned.max()

        z_range = z_max - z_min
        dx = max(x_max - x_min, 1e-12)
        dy = max(y_max - y_min, 1e-12)

        roughness = np.std(z_cleaned) / max(dx, dy)
        unique_x = np.unique(x_cleaned)
        unique_y = np.unique(y_cleaned)

        n_gx = max(min(len(unique_x), max_n_gx), 4)
        n_gy = max(min(len(unique_y), max_n_gy), 4)
        # RBF grid covers the full extent of the input surface points.
        # For no-fault models these come from interpolation on a padded grid,
        # so they already extend past the model bounding box — no artificial
        # expansion needed here.

        gx_min, gx_max = x_min, x_max
        gy_min, gy_max = y_min, y_max
        gz_min, gz_max = z_min, z_max

        if  not is_vertical_yz:
                grid_x, grid_y = np.meshgrid(np.linspace(gx_min, gx_max, n_gx),
                                             np.linspace(gy_min, gy_max, n_gy))

                rbf = Rbf(x_cleaned, y_cleaned, z_cleaned, function='multiquadric', epsilon=2, smooth=smooth)
                z_interp_real = rbf(grid_x, grid_y)
                grid_x_real, grid_y_real = grid_x, grid_y
        else:
                grid_y, grid_z = np.meshgrid(np.linspace(gy_min, gy_max, n_gx),
                                             np.linspace(gz_min, gz_max, n_gy))

                rbf = Rbf(y_cleaned, z_cleaned, x_cleaned, function='multiquadric', epsilon=2, smooth=smooth)
                x_interp_real = rbf(grid_y, grid_z)
                grid_y_real, grid_z_real = grid_y, grid_z

        # Final stacking
        if  not is_vertical_yz:
            interpolated_grid = np.column_stack((grid_x_real.flatten(),
                                                 grid_y_real.flatten(),
                                                 z_interp_real.flatten()))

        else:
            interpolated_grid = np.column_stack((x_interp_real.flatten(),
                                                 grid_y_real.flatten(),
                                                 grid_z_real.flatten()))

        interpolated_surfaces.append(interpolated_grid)
    #plot_surfaces_individually(interpolated_surfaces)
    #plot_all_surfaces_together(interpolated_surfaces)
    return interpolated_surfaces


def import_surfaces(interpolated_s: List[NDArray[np.float64]], extent: Optional[ExtentData] = None, tolerance: float = 50.0,) -> Tuple[List[int], List[float]]:
    """
    Import interpolated surface grids into GMSH as B-spline surfaces and
    determine an adjusted model bounding box.
    Each interpolated surface is assumed to be a structured (x, y, z) grid
    that can be reshaped into a tensor-product B-spline surface. GMSH OCC
    B-spline surfaces are created using the inferred grid resolution.
    The function also computes surface-wise bounding coordinates and
    optionally adjusts a provided extent to match the imported surfaces
    within a given tolerance.

    Args:
        interpolated_s: List of interpolated surface point clouds. Each surface must be a NumPy array of shape (N, 3), with columns representing (x, y, z).
        extent: Optional bounding box defined as (xmin, xmax, ymin, ymax, zmin, zmax). If provided, bounds are adjusted to match
                surface limits within the specified tolerance.
        tolerance: Maximum allowed deviation (in model units) when matching surface bounds to the provided extent.

    Returns:
        surfaces: List of GMSH OCC B-spline surface tags created from the input surfaces.
        bounds: Adjusted bounding box (xmin, xmax, ymin, ymax, zmin, zmax) that best fits all imported surfaces while respecting the tolerance.
               Returns None if no extent was provided or no surfaces were created.
    """

    # Initialize an empty list to store GMSH surface IDs.
    surfaces: List[int] = []

    # Initialize lists to store the minimum and maximum x, y, z coordinates of each surface.
    min_x_list: List[float] = []
    max_x_list: List[float] = []
    min_y_list: List[float] = []
    max_y_list: List[float] = []
    min_z_list: List[float] = []
    max_z_list: List[float] = []

    # Iterate through each interpolated surface in the input list
    for surface_points in interpolated_s:

        # Ensure that each element is a NumPy array with 3 columns (x, y, z)
        if not isinstance(surface_points, np.ndarray) or surface_points.shape[1] != 3:
            print("Invalid surface points format")
            continue  # Skip invalid surfaces

        # Compute min and max coordinates for this surface and store them
        min_x_list.append(float(np.min(surface_points[:, 0])))
        max_x_list.append(float(np.max(surface_points[:, 0])))
        min_y_list.append(float(np.min(surface_points[:, 1])))
        max_y_list.append(float(np.max(surface_points[:, 1])))
        min_z_list.append(float(np.min(surface_points[:, 2])))
        max_z_list.append(float(np.max(surface_points[:, 2])))

        # Detect truly vertical (YZ-plane) surfaces: X range is negligible compared to Y/Z range.
        # For these, the RBF produces nearly-identical X floats that all appear "unique" to
        # np.unique, making numPointsU explode and causing the mismatch check to skip the surface.
        x_range = float(np.ptp(surface_points[:, 0]))
        y_range = float(np.ptp(surface_points[:, 1]))
        z_range = float(np.ptp(surface_points[:, 2]))

        is_planar_x = x_range < 0.05 * max(y_range, z_range, 1e-12)
        if is_planar_x:
            # Create a rectangular plane surface spanning the full Y/Z extent of the surface.
            x_c = float(np.mean(surface_points[:, 0]))
            y0, y1 = float(np.min(surface_points[:, 1])), float(np.max(surface_points[:, 1]))
            z0, z1 = float(np.min(surface_points[:, 2])), float(np.max(surface_points[:, 2]))
            p1 = gmsh.model.occ.addPoint(x_c, y0, z0)
            p2 = gmsh.model.occ.addPoint(x_c, y1, z0)
            p3 = gmsh.model.occ.addPoint(x_c, y1, z1)
            p4 = gmsh.model.occ.addPoint(x_c, y0, z1)

            loop = gmsh.model.occ.addCurveLoop([
                gmsh.model.occ.addLine(p1, p2),
                gmsh.model.occ.addLine(p2, p3),
                gmsh.model.occ.addLine(p3, p4),
                gmsh.model.occ.addLine(p4, p1),
            ])

            s = gmsh.model.occ.addPlaneSurface([loop])
            surfaces.append(s)
            continue

        # Extract x and y coordinates to identify the grid structure
        x = surface_points[:, 0]
        y = surface_points[:, 1]

        # Unique x and y values define the surface grid resolution
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        numPointsU: int  = len(unique_x)  # Number of control points in the U direction
        numPointsV: int = len(unique_y)  # Number of control points in the V direction

        # Create a list of GMSH point IDs for this surface
        ps: List[int] = []
        for i in range(numPointsU):
            for j in range(numPointsV):
                index = i * numPointsV + j

                if index < len(surface_points):
                    point = surface_points[index]

                    # Create a GMSH point for each grid node
                    ps.append(gmsh.model.occ.addPoint(point[0], point[1], point[2]))

        # Check if the expected number of points matches the actual number
        if len(ps) != numPointsU * numPointsV:

            print(f"Warning: Skipping B-spline surface due to mismatch in control points: "
                  f"{len(ps)} != {numPointsU * numPointsV}")
            continue

        # Create a B-spline surface using the list of GMSH points
        s = gmsh.model.occ.addBSplineSurface(ps, numPointsU=numPointsU)

        surfaces.append(s)


    # Define an internal helper function to adjust bounds to fit within tolerance
    def find_adjusted_bound(bound_list: List[float], target_value: float, mode: str = 'max') -> float:
        """
        Returns the closest bound within tolerance to the target_value.
        The 'mode' parameter controls whether the function seeks a minimum or maximum bound.
        """

        # Sort bounds: descending for max mode, ascending for min mode
        sorted_list = sorted(bound_list, reverse=(mode == 'max'))

        for val in sorted_list:
            # Check if the value is within the acceptable tolerance range

            if abs(val - target_value) <= tolerance:
                # Prevent over-adjusting beyond the target

                if (mode == 'min' and val > target_value) or (mode == 'max' and val < target_value):
                    return target_value

                else:

                    return val

        # If no suitable value is found, return the original target
        print(f"Warning: No bounds found within ±{tolerance} of {target_value}")

        return target_value

    # Initialize the bounding box output
    bounds:  List[float] = None

    # Ensure the extent tuple is valid before processing
    extent = tuple(extent)

    # If surfaces were successfully created and extent is provided
    if surfaces and extent:
        # Unpack extent into individual boundary coordinates

        x_b_min, x_b_max, y_b_min, y_b_max, z_b_min, z_b_max = extent

        # Compute adjusted bounds using tolerance-based matching
        bounds = (
            find_adjusted_bound(min_x_list, x_b_min, mode='max'),
            find_adjusted_bound(max_x_list, x_b_max, mode='min'),
            find_adjusted_bound(min_y_list, y_b_min, mode='max'),
            find_adjusted_bound(max_y_list, y_b_max, mode='min'),
            find_adjusted_bound(min_z_list, z_b_min, mode='max'),
            find_adjusted_bound(max_z_list, z_b_max, mode='min')
        )

    # Confirmation message after successful import

    print('B-spline surfaces have been imported!')
    # Return the list of surface IDs and the final bounding box
    return surfaces, bounds


def fragment_surfaces(surfaces: List[int], extent: List[float], ref_surface_indices: Dict[int, int], wells: List[Tuple[float, ...]],
    extra_planes: List[Tuple[float, ...]], source_points: List[Tuple[float, ...]], mine_shafts: List[Dict[str, Any]], ellipses: List[Dict[str, Any]] = [],
    triangulations: List[np.ndarray] = [] , mesh_size: float = 20.0, curve_mesh_size: float = 5.0, refinement: Optional[Refinement] = None,) -> Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]],
    List[int], List[int], List[int], Dict[int, List[int]], List[int]]:

    """
    Creates and fragments geological surfaces using GMSH by embedding additional
    engineering and geological objects (wells, sources, extra planes, mine shafts,
    ellipses, and triangulated surfaces) into a bounded model domain.
    The function builds a 3D bounding box from the provided extent and adds source points, wells (as polylines), extra planes, ellipses, and mine shafts (as cylinders)
    to the box. It clamps all geometries to remain inside the model extent. It then creates triangulated surfaces from given point sets.
    It performs Boolean fragmentation between surfaces, volumes, and embedded objects and assigns physical groups for later meshing and post-processing

    Args:
        surfaces (List[int]): GMSH surface tags to be fragmented.
        extent (List[float]): Model bounding box given as (x_min, x_max, y_min, y_max, z_min, z_max).
        ref_surface_indices (Dict[int, int]): Mapping that identifies reference surfaces (faults) and their indices
                        in the `surfaces` list. Used to preserve and track fault surfaces.
        wells (List[Tuple[float, ...]]): Well trajectories defined by sequences of (x, y, z) coordinates. Each tuple
                       length must be a multiple of 3.
        extra_planes (List[Tuple[float, ...]]): Additional planes defined by 4 corner points (x1, y1, z1, ..., x4, y4, z4).
        source_points (List[Tuple[float, ...]]): Point sources defined as (x, y, z).
        mine_shafts (List[Dict[str, Any]]): Cylindrical mine shafts with keys:
                "center": (x, y, z), "axis": (dx, dy, dz), "radius": float
        ellipses (List[Dict[str, Any]], optional): Elliptical surfaces with keys:
                "center": (x, y, z), "radii": (r1, r2), "angle1": start angle in radians (default=0.0), "angle2": end angle in radians (default=2*pi),
                "zAxis": z-axis of the ellipse plane (default=[0,0,1]), "xAxis": x-axis of the ellipse plane (optional)
        triangulations (List[np.ndarray], optional): List of point arrays to create triangulated surfaces.
        mesh_size (float, optional): Target mesh size for surfaces and volumes. Default is 20.0.
        curve_mesh_size (float, optional): Finer mesh size applied to curves and embedded lines. Default is 5.0.
        refinement : Optional: Refinement configuration controlling meshes around wells, sources, faults, etc.
    Returns:
        ov (List[Tuple[int, int]]): Original 3D entities (volumes) before fragmentation. Each tuple is (dimension, tag).
        well_tags (List[int]): Physical group tags assigned to all well trajectories after fragmentation.
        shaft_tags (List[int]): Physical group tags assigned to all mine shaft volumes after fragmentation.
        tri_group_tags (List[int]): Physical group tags assigned to triangulated *groups* (intermediate grouping before surface tagging).
        tri_surface_tags (List[int]): Physical group tags assigned to triangulated surfaces created from input point sets.
        source_tag (List[int]): Physical group tags assigned to embedded source points (0D entities) after fragmentation.
        boudrary_tags (List(int)): physical group tags of outer boundary surfaces.
    """

    outside_threshold = 0

    ##############
    # CREAT A BOX
    ##############
    x_min, x_max, y_min, y_max, z_min, z_max = extent

    v: int = gmsh.model.occ.addBox(x_min , y_min , z_min, x_max - x_min, y_max - y_min, z_max - z_min)

    def clamp_to_nearest_boundary(val: float, min_val: float, max_val: float) -> float:
      """
        Clamp a coordinate value to the nearest model boundary.

        If the value lies outside the [min_val, max_val] interval,
        it is snapped to the nearest boundary.
      """

      if val < min_val:
        # outside below min, snap to min
        return min_val

      elif val > max_val:
        # outside above max, snap to max
        return max_val

      else:
        # inside, keep original
       return val

    ###################
    # ADD SOURCE POINTS
    # ################
    if source_points:

        source: List[int]=[]

        for i in range(len(source_points)):

            x, y, z = source_points[i][0], source_points[i][1], source_points[i][2]
            p_s: int = gmsh.model.occ.addPoint(x,y,z)

            source.append(p_s)

        gmsh.model.occ.synchronize()

    #########################
    # ADD WELLS AS POLYLINES
    ########################
    if wells:

        well_lines: List[List[int]] = []

        for coords in wells:

            points: List[int] = []
            # create clamped points

            for j in range(0, len(coords), 3):

                x, y, z = coords[j], coords[j+1], coords[j+2]
                x = clamp_to_nearest_boundary(x, x_min, x_max)
                y = clamp_to_nearest_boundary(y, y_min, y_max)
                z = clamp_to_nearest_boundary(z, z_min, z_max)
                p = gmsh.model.occ.addPoint(x, y, z)

                points.append(p)

            # skip invalid wells
            if len(points) < 2:

                print("Skipping degenerate well")
                continue

            # create lines
            lines: List[int] = []

            for k in range(len(points) - 1):

                line = gmsh.model.occ.addLine(points[k], points[k+1])
                lines.append(line)

            well_lines.append(lines)

        gmsh.model.occ.synchronize()

    ####################
    # ADD EXTRA PLANES
    ###################
    if extra_planes:

        extra_planes_surface_indices = set()

        for i_layer in range(len(extra_planes)):

            (x1_m, y1_m, z1_m,
            x2_m, y2_m, z2_m,
            x3_m, y3_m, z3_m,
            x4_m, y4_m, z4_m) = extra_planes[i_layer]

            # Clamp ALL points to extent
            x1_m = clamp_to_nearest_boundary(x1_m, x_min, x_max)
            y1_m = clamp_to_nearest_boundary(y1_m, y_min, y_max)
            z1_m = clamp_to_nearest_boundary(z1_m, z_min, z_max)

            x2_m = clamp_to_nearest_boundary(x2_m, x_min, x_max)
            y2_m = clamp_to_nearest_boundary(y2_m, y_min, y_max)
            z2_m = clamp_to_nearest_boundary(z2_m, z_min, z_max)

            x3_m = clamp_to_nearest_boundary(x3_m, x_min, x_max)
            y3_m = clamp_to_nearest_boundary(y3_m, y_min, y_max)
            z3_m = clamp_to_nearest_boundary(z3_m, z_min, z_max)

            x4_m = clamp_to_nearest_boundary(x4_m, x_min, x_max)
            y4_m = clamp_to_nearest_boundary(y4_m, y_min, y_max)
            z4_m = clamp_to_nearest_boundary(z4_m, z_min, z_max)

            # Create OCC points
            p1 = gmsh.model.occ.addPoint(x1_m, y1_m, z1_m)
            p2 = gmsh.model.occ.addPoint(x2_m, y2_m, z2_m)
            p3 = gmsh.model.occ.addPoint(x3_m, y3_m, z3_m)
            p4 = gmsh.model.occ.addPoint(x4_m, y4_m, z4_m)

            # Create lines
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)

            # Surface
            loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

            surface_mine = gmsh.model.occ.addPlaneSurface([loop])
            surfaces.append(surface_mine)

            ref_surface_indices[len(surfaces) - 1] = len(surfaces) - 1
            extra_planes_surface_indices.add(len(surfaces) - 1)

        gmsh.model.occ.synchronize()

    ###############
    # ADD ELLIPSES
    ##############
    if ellipses:

        ellipse_surface_indices = set()

        for ell in ellipses:

            cx, cy, cz = ell["center"]
            r1, r2 = ell["radii"]
            angle1 = ell.get("angle1", 0.0)
            angle2 = ell.get("angle2", 2*np.pi)
            z_axis = ell.get("zAxis", [0,0,1])
            x_axis = ell.get("xAxis", None)

            # build kwargs safely
            kwargs = dict(
                angle1=angle1,
                angle2=angle2,
                zAxis=z_axis
            )

            if x_axis is not None:

                kwargs["xAxis"] = x_axis

            ellipse_curve = gmsh.model.occ.addEllipse(
                cx, cy, cz, r1, r2,
                **kwargs
            )

            gmsh.model.occ.synchronize()
            # If partial ellipse, create lines connecting start/end to center

            if angle2 - angle1 < 2*np.pi:
                # Get start and end points of the ellipse arc

                start_x = cx + r1 * np.cos(angle1)
                start_y = cy + r2 * np.sin(angle1)
                start_z = cz

                end_x   = cx + r1 * np.cos(angle2)
                end_y   = cy + r2 * np.sin(angle2)
                end_z   = cz

                p_start = gmsh.model.occ.addPoint(start_x, start_y, start_z)
                p_end   = gmsh.model.occ.addPoint(end_x, end_y, end_z)
                p_center = gmsh.model.occ.addPoint(cx, cy, cz)

                # Create lines from center to start/end points
                l1 = gmsh.model.occ.addLine(p_center, p_start)
                l2 = gmsh.model.occ.addLine(p_center, p_end)

                # Create curve loop including the arc + connecting lines
                ellipse_loop = gmsh.model.occ.addCurveLoop([-l2, ellipse_curve, l1])

            else:

                # Full ellipse
                ellipse_loop = gmsh.model.occ.addCurveLoop([ellipse_curve])

            # create surfaces
            ellipse_surface = gmsh.model.occ.addPlaneSurface([ellipse_loop])
            # Store for fragmentation

            surfaces.append(ellipse_surface)
            ref_surface_indices[len(surfaces) - 1] = len(surfaces) - 1

            ellipse_surface_indices.add(len(surfaces) - 1)

        gmsh.model.occ.synchronize()

    #############################
    # ADD TRIANGULATED SURFACES
    #############################
    if triangulations is not None and len(triangulations) > 0:

        from scipy.spatial import Delaunay
        from sklearn.decomposition import PCA

        triangulations = np.asarray(triangulations)
        # -------------------------------------------------
        # Support:
        # [x,y,z]                  -> single triangulation group
        # [x,y,z,id]               -> multiple triangulation groups
        # -------------------------------------------------
        if triangulations.shape[1] == 3:

            tri_groups = {1: triangulations}

        else:

            tri_groups = {}

            for row in triangulations:

                x, y, z, pid = row
                pid = int(pid)
                tri_groups.setdefault(pid, []).append([x, y, z])

            for pid in tri_groups:

                tri_groups[pid] = np.asarray(
                    tri_groups[pid],
                    dtype=np.float64
                )

        # bookkeeping
        triangulated_surface_tags = []

        # maps triangulation id -> gmsh surfaces
        tri_surface_to_child_fragments = {}
        tri_surface_indices = set()

        # Create triangulations separately for each file/id

        for pid, points in tri_groups.items():

            if len(points) < 3:
                continue

            # Create gmsh points

            pt_tags = []

            for p in points:

                tag = gmsh.model.occ.addPoint(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    mesh_size
                )

                pt_tags.append(tag)

            # Project to best-fit plane
            pca = PCA(n_components=2)

            points_2d = pca.fit_transform(points)

            # Local triangulation for THIS group only
            tri = Delaunay(points_2d)

            # avoid duplicate edges inside same triangulation
            edge_map = {}

            # store surfaces belonging to this triangulation
            tri_surface_to_child_fragments[pid] = []

            for simplex in tri.simplices:

                try:
                    ids = [
                        simplex[0],
                        simplex[1],
                        simplex[2]
                    ]

                    edges = [
                        (ids[0], ids[1]),
                        (ids[1], ids[2]),
                        (ids[2], ids[0])
                    ]

                    line_tags = []

                    for a, b in edges:

                        key = tuple(sorted((a, b)))

                        if key in edge_map:

                            line = edge_map[key]

                        else:

                            line = gmsh.model.occ.addLine(
                                pt_tags[a],
                                pt_tags[b]
                            )

                            edge_map[key] = line

                        line_tags.append(line)

                    loop = gmsh.model.occ.addCurveLoop(line_tags)
                    surf = gmsh.model.occ.addPlaneSurface([loop])

                    # IMPORTANT bookkeeping
                    surfaces.append(surf)

                    # IMPORTANT:
                    # add to ref_surface_indices
                    ref_surface_indices[surf] = surf
                    triangulated_surface_tags.append(surf)

                    tri_surface_indices.add(surf)
                    # keep surfaces grouped by triangulation id
                    tri_surface_to_child_fragments[pid].append(surf)

                except Exception as e:

                    print(
                        f"Triangle creation failed for triangulation {pid}:",
                        e
                    )

        gmsh.model.occ.synchronize()

    ########################################
    # ADD MINE SHAFTS AS CYLINDERAL VOLUMES
    ########################################
    mine_shaft_volumes: List[int] = []

    if mine_shafts:

        for i, shaft in enumerate(mine_shafts):

            # original geometry
            x, y, z = shaft["center"]
            dx, dy, dz = shaft["axis"]
            x2, y2, z2 = x + dx, y + dy, z + dz

            # clamp endpoints
            x = clamp_to_nearest_boundary(x, x_min, x_max)
            y = clamp_to_nearest_boundary(y, y_min, y_max)
            z = clamp_to_nearest_boundary(z, z_min, z_max)

            x2 = clamp_to_nearest_boundary(x2, x_min, x_max)
            y2 = clamp_to_nearest_boundary(y2, y_min, y_max)
            z2 = clamp_to_nearest_boundary(z2, z_min, z_max)

            # recompute final vector
            dx = x2 - x
            dy = y2 - y
            dz = z2 - z

            axis_len = np.linalg.norm([dx, dy, dz])
            # skip degenerate shafts

            if axis_len < 1e-10:

                print(f"Skipping degenerate shaft {i}")
                continue

            shaft["center"] = (x, y, z)
            shaft["axis"] = (dx, dy, dz)

            r = shaft["radius"]

            tag = gmsh.model.occ.addCylinder(
                x, y, z, dx, dy, dz, r, 2500 + i + 1

            )

            mine_shaft_volumes.append(tag)

    gmsh.model.occ.synchronize()

    ########################################
    # CREATE TOOL_ENTITIES FOR FRAGMENTATION
    ########################################
    tool_entities: List[Tuple[int, int]] = [(2, s) for s in surfaces]  # start with surfaces

    if wells:
        all_well_lines: List[int]  = [l for sublist in well_lines for l in sublist]
        tool_entities += [(1, l) for l in all_well_lines]

    if source_points:
        tool_entities += [(0, p) for p in source]

    if mine_shafts:
        tool_entities += [(3, tag) for tag in mine_shaft_volumes]

    #################
    # FRAGMENTATION
    #################
    ov: List[Tuple[int, int]]
    ovv: List[List[Tuple[int, int]]]
    ov, ovv = gmsh.model.occ.fragment(
        [(3, v)] + tool_entities, [],
        removeObject=True,
        removeTool=True
    )

    gmsh.model.occ.synchronize()

    ####################################
    # INPUT FOR ZOP (PARENT ---> CHILD)
    ####################################
    zip_inputs: List[Tuple[int, int]] = [(3, v)] + [(2, s) for s in surfaces]

    if wells:
        zip_inputs += [(1, l) for l in all_well_lines]

    if source_points:
        zip_inputs += [(0, p) for p in source]

    if mine_shafts:
        # Flatten shaft volumes in case some are lists
        flat_mine_shaft_volumes = []

        for shaft in mine_shaft_volumes:
            flat_mine_shaft_volumes.extend(shaft if isinstance(shaft, list) else [shaft])

        zip_inputs += [(3, tag) for tag in flat_mine_shaft_volumes]
    ##########################
    # FIND FAULT SURFACES
    #########################
    counter=0
    if ref_surface_indices:

        for _, ref_index in ref_surface_indices.items():
                # SKIP TRIANGULATED SURFACES HERE

                if triangulations is not None and len(triangulations) > 0:

                    if ref_index in tri_surface_indices:
                        continue

                if ellipses:

                    if ref_index in  ellipse_surface_indices:
                        continue

                if extra_planes:

                    if ref_index in  extra_planes_surface_indices:
                        continue

                counter=1
                # Now zip with ovv
                zip_info = list(zip(zip_inputs, ovv))

                print("before/after fragment relations:", ref_index)

                for e in zip_info:
                    print("parent " + str(e[0]) + " -> child " + str(e[1]))

                zipped_list = list(zip_info)  # Convert zip object to a list

                for i in range(1, len(zipped_list)):

                    if i == ref_index + 1:
                        values = list(zipped_list[i][1])
                        fault_surf = values

    # Filter to get only 3D volumes from ov
    fragmented_volumes : List[Tuple[int, int]] = [entity for entity in ov if entity[0] == 3]

    print(f"Number of volumes created: {len(fragmented_volumes)}")
    print("Volume tags:", [tag for dim, tag in fragmented_volumes])

    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)

    gmsh.model.occ.synchronize()

    ##################
    # Mesh refinement
    ##################
    tri_cfg = (
        refinement.triangulation
        if refinement and refinement.triangulation
        else None
    )

    ell_cfg = (
        refinement.ellipses
        if refinement and refinement.ellipses
        else None
    )

    fault_cfg = (
        refinement.faults
        if refinement and refinement.faults
        else None
    )

    # callbavk
    if counter==0:
        fault_surf=None

    cb = build_combined_mesh_callback(
        triangulations=triangulations,
        ellipses=ellipses,
        fault_surfaces=surfaces,
        fault_fragments=fault_surf,
        tri_cfg=tri_cfg,
        ell_cfg=ell_cfg,
        fault_cfg=fault_cfg
    )

    if cb is not None:
        gmsh.model.mesh.setSizeCallback(cb)

    # Wells + Sources + triangulation
    if not source_points:
        source=None

    if not wells:
        well_lines=None

    if len(triangulations) == 0:
        triangulations=None

    zip_info = list(zip(zip_inputs, ovv))

    active_fields = build_refinement_fields(
        zip_info=zip_info,
        well_lines=well_lines,
        source_points=source,
        triangulations=triangulations,
        refinement=refinement
    )

    apply_background_fields(active_fields)

    ##############################################################
    # REMOVING SURFACES OUTSIDE BOX + PHYSICAL GROUPS FOR SURFACES
    ##############################################################
    tagsss: List[int] = []  # List to store physical groups

    # Handle the case when there are ref_surface_indices
    if ref_surface_indices:
        # Mesh generation for surfaces
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()

        # input for zip (parent -> child)
        zip_inputs:  List[Union[int, Tuple[int, int]]] = [v] + surfaces  # always include v and surfaces

        if wells:
            all_well_lines = [l for sublist in well_lines for l in sublist]
            zip_inputs += all_well_lines

        if source_points:
            all_source_points = [(0, p) for p in source]
            zip_inputs += all_source_points

        if mine_shafts:
            # Flatten shaft volumes in case some are lists
            flat_mine_shaft_volumes = []

            for shaft in mine_shaft_volumes:
                flat_mine_shaft_volumes.extend(shaft if isinstance(shaft, list) else [shaft])

            zip_inputs += flat_mine_shaft_volumes

        # Loop through fault surfaces

        for _, ref_index in ref_surface_indices.items():
            # SKIP TRIANGULATED SURFACES HERE

            if triangulations is not None and len(triangulations) > 0:

                if ref_index in tri_surface_indices:
                    continue

            # Now zip with ovv
            zip_info = list(zip(zip_inputs, ovv))
            print("before/after fragment relations:", ref_index)

            for e in zip_info:
                print("parent " + str(e[0]) + " -> child " + str(e[1]))

            zipped_list = list(zip_info)  # Convert zip object to a list

            for i in range(1, len(zipped_list)):

                if i == ref_index + 1:
                    values = [item[1] for item in zipped_list[i][1]]
                    # Filter surfaces outside the box
                    filtered_surfaces = []

                    for surface in values:
                        nodeTags, coords, _ = gmsh.model.mesh.getNodes(2, surface)
                        # Skip empty surfaces (when no nodes are present)
                        #if len(coords) < 3:
                        #    filtered_surfaces.append(surface)
                        #    continue

                        # Convert flattened list to (x, y, z) tuples

                        node_coords = np.array(coords).reshape(-1, 3)  # shape (n_nodes, 3)
                        # Check if any node is outside the box

                        outside_mask = (
                            (node_coords[:,0] < x_min - outside_threshold) |
                            (node_coords[:,0] > x_max + outside_threshold) |
                            (node_coords[:,1] < y_min - outside_threshold) |
                            (node_coords[:,1] > y_max + outside_threshold) |
                            (node_coords[:,2] < z_min - outside_threshold) |
                            (node_coords[:,2] > z_max + outside_threshold)
                        )

                        if np.any(outside_mask):
                            # At least one node is outside → remove surface
                            filtered_surfaces.append(surface)
                        # Check if any node is outside the adjusted bounding box

                        for x, y, z in node_coords:
                            if mine_shafts:
                              # Check if any node is inside a mine shaft cylinder

                              for shaft in mine_shafts:
                                cx, cy, cz = shaft["center"]
                                dx, dy, dz = shaft["axis"]
                                r = shaft["radius"]
                                h = shaft.get("height", ((dx**2 + dy**2 + dz**2) ** 0.5))

                                # Normalize axis
                                length = (dx**2 + dy**2 + dz**2) ** 0.5
                                dx /= length
                                dy /= length
                                dz /= length

                                vx = x - cx
                                vy = y - cy
                                vz = z - cz
                                dot = vx * dx + vy * dy + vz * dz

                                # Check if point is within the cylinder's height
                                if 0 <= dot <= h:
                                    # Project to axis
                                    projx = cx + dot * dx
                                    projy = cy + dot * dy
                                    projz = cz + dot * dz

                                    dist2 = (x - projx)**2 + (y - projy)**2 + (z - projz)**2

                                    if dist2 <= r**2:
                                        filtered_surfaces.append(surface)
                                        break  # No need to check other shafts for this point

                    values = [s for s in values if s not in filtered_surfaces]
                    # Add physical group for each fault/extra plane surface
                    tagsss.append(gmsh.model.addPhysicalGroup(2, values, 1000 * i))

    ###################
    # Build parent list
    ###################
    parents = [(3, v)] + tool_entities

    well_tags: List[int] = []

    shaft_tags: List[int] = []

    shaft_to_child_fragments: Dict[int, List[int]] = {}

    source_tag: List[int] = []

    ####################################
    # WELLS (preserve grouping per well)
    ####################################
    if wells:

        well_counter = 1

        for well in well_lines:
            all_children: List[int] = []

            for line in well:

                for parent, children in zip(parents, ovv):

                    if parent == (1, line):
                        all_children.extend([tag for dim, tag in children if dim == 1])

            if all_children:
                tag = gmsh.model.addPhysicalGroup(1, all_children, 1060 + well_counter)
                well_tags.append(tag)
                well_counter += 1

    ##################
    # SHAFTS (volumes)
    ##################
    if mine_shafts:
        shaft_counter = 1

        for parent, children in zip(parents, ovv):
            dim, parent_id = parent

            if dim == 3 and parent_id in mine_shaft_volumes:
                vol_children = [tag for d, tag in children if d == 3]

                if vol_children:
                    shaft_to_child_fragments[parent_id] = vol_children.copy()
                    tag = gmsh.model.addPhysicalGroup(3, vol_children, 3500 + shaft_counter)
                    shaft_tags.append(tag)
                    shaft_counter += 1
                    # OPTIONAL: remove shaft fragments from ov (same behavior as before)
                    ov = [entry for entry in ov if not (entry[0] == 3 and entry[1] in vol_children)]

    ################
    # SOURCE POINTS
    ################
    def get_next_free_physical_tag(dim: int, start: int = 1) -> int:
        """
        Returns the next available physical group tag for a given dimension.
        """

        existing = gmsh.model.getPhysicalGroups(dim)
        used_tags = {tag for d, tag in existing}

        tag = start

        while tag in used_tags:
            tag += 1

        return tag

    if source_points:

        for parent, children in zip(parents, ovv):
            dim, parent_id = parent

            if dim == 0:
                point_children = [tag for d, tag in children if d == 0]

                for ptag in point_children:
                    tag_id = get_next_free_physical_tag(0, start=2000)

                    tag = gmsh.model.addPhysicalGroup(0, [ptag], tag_id)
                    source_tag.append(tag)

    #######################
    # TRIANGULATED SURFACES
    #######################
    print("\n==============================")
    print("BUILD TRIANGULATED GROUPS")
    print("==============================")

    tri_group_tags = []
    tri_surface_tags = []

    if triangulations is not None and len(triangulations) > 0:
        fragment_parents = [(3, v)] + tool_entities
        print("\nfragment_parents:")
        print(fragment_parents)

        # Build:
        # original surface -> fragmented child surfaces
        tri_id_to_children = {}

        print("\nFragment mapping from ovv:\n")

        for parent, children in zip(fragment_parents, ovv):
            print(f"parent {parent} -> child {children}")
            dim, parent_id = parent
            # only 2D surfaces

            if dim != 2:
                continue

            child_surfaces = []
            # collect fragmented children

            for d, tag in children:

                if d != 2:
                    continue

                try:
                    gmsh.model.getType(2, tag)
                    child_surfaces.append(tag)

                except:
                    pass

            # if no children:
            # surface survived unchanged
            if len(child_surfaces) == 0:

                try:
                    gmsh.model.getType(2, parent_id)
                    child_surfaces = [parent_id]

                except:
                    child_surfaces = []

            tri_id_to_children[parent_id] = child_surfaces

        # DEBUG
        print("\ntri_id_to_children:")
        print(tri_id_to_children)
        print("\ntri_surface_to_child_fragments:")
        print(tri_surface_to_child_fragments)

        # regroup by triangulation pid
        for pid, original_surfaces in tri_surface_to_child_fragments.items():

            print("\n--------------------------------")
            print(f"PID = {pid}")
            print(f"original_surfaces = {original_surfaces}")

            merged_children = []
            # gather fragmented children

            for surf in original_surfaces:

                print(f"  surface {surf}")

                if surf in tri_id_to_children:

                    children = tri_id_to_children[surf]
                    print(f"    children = {children}")

                    merged_children.extend(children)

                else:

                    print("    no children -> keep original")

                    try:

                        gmsh.model.getType(2, surf)
                        merged_children.append(surf)

                    except:
                        pass

            # remove duplicates
            merged_children = sorted(set(merged_children))

            print(f"merged_children = {merged_children}")
            # keep only valid existing surfaces
            existing_surfaces = []

            for s in merged_children:

                try:
                    gmsh.model.getType(2, s)
                    existing_surfaces.append(s)

                except:
                    print(f"surface {s} does not exist")

            print(f"existing_surfaces = {existing_surfaces}")

            # create physical group
            if len(existing_surfaces) > 0:
                group_tag = gmsh.model.addPhysicalGroup(
                    2,
                    existing_surfaces,
                    5000 + int(pid)
                )

                gmsh.model.setPhysicalName(
                    2,
                    group_tag,
                    f"TRIANGULATED_SURFACE_{pid}"
                )

                tri_group_tags.append(group_tag)
                tri_surface_tags.extend(existing_surfaces)

    ##########
    # Get BCs
    #########
    diag = np.linalg.norm([
    x_max - x_min,
    y_max - y_min,
    z_max - z_min])

    tol = diag * 1e-4   # or 1e-5 depending on mesh precision

    def extract_boundaries_from_volumes(volumes, extent, tol=tol):

        # Collect all surfaces of all volumes
        surface_count = {}  # surface -> how many volumes share it

        for dim, vol in volumes:

            boundaries = gmsh.model.getBoundary([(3, vol)], oriented=False)

            for d, s in boundaries:

                if d != 2:
                    continue

                surface_count[s] = surface_count.get(s, 0) + 1

        # Keep ONLY external surfaces
        boundary_surfaces = [s for s, count in surface_count.items() if count == 1]

        print(f"Detected {len(boundary_surfaces)} external surfaces")
        # lassify them by location

        boundary_groups = {
            "XMIN": [],
            "XMAX": [],
            "YMIN": [],
            "YMAX": [],
            "ZMIN": [],
            "ZMAX": []
        }

        for surf in boundary_surfaces:

            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = \
                gmsh.model.occ.getBoundingBox(2, surf)

            if abs(xmin_s - x_min) <= tol and abs(xmax_s - x_min) <= tol:
                boundary_groups["XMIN"].append(surf)

            elif abs(xmin_s - x_max) <= tol and abs(xmax_s - x_max) <= tol:
                boundary_groups["XMAX"].append(surf)

            elif abs(ymin_s - y_min) <= tol and abs(ymax_s - y_min) <= tol:
                boundary_groups["YMIN"].append(surf)

            elif abs(ymin_s - y_max) <= tol and abs(ymax_s - y_max) <= tol:
                boundary_groups["YMAX"].append(surf)

            elif abs(zmin_s - z_min) <= tol and abs(zmax_s - z_min) <= tol:
                boundary_groups["ZMIN"].append(surf)

            elif abs(zmin_s - z_max) <= tol and abs(zmax_s - z_max) <= tol:
                boundary_groups["ZMAX"].append(surf)

        # Create physical groups
        boundary_tags = []
        BOUNDARY_START = 9000

        for i, (name, surfaces) in enumerate(boundary_groups.items()):

            if surfaces:

                tag = gmsh.model.addPhysicalGroup(2, surfaces, BOUNDARY_START + i)
                gmsh.model.setPhysicalName(2, tag, f"BOUNDARY_{name}")

                boundary_tags.append(tag)

        return boundary_groups, boundary_tags

    valid_volumes = gmsh.model.getEntities(3)

    boundary_groups, boundary_tags = extract_boundaries_from_volumes(

    valid_volumes, extent
)
    if not wells:
        well_tags = []

    if not mine_shafts:
        shaft_tags = []
        shaft_to_child_fragments = {}

    if triangulations is None or len(triangulations) == 0:
        tri_surface_tags = []
        tri_group_tags= []
        tri_surface_to_child_fragments = {}

    if not source_points:
        source_tag = []
    # Return updated entities

    return ov, well_tags, shaft_tags, tri_group_tags, tri_surface_tags, source_tag, boundary_tags

