import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.cluster import HDBSCAN
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict, Optional, Any
from core.object_components import StructuralModelResults

import colorcet as cc
from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
from core.object_components import StructuralModelResults
from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
from scipy.spatial import cKDTree

def get_normals(near_points: NDArray[np.floating], points: NDArray[np.floating], normal_vec: Tuple[int, NDArray[np.floating]],
        ) -> NDArray[np.floating]:
    """
    Retrieve normal vectors corresponding to a subset of surface points.

    Args:
    near_points (NDArray[float]): Subset of surface points.
    points (NDArray[float]): All points defining the surface.
    normal_vec (normal_vec: Tuple[int, NDArray[np.floating]]): Container holding normal-related arrays.

    Returns:
    NDArray[float]): Normal vectors corresponding to `near_points`.

    """
    # Find indices of near_points in points
    indices = np.where((points[:, None] == near_points).all(axis=2))[0]
    # Get corresponding normal vectors
    return normal_vec[1][indices]


def calculate_normals(points: NDArray[np.float64], cleaned_surfaces: List[Tuple[Union[int, str], NDArray[np.float64]]],
    cleaned_normals: List[Tuple[Union[int, str], NDArray[np.float64]]], file_index: Union[int, str]) -> NDArray[np.float64]:
    """
    Retrieve normal vectors corresponding to a subset of surface points.

    For a given surface identified by `file_index`, this function finds the
    indices of `points` within the stored surface point array and returns
    the matching normal vectors.

    Args:
        points (Array): contains a subset of surface points.
        cleaned_surfaces (list):   List of (surface_id, surface_points
        cleaned_normals (list): List of (surface_id, surface_normals)
        file_index: Identifier of the surface.

    Returns:
        norm (Array): contains the normal vectors corresponding to `points`, in the same order.
    """
    # Find the desired surface and its normal vectors by searching for its lable


    for label, surface in cleaned_surfaces:
        if label== file_index:
            surface_points=surface
    for label, normal in cleaned_normals:
        if label== file_index:
            normal_points=normal


    # Find the indices of points in surface_points
    indices: List[int] = []
    for p in points:
        # Check if point p is in surface_points, and get its index
        match_index = np.where(np.all(surface_points == p, axis=1))[0]

        if match_index.size > 0:
            indices.append(match_index[0])

    # Save normal vectors to norm_array
    norm_array: List[NDArray[np.float64]] = []
    for j in indices:
        norm_array.append(normal_points[j])

    norm: NDArray[np.float64] = np.vstack(norm_array)
    return norm


def correct_extrusion_direction(points: NDArray[np.float64], normals: NDArray[np.float64], file: Union[int, str],
    intersection_points: NDArray[np.float64], nearest_points_dict: Dict[Union[int, str], NDArray[np.float64]],
    EXTRUSION_FACTOR: float = 100.0, num_steps: int = 5) -> NDArray[np.float64]:
    """
    Extrude surface points along a corrected direction perpendicular to their normals.

    For each point, a perpendicular vector is constructed from its normal, and
    the extrusion direction is chosen based on proximity to reference (fault)
    intersection points versus nearby surface points.

    Args:
        points (Array): contains surface points to be extruded.
        normals (Array): contains normal vectors corresponding to `points`.
        file: Surface identifier used as a key in `nearest_points_dict`.
        intersection_points (Array): contains reference (fault) intersection points.
        nearest_points_dict (Dic): mapping surface IDs to arrays of nearby surface points,
        EXTRUSION_FACTOR: Total extrusion length.
        num_steps: Number of discrete extrusion steps.

    Returns:
        extruded_points (Array): contains extrusion paths for each input point.
    """

    intersection_tree = cKDTree(intersection_points)
    near_point_tree = cKDTree(nearest_points_dict[file])
    step_size: float = EXTRUSION_FACTOR / num_steps
    extruded_points: List[NDArray[np.float64]] = []

    for i, point in enumerate(points):
        point: NDArray[np.float64]
        normal: NDArray[np.float64]
        normal = normals[i]
        normal = normal / np.linalg.norm(normal)

        # --- Use original formula for most surfaces ---
        perpendicular_vector: NDArray[np.float64] = np.array([-normal[2], 0, normal[0]])

        perpendicular_vector /= np.linalg.norm(perpendicular_vector)

        # --- Decide extrusion direction ---
        test_point: NDArray[np.float64]  = point + EXTRUSION_FACTOR * perpendicular_vector
        d2, _ = intersection_tree.query(test_point)
        d3, _ = near_point_tree.query(test_point)

        if d3 < d2:
            d2: float
            d3: float
            perpendicular_vector *= -1

        extrusion_path = [point + j * step_size * perpendicular_vector for j in range(num_steps + 1)]
        extruded_points.append(extrusion_path)

    return np.array(extruded_points)



def plot_surfaces_excluding_ref(surfaces: List[Tuple[int, NDArray[np.float64]]], result: List[bool]) -> None:
    """
    Plot geological surfaces in two groups: reference (fault) surfaces and non-fault surfaces.

    Surfaces marked as faults in `result` are plotted separately from regular surfaces
    to allow visual inspection of fault geometry and spatial separation.

    Args:
        surfaces (list): List of surfaces.
        result (boolean list): Boolean sequence with the same length as `surfaces`, where
            True indicates a reference (fault) surface and False indicates a non-fault (regular) surface.

    Returns:
        None. Displays a Matplotlib figure with two 3D subplots.
    """

    ref_surfaces: List[Tuple[int, NDArray[np.float64]]]  = [(i, surfaces[i][1]) for i, is_fault in enumerate(result) if is_fault]
    non_ref_surfaces: List[Tuple[int, NDArray[np.float64]]]  = [(i, surfaces[i][1]) for i, is_fault in enumerate(result) if not is_fault]
    fig = plt.figure(figsize=(14, 6))

    # Plot non-fault surfaces
    ax1 = fig.add_subplot(121, projection='3d')
    for sid, points in non_ref_surfaces:
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, label=f"Surface {sid}")
    ax1.set_title("Non-Fault (Regular) Surfaces")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    # Plot fault (reference) surfaces
    if ref_surfaces:
        ax2 = fig.add_subplot(122, projection='3d')
        for sid, ref_points in ref_surfaces:
            ax2.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], s=1, label=f"Fault {sid}")
        ax2.set_title("Fault (Reference) Surfaces")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.show()



def plot_cleaned_surfaces(cleaned_surfaces: List[Tuple[int, NDArray[np.floating]]], output_file: Optional[str] = None) -> None:
    """
    Plot cleaned 3D surfaces and annotate each surface with its surface ID.

    Each surface is visualized as a 3D scatter plot, and its ID is displayed at the centroid (mean XYZ position) of the surface points.

    Args:
        cleaned_surfaces (list): list of (surface_id, points), where:
              - surface_id is an integer identifier
              - points is an (N, 3) NumPy array of XYZ coordinates
        output_file (Optional): path to save the figure as an image file. If None, the plot is displayed interactively.

    Returns:
        None. Displays or saves a Matplotlib figure.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for surface_id, surface_points in cleaned_surfaces:
            # Scatter plot the surface
            ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                    label=f"Surface {surface_id}", s=1)

            # Annotate at the mean point of surface
            center = surface_points.mean(axis=0)
            ax.text(center[0], center[1], center[2], f"{surface_id}", fontsize=10, color='black')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cleaned Surfaces with IDs")
    ax.legend(loc='upper right', fontsize='small')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_surfaces_with_extrusions(cleaned_surfaces: List[Tuple[int, NDArray[np.floating]]], extruded_points: List[NDArray[np.floating]],
    intersection_points: Optional[List[NDArray[np.floating]]] = None, point_size: int = 5) -> None:
    """
    Visualize cleaned surfaces together with extruded point paths and optional intersection points using PyVista.

    Surface points are rendered as semi-transparent point clouds. Extruded points
    (e.g., paths generated by normal-based extrusion) are automatically flattened
    to (N, 3) before visualization to ensure compatibility with PyVista.
    Intersection points, if provided, are rendered as a separate point cloud.

    Args:
        cleaned_surfaces (list): List of (surface_id, points).
        extruded_points (List):  Extruded point paths. Each array may have shape:
              - (M, 3) for flat point clouds, or
              - (K, L, 3) for extrusion paths, which will be flattened internally.
        intersection_points (Optional List): contains intersection points.
        point_size: Size of rendered points in the PyVista scene.

    Returns:
        None. Opens an interactive PyVista visualization window.
    """

    plotter = pv.Plotter()

    # Plot all surfaces
    for idx, (file, points) in enumerate(cleaned_surfaces):
        cloud = pv.PolyData(points)
        plotter.add_mesh(cloud, point_size=point_size, render_points_as_spheres=True, opacity=0.7, label=f'Surface {idx}')

    # Plot extruded points
    if extruded_points and len(extruded_points) > 0:
        extruded_all = np.vstack([p.reshape(-1, 3) if p.ndim == 3 else p for p in extruded_points])
        extruded_cloud = pv.PolyData(extruded_all)
        plotter.add_mesh(extruded_cloud, color='red', point_size=point_size, render_points_as_spheres=True, label='Extruded Points')

    # Plot intersection points if provided
    if intersection_points and len(intersection_points) > 0:
        intersection_all = np.vstack([p.reshape(-1, 3) if p.ndim == 3 else p for p in intersection_points])
        intersection_cloud = pv.PolyData(intersection_all)
        plotter.add_mesh(intersection_cloud, color='black', point_size=point_size, render_points_as_spheres=True, label='Intersection Points')

    plotter.add_legend()
    plotter.show()



def data_prepration(geomodel_result: StructuralModelResults, DISTANCE_THRESHOLD: float = 50.0, PROJECTION_THRESHOLD: float = 60.0,
    EXTRUSION_FACTOR: float = 100.0, z_threshold: float = 10.0) -> Tuple[List[Tuple[str | int, NDArray[np.float64]]],
    Dict[str | int, int], pd.DataFrame]:
    """
    Prepares geological surface data by cleaning overlapping points, identifying faults,
    clustering remaining surfaces, and computing extrusion based on proximity to reference (fault) surfaces.

    Args:
        geomodel_result: results of geological modeling.
        DISTANCE_THRESHOLD (int, optional): Distance between each geological surface to a fault surface below which points are treated as
                        overlapping and are removed. Default is 50.
        PROJECTION_THRESHOLD (int, optional): Distance used to search for points that are close enough to a fault surface to perform extrusion.
                        Default is 60.
        EXTRUSION_FACTOR (int, optional): Scale factor for extruding surface points. Default is 100.
        z_threshold (int, optional): A floating-point value that defines the vertical tolerance used to identify surfaces that are effectively at
                        the same elevation — meaning they are nearly flat or lie at similar heights across the faults.. Default is 10.

    Returns:
        cleaned_surfaces (list of tuples): Each tuple is (surface ID, cleaned surface points),representing the processed and de-overlapped surfaces.
        ref_surface_indices (list): List of surface indices (or IDs) identifying the reference fault surfaces within the `cleaned_surfaces` list.
        grid_litho (np.ndarray): Structured lithology grid that represents the geological model domain — used for assigning physical groups in meshing.
    """

    frame = geomodel_result.structural_frame
    fault_frame = frame.fault_frame
    has_faults = fault_frame is not None and len(fault_frame.fault_elements) > 0

    # Build grid_litho from the evaluated lithology block and grid coordinates.
    # grid_coordinates is (N, 3); lith_block is (nx, ny, nz) — flatten to (N,).
    grid_litho = pd.concat([
        pd.DataFrame(frame.grid.grid_coordinates, columns=["x", "y", "z"]),
        pd.DataFrame(frame.lith_block.flatten(), columns=["lithology"]),
    ], axis=1)

    # If there are faults: use interpolated surface points and their normals.
    # Geological surfaces come first (result=False), fault surfaces last (result=True).
    if has_faults:
        # get_surface_mesh_gradients returns two dicts: geological and fault elements,
        # each mapping element_name → {"points": (N,3), "vectors": (N,3)}.
        # Fault surfaces use "extended" meshes (from padded interpolation) so they
        # naturally cross the GMSH bounding box for volume fragmentation.
        geo_grads, fault_grads = surface_mesh_gradients.get_surface_mesh_gradients(
            geomodel_result, mesh_type="unmasked", fault_mesh_type="extended"
        )

        all_entries = list(geo_grads.values()) + list(fault_grads.values())
        surfaces = [(i, entry["points"]) for i, entry in enumerate(all_entries)]
        normal_surfaces = [(i, entry["vectors"]) for i, entry in enumerate(all_entries)]
        result = [False] * len(geo_grads) + [True] * len(fault_grads)
        print(result)

    # If there are no faults: collect extended surface vertices from structural elements.
    # Extended meshes come from interpolation on a padded grid and naturally cross
    # the model bounding box, enabling GMSH volume fragmentation.
    else:
        raw_surfaces = []
        for group in frame.structural_groups:
            for elem in group.structural_elements:
                try:
                    verts, _ = elem.get_mesh("extended")
                except KeyError:
                    verts, _ = elem.get_mesh("unmasked")
                if verts is not None and len(verts) > 0:
                    raw_surfaces.append(verts)
        surfaces = [(i, surface) for i, surface in enumerate(raw_surfaces)]
        result = [False] * len(surfaces)

    # Save fault surfaces in ref_surfaces
    ref_surfaces = [surfaces[i] for i, is_fault in enumerate(result) if is_fault]

     # Dictionary to store reference surface - fault surfaces- in cleaned_surfaces
    ref_surface_indices: Dict[str | int, int]  = {}
    # Call the plotting function
    ## plot_surfaces_excluding_ref(surfaces, result)

    # Remove overlapped points and store the surfaces in cleaned_surfaces and their normals in cleaned_normals
    # This step is only done if there is any fault in the model
    cleaned_surfaces: List[Tuple[str | int, NDArray[np.float64]]] = []
    cleaned_normals: List[Tuple[str | int, NDArray[np.float64]]] = []
    if ref_surfaces:
        for ref_file, ref_points in ref_surfaces:
            print(f"Using {ref_file} as reference")

            # Append reference surface and their normal vectors to cleaned_surfaces and cleaned_normals
            ref_surface_indices[ref_file] = len(cleaned_surfaces)
            cleaned_surfaces.append((ref_file, ref_points))
            normal_subset: NDArray[np.float64] = normal_surfaces[ref_file]  # Extract the corresponding normals
            cleaned_normals.append(normal_subset)

            print(f"Reference surface {ref_file} added at index {ref_surface_indices[ref_file]}")

        # Process other surfaces to remove overlapping points
        for idx, (file, points) in enumerate(surfaces):

            if file not in ref_surface_indices:  # Skip if it's already a reference surface
                print(f"Processing {file} against reference surfaces")
                # check if  plane has no offset with respect to fault ignore splitting the plane
                # Extract z-values from the list of points
                z_values: NDArray[np.float64] = [point[2] for point in points]

                # Get maximum z
                max_z = max(z_values)
                min_z = min(z_values)

                # This part is for checking if the surface cutting fault has the same z (no layering difference on both sides of faukts)
                # then it dose not remove the points near the fault and dose not seprate the surface into two parts
                if abs(min_z -max_z) <= z_threshold:
                    print(f"All points have almost the same z for {file}, skipping removing.")
                    cleaned_surfaces.append((file, points))

                    continue
                else:
                    filtered: List[NDArray[np.float64]] = []
                    dist_po: List[NDArray[np.float64]] = []

                    # Check against each reference surface
                    for ref_file, ref_points in ref_surfaces:
                        print(f"Using {ref_file} as reference")
                        # Get normal vectors of ref_points
                        normal_vec_ref = normal_surfaces[ref_file]
                        normals_points=get_normals(ref_points, ref_points, normal_vec_ref)
                        # Create a k-d tree from the current surface points
                        points_tree = cKDTree(points)


                        # Find all current surface points that are within DISTANCE_THRESHOLD of any reference point
                        indices_list: List[List[int]] = points_tree.query_ball_point(ref_points, r=DISTANCE_THRESHOLD)

                        # Flatten list and get unique indices
                        indices: NDArray[np.int64] = np.unique(np.hstack(indices_list))

                        # Extract points within the distance
                        nearest_to_ref: NDArray[np.float64] = points[indices.astype(int)]
                        # Find the normal vectors of these near points to reference surface
                        normal_vec_point: NDArray[np.float64] = normal_surfaces[file]
                        normals_ref_points: NDArray[np.float64] = get_normals(nearest_to_ref, points, normal_vec_point)

                        # Build KDTree for ref_points (to find nearest ref_point for each nearest_to_ref point)
                        ref_tree = cKDTree(ref_points)

                        filtered_in: List[NDArray[np.float64]] = []
                        for i, pt in enumerate(nearest_to_ref):
                            # For each point near the reference surface, find its nearest reference point
                            dist_in, nearest_idx = ref_tree.query(pt)

                            # Get the normal of this nearest reference point
                            normal_ref = normals_points[nearest_idx]

                            # Get the normal of the nearest_to_ref point
                            normal_near = normals_ref_points[i]

                            # Compute the dot product of the two normals
                            dot_product = np.dot(normal_ref, normal_near)

                            # Check if the normals are parallel
                            if abs(dot_product) > 0.95:

                                filtered_in.append(pt)
                                dist_po.append(dist_in)


                        #max_dist = max(dist_po)
                        max_dist: float = DISTANCE_THRESHOLD
                        # Remove points in distance less than max_dist (to keep the filtered points in uniform distance)
                        for i, pt in enumerate(nearest_to_ref):
                            # Find the nearest ref_point for each nearest_to_ref point
                            dist_in, nearest_idx = ref_tree.query(pt)
                            if dist_in <= max_dist:
                                filtered_in.append(pt)

                        # Extract z-values from the list of points
                        z_values: List[np.float64] = [point[2] for point in filtered_in]

                        # Get maximum z
                        max_z:float = max(z_values)
                        min_z:float = min(z_values)

                        # This part is for checking if the surface cutting fault has the same z (no layering difference on both sides of faukts)
                        # then it dose not remove the points near the fault and dose not seprate the surface into two parts
                        if abs(min_z -max_z) <= z_threshold:
                            print(f"All points have almost the same z for {file}, skipping removing.")
                            continue
                        else:
                            filtered.append(filtered_in)


                    if len(filtered) > 0:
                        filtered_array = np.vstack(filtered)
                        filtered_array = np.unique(filtered_array, axis=0)

                    # Ensure both arrays are numpy arrays
                    points: NDArray[np.float64]  = np.array(points)
                    filtered_array: NDArray[np.float64]  = np.array(filtered_array)

                    ## Create a mask to keep only points NOT in filtered_array
                    mask: NDArray[np.bool_] = ~np.any(np.all(points[:, None, :] == filtered_array, axis=2), axis=1)

                    # Apply the mask to get the filtered points
                    filtered_points: NDArray[np.float64] = points[mask]

                    if len(filtered_points) == 0:
                        print(f"All points removed for {file}, skipping clustering.")

                        continue
                    # Filter corresponding normals using the same mask
                    filtered_normals: NDArray[np.float64] = normal_surfaces[file][1][mask]

                    # Perform clustering to identify disconnected parts
                    # cluster must have at least 10 points (min_cluster_size=10) and
                    # Points that have fewer than 10 neighbors within the "core distance" may be labeled as noise (min_samples=10).
                    clustering = HDBSCAN(min_cluster_size=10, min_samples=10).fit(filtered_points)
                    labels: NDArray[np.int64] = clustering.labels_
                    unique_labels = np.unique(labels)

                    # Store separated clusters
                    for label in unique_labels:
                        if label != -1:  # Ignore noise (-1)
                            cluster_points = filtered_points[labels == label]
                            cluster_normals = filtered_normals[labels == label]  # Corresponding normals for the cluster

                            cluster_file = f"{file}_surface_{label}"  # Unique name for the cluster


                            cleaned_surfaces.append((cluster_file, cluster_points))

                        # Also store the corresponding normals in cleaned_normals
                            cleaned_normals.append((cluster_file, cluster_normals))

        print(f"Reference surfaces are located at indices: {ref_surface_indices}", len(cleaned_surfaces))
        # Optional
        #plot_cleaned_surfaces(cleaned_surfaces)


        # Find intersection points, add them to surfaces, and extrude them
        nearest_points_dict: Dict[Any, NDArray[np.floating]] = {}
        intersection_points: List[NDArray[np.floating]] = []
        for i, (file, points) in enumerate(cleaned_surfaces):

            # Skip surfaces that are reference surfaces
            if i not in ref_surface_indices.values():
                all_extruded_points_for_current_surface: List[NDArray[np.floating]] = []

                # Loop through each reference surface to find intersections
                for ref_file, ref_points in ref_surfaces:

                    # Build a KDTree for the reference surface points for fast nearest-neighbor queries
                    ref_tree = cKDTree(ref_points)
                    # Find nearest reference points to all points on the current surface within the threshold
                    distances_ref, indices_ref = ref_tree.query(points, distance_upper_bound=PROJECTION_THRESHOLD)

                    # Build a KDTree for the current surface points
                    points_tree = cKDTree(points)
                    # Find all current surface points that are within PROJECTION_THRESHOLD of any reference point
                    indices_list: List[List[int]] = points_tree.query_ball_point(ref_points, r=PROJECTION_THRESHOLD)
                    indices: NDArray[np.integer]  = np.unique(np.hstack(indices_list))
                    nearest_to_ref: NDArray[np.floating] = points[indices.astype(int)]  # Points on current surface near reference

                    # Skip if there are no points close to the reference surface
                    if nearest_to_ref.size == 0:
                        continue

                    # Mask to select reference points that are within the threshold
                    close_ref_points_mask: NDArray[np.bool_] = distances_ref <= PROJECTION_THRESHOLD
                    nearest_ref_points: NDArray[np.floating] = ref_points[indices_ref[close_ref_points_mask]]  # Nearby points on the reference surface

                    # Query again to get current surface points near the reference points
                    distances_point, indices_point = points_tree.query(ref_points, distance_upper_bound=PROJECTION_THRESHOLD)
                    close_points_mask: NDArray[np.bool_]  = distances_point <= PROJECTION_THRESHOLD
                    nearest_or_points: NDArray[np.floating] = np.unique(points[indices_point[close_points_mask]], axis=0)  # Current surface points near ref

                    # Skip if either set is empty
                    if nearest_or_points.size == 0 or nearest_ref_points.size == 0:
                        continue

                    # Extract farthest points per unique y-coordinate to ensure points are well distributed
                    unique_y: NDArray[np.floating] = np.unique(nearest_or_points[:, 1])
                    farthest_x_points: List[NDArray[np.floating]] = []

                    for y in unique_y:
                        # Subset of current surface points at this y-coordinate
                        subset: NDArray[np.floating] = nearest_or_points[nearest_or_points[:, 1] == y]
                        # Subset of reference points at this y-coordinate
                        fault_subset: NDArray[np.floating] = nearest_ref_points[nearest_ref_points[:, 1] == y]
                        if len(fault_subset) == 0:
                            continue  # Skip if no reference points at this y

                        # Use the center of current subset to find the closest reference point
                        center_point: NDArray[np.floating] = np.mean(subset, axis=0)
                        distances: NDArray[np.floating] = np.linalg.norm(fault_subset - center_point, axis=1)
                        closest_fault_point: NDArray[np.floating] = fault_subset[np.argmin(distances)]
                        fault_x: float = closest_fault_point[0]

                        # Find the point in the subset farthest in x-direction from the fault
                        x_values = subset[:, 0]
                        x_distances: NDArray[np.floating] = np.abs(x_values - fault_x)
                        farthest_idx:int = np.argmax(x_distances)
                        farthest_x_points.append(subset[farthest_idx])

                    # Final set of surface points used for extrusion
                    nearest_or_points = np.array(farthest_x_points)
                    # Store points associated with this surface for later reference
                    nearest_points_dict[file] = nearest_or_points

                    # Check if the surface is clustered: skip if too many points are very close
                    close_points_mask = distances_point <= (DISTANCE_THRESHOLD / 4)
                    nearest_thres_points: NDArray[np.floating] = np.unique(points[indices_point[close_points_mask]], axis=0)
                    if nearest_thres_points.size != 0:
                        continue

                    # Calculate normals for the selected points on the surface
                    surface_normals: NDArray[np.floating] = calculate_normals(nearest_or_points, cleaned_surfaces, cleaned_normals, file)
                    # Keep track of reference points for extrusion
                    intersection_points.extend(nearest_ref_points)

                    # Extrude points along a direction perpendicular to their normals
                    extruded_intersection_points: NDArray[np.floating] = correct_extrusion_direction(
                        nearest_or_points, surface_normals, file, intersection_points, nearest_points_dict, EXTRUSION_FACTOR
                    )
                    # Add extruded points for this reference surface to the list
                    all_extruded_points_for_current_surface.append(extruded_intersection_points)

                # Combine all extruded points with the current surface points
                if all_extruded_points_for_current_surface:
                    extruded_combined: NDArray[np.floating] = np.vstack(all_extruded_points_for_current_surface)
                    updated_points: NDArray[np.floating] = np.vstack([points, extruded_combined.reshape(-1, 3)])
                    # Update the surface with the new set of points
                    cleaned_surfaces[i] = (file, updated_points)



        # Optionally, visualize all surfaces including extruded points
        #plot_surfaces_with_extrusions(
        #    cleaned_surfaces=cleaned_surfaces,
        #    intersection_points=intersection_points,
        #    extruded_points=all_extruded_points_for_current_surface,  # or your full list of extruded points
        #    point_size=5
        #    )



    else:
      cleaned_surfaces =surfaces

    return cleaned_surfaces, ref_surface_indices, grid_litho
