import pandas as pd
import glob
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import colorcet as cc
from core.utility import surface_mesh_gradients
from sklearn.cluster import HDBSCAN
from scipy.spatial import cKDTree, KDTree


def get_normals(near_points, points, normal_vec):
    """
    Finds the corresponding normals for the given points in a surface.

    Parameters:
        near_points (np.ndarray): Subset of points from a surface.
        points (np.ndarray): points of the whole surface.
        normal_vec (np.ndarray): normal vectors of the points of the whole surface

    Returns:
        np.ndarray: Array of normals corresponding to the given points.
    """
    # Find indices of near_points in points
    indices = np.where((points[:, None] == near_points).all(axis=2))[0]
    # Get corresponding normal vectors
    return normal_vec[1][indices]


def calculate_normals(points, cleaned_surfaces, cleaned_normals, file_index):
    """
    Finds the corresponding normals for the given points in a surface.

    Parameters:
        points (np.ndarray): Subset of points from a surface.
        cleaned_surfaces (list): List of surfaces, each with a file_index (string) and associated points.
        cleaned_normals (list): List of normals corresponding to each surface.
        file_index (str or int): Index of the surface (may be string if from a filename).

    Returns:
        norm (np.ndarray): Array of normals corresponding to the given points.
    """
    # Find the desired surface and its normal vectors by searching for its lable
    for label, surface in cleaned_surfaces:
        if label== file_index:
            surface_points=surface
    for label, normal in cleaned_normals:
        if label== file_index:
            normal_points=normal


    # Find the indices of points in surface_points
    indices = []
    for p in points:
        # Check if point p is in surface_points, and get its index
        match_index = np.where(np.all(surface_points == p, axis=1))[0]

        if match_index.size > 0:
            indices.append(match_index[0])

    # Save normal vectors to norm_array
    norm_array=[]
    for j in indices:
        norm_array.append(normal_points[j])

    norm = np.vstack(norm_array)
    return norm


def correct_extrusion_direction(points, normals, file, intersection_points, nearest_points_dict, EXTRUSION_FACTOR=100):
    """
    Correct the extrusion direction by extruding points perpendicular to the normal vector in the x-z plane.

    Parameters:
        points: (n, 3) array of 3D points to adjust the extrusion direction.
        normals: (n, 3) array of normal vectors for each point.
        file: file identifier for the nearest points.
        intersection_points: points where intersections occur.
        nearest_points_dict: dictionary of nearest points for each file.
        EXTRUSION_FACTOR: Factor to scale the extrusion length.

    Returns:
        extruded_points: Corrected points with updated extrusion direction.
    """

    extruded_points = []
    for i, point in enumerate(points):
        # Extract the normal for the current point
        normal = normals[i]

        # Compute a perpendicular vector in the x-z plane
        perpendicular_vector = np.array([-normal[2], 0, normal[0]])

        # Normalize to ensure consistent scaling
        perpendicular_vector /= np.linalg.norm(perpendicular_vector)

        # Extrude the point in the perpendicular direction
        extruded_point = point.copy()
        extruded_point += EXTRUSION_FACTOR * perpendicular_vector  # Move in the perpendicular direction

        # Find the nearest intersection point (d2)
        intersection_tree = cKDTree(intersection_points)
        d2, idx1 = intersection_tree.query(extruded_point)
        nearest_intersection_point = intersection_points[idx1]

        # Find the nearest point from the dictionary (d3)
        near_point_tree = cKDTree(nearest_points_dict[file])
        d3, idx2 = near_point_tree.query(extruded_point)
        nearest_point = nearest_points_dict[file][idx2]

        # Check if extrusion is valid, otherwise reverse direction
        if d3 < d2:
            extruded_point = point.copy()
            extruded_point -= EXTRUSION_FACTOR * perpendicular_vector  # Reverse direction

        # Store the extruded point
        extruded_points.append(extruded_point)

    return np.array(extruded_points)



def plot_cleaned_surfaces_with_normals(cleaned_surfaces, cleaned_normals):
    """
    Plots points of cleaned_surfaces and their normal vectors
    Parameters:
        cleaned_surfaces : List of tuples (surface ID, cleaned points)
        cleaned_normals : List of tuples (surface ID, corresponding normal vectors)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through cleaned surfaces and normals
    for (surface_file, points), (surf_file, normals) in zip(cleaned_surfaces, cleaned_normals):

        # Extract x, y, z from points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Extract normal components
        u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

        # Plot the points
        ax.scatter(x, y, z, label=f'{surface_file}', s=1)

        # Plot normal vectors
        ax.quiver(x, y, z, u, v, w, length=0.5, color='black', normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cleaned Surfaces with Normals')
    ax.legend()
    plt.show()



def plot_surfaces_excluding_ref(surfaces, result):

    ref_surfaces = [(i, surfaces[i][1]) for i, is_fault in enumerate(result) if is_fault]
    non_ref_surfaces = [(i, surfaces[i][1]) for i, is_fault in enumerate(result) if not is_fault]
    print((non_ref_surfaces))
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



def plot_surface_with_normals(points, normals, title="Surface Normals"):
    """
    Plot a 3D surface using points and overlay the normal vectors.

    Parameters:
        points (np.ndarray): Nx3 array of 3D coordinates.
        normals (np.ndarray): Nx3 array of normal vectors corresponding to the points.
        title (str): Plot title.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Unpack point coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Unpack normal vector components
    u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

    # Plot the surface points
    ax.scatter(x, y, z, c='b', marker='o', s=5, alpha=0.3, label='Surface Points')

    # Plot normals as arrows
    ax.quiver(x, y, z, u, v, w, length=2.0, normalize=True, color='r', linewidth=0.5, label='Normals')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.tight_layout()
    plt.show()


def data_prepration(data_test, geomodel_result, DISTANCE_THRESHOLD = 50, PROJECTION_THRESHOLD = 60, EXTRUSION_FACTOR = 100, z_threshold = 10, num_wells=0, wells=[], num_sources=0, sources=[], num_shafts=0, centers=[], axes=[], radii=[], num_planes=0,extra_planes=[]):

    """
    Prepares geological surface input_data by cleaning overlapping points, identifying faults,
    clustering remaining surfaces, and computing extrusion based on proximity to reference (fault) surfaces.

    Args:
        data_test (InputData): Object containing model metadata, surface points, orientations, fault flags,
                               and the extent/resolution of the model.
        geomodel_result (list of np.ndarray): List of interpolated geological surfaces,
                                              each with shape (N, 3) representing (x, y, z) points.
        DISTANCE_THRESHOLD (int, optional): Distance used to filter overlapping points between surfaces. Default is 50.
        PROJECTION_THRESHOLD (int, optional): Distance threshold for projecting surface points to reference surfaces. Default is 60.
        EXTRUSION_FACTOR (int, optional): Scale factor for extruding surface points along normals. Default is 100.
        z_threshold (int, optional): Vertical threshold to identify surfaces at similar levels across faults. Default is 10.
        num_wells (int, optional): Number of wells in the model.
        wells (list of tuples, optional): List of well coordinates as tuples of at least 6 floats (x1,y1,z1,x2,y2,z2) per well.
        num_sources (int, optional): Number of point sources in the model.
        sources (list of tuples, optional): List of (x, y, z) coordinates for point sources.
        num_shafts (int, optional): Number of mine shafts in the model.
        centers (list of tuples, optional): List of center points (x, y, z) for cylindrical shafts.
        axes (list of tuples, optional): List of axis direction vectors (x, y, z) for each shaft.
        radii (list of float, optional): List of radii for the cylindrical shafts.
        num_planes (int, optional): Number of additional planes.
        extra_planes (list of tuples, optional): List of 4-corner planes defined as 12-tuple (x1,y1,z1,...,x4,y4,z4).
    Returns:
        cleaned_surfaces (list of tuples): Each tuple is (surface ID, cleaned surface points),representing the processed and de-overlapped surfaces.
        ref_surface_indices (list): List of surface indices (or IDs) identifying the reference fault surfaces within the `cleaned_surfaces` list.
        grid_litho (np.ndarray): Structured lithology grid that represents the geological model domain — used for assigning physical groups in simulations or meshing.
        wells (list of tuples): Forwarded list of well coordinates (x1,y1,z1,x2,y2,z2) per well.
        extra_planes (list of tuples): Forwarded list of additional planes defined by their corner coordinates.
        mine_shafts (list of dicts): List of dictionaries, each representing a shaft with center, axis, and radius.
        sources (list of tuples): Forwarded list of (x, y, z) coordinates of source points.
    """


    mine_shafts = []
    if num_shafts !=0:
      for i in range(len(centers)):
        mine_shafts.append({
            "center": centers[i],
            "axis": axes[i],
            "radius": radii[i]  # each is a tuple like (600, 500, 400)
        })
      # Print confirmation
      for i, shaft in enumerate(mine_shafts, 1):
        print(f"Mine shaft {i}:")
        print(f"  center = {shaft['center']}")
        print(f"  axis   = {shaft['axis']}")
        print(f"  radius  = {shaft['radius']}")

    else:
      mine_shafts = []

    # Fault information
    faults= data_test.faults
    # Mapping rocks in different layers
    mapping=data_test.mapping_object
    # Ensure all values are tuples
    maping = {
            k: v if isinstance(v, tuple) else (v,)
            for k, v in mapping.items()
        }
    # Getting fault information for each layer and sublayers
    fault_dict = {}
    for fault_status, key in zip(faults, maping):
        for component in maping[key]:
            fault_dict[component] = fault_status

    result = list(fault_dict.values())
    print(result)
    # If there is any fault in the model: use interpolated points and their normals
    if any(faults):
        points_list, vectors_list = surface_mesh_gradients.get_surface_mesh_gradients(geomodel_result, mesh_type="unmasked")

        # Save the surfaces points of each surface stored in points_list
        surfaces = [(i, points) for i, points in enumerate(points_list)]
        # Save the normal vectors in vectors_list
        normal_surfaces = [(i, norms) for i, norms in enumerate(vectors_list)]

        # Import geological grid, containing coordinates of grids
        grid_file=geomodel_result.grid
        # Import lithological input_data related to each grid points
        lith_block_file=geomodel_result.lith_block
        # Merge grids and their lithological input_data

        grid_litho = pd.concat([pd.DataFrame(grid_file), pd.DataFrame(lith_block_file)], axis=1)


    # if there is no fault: use the surface vertices
    else:
        # get all surfaces
        surfaces = geomodel_result.surface_meshes_vertices[1]
        surfaces = [(i, surface) for i, surface in enumerate(surfaces)]

        # Import geological grid, containing coordinates of grids
        grid_file=geomodel_result.grid
        # Import lithological input_data related to each grid points
        lith_block_file=geomodel_result.lith_block
        # Convert to DataFrame
        grid_df = pd.DataFrame(grid_file, columns=["x", "y", "z"])
        lith_df = pd.DataFrame(lith_block_file, columns=["lithology"])

        # Merge them
        grid_litho = pd.concat([grid_df, lith_df], axis=1)

    # Save fault surfaces in ref_surfaces
    ref_surfaces = [surfaces[i] for i, is_fault in enumerate(result) if is_fault]


    # Dictionary to store reference surface - fault surfaces- in cleaned_surfaces
    ref_surface_indices = {}
    # Call the plotting function
    ### plot_surfaces_excluding_ref(surfaces, result)

    # Remove overlapped points and store the surfaces in cleaned_surfaces and their normals in cleaned_normals
    # This step is only done if there is any fault in the model
    cleaned_surfaces = []
    cleaned_normals=[]
    if ref_surfaces:
        for ref_file, ref_points in ref_surfaces:
            print(f"Using {ref_file} as reference")

            # Append reference surface and their normal vectors to cleaned_surfaces and cleaned_normals
            ref_surface_indices[ref_file] = len(cleaned_surfaces)
            cleaned_surfaces.append((ref_file, ref_points))
            normal_subset = normal_surfaces[ref_file]  # Extract the corresponding normals
            cleaned_normals.append(normal_subset)

            print(f"Reference surface {ref_file} added at index {ref_surface_indices[ref_file]}")

        # Process other surfaces to remove overlapping points
        for idx, (file, points) in enumerate(surfaces):

            if file not in ref_surface_indices:  # Skip if it's already a reference surface
                print(f"Processing {file} against reference surfaces")

                filtered=[]
                dist_po=[]
                # Check against each reference surface
                for ref_file, ref_points in ref_surfaces:
                    print(f"Using {ref_file} as reference")
                    # Get normal vectors of ref_points
                    normal_vec_ref = normal_surfaces[ref_file]
                    normals_points=get_normals(ref_points, ref_points, normal_vec_ref)
                    # Create a k-d tree from the current surface points
                    points_tree = cKDTree(points)


                    # Find all current surface points that are within DISTANCE_THRESHOLD of any reference point
                    indices_list = points_tree.query_ball_point(ref_points, r=DISTANCE_THRESHOLD)

                    # Flatten list and get unique indices
                    indices = np.unique(np.hstack(indices_list))

                    # Extract points within the distance
                    nearest_to_ref = points[indices.astype(int)]
                    # Find the normal vectors of these near points to reference surface
                    normal_vec_point = normal_surfaces[file]
                    normals_ref_points = get_normals(nearest_to_ref, points, normal_vec_point)

                    # Build KDTree for ref_points (to find nearest ref_point for each nearest_to_ref point)
                    ref_tree = cKDTree(ref_points)

                    filtered_in=[]
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


                    max_dist = max(dist_po)

                    # Remove points in distance less than max_dist (to keep the filtered points in uniform distance)
                    for i, pt in enumerate(nearest_to_ref):
                        # Find the nearest ref_point for each nearest_to_ref point
                        dist_in, nearest_idx = ref_tree.query(pt)
                        if dist_in <= max_dist:
                            filtered_in.append(pt)

                    # Extract z-values from the list of points
                    z_values = [point[2] for point in filtered_in]

                    # Get maximum z
                    max_z = max(z_values)
                    min_z = min(z_values)

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
                points = np.array(points)
                filtered_array = np.array(filtered_array)

                ## Create a mask to keep only points NOT in filtered_array
                mask = ~np.any(np.all(points[:, None, :] == filtered_array, axis=2), axis=1)

                # Apply the mask to get the filtered points
                filtered_points = points[mask]

                if len(filtered_points) == 0:
                    print(f"All points removed for {file}, skipping clustering.")
                    continue
                # Filter corresponding normals using the same mask
                filtered_normals = normal_surfaces[file][1][mask]

                # Perform clustering to identify disconnected parts
                # cluster must have at least 10 points (min_cluster_size=10) and
                # Points that have fewer than 10 neighbors within the "core distance" may be labeled as noise (min_samples=10).
                clustering = HDBSCAN(min_cluster_size=10, min_samples=10).fit(filtered_points)
                labels = clustering.labels_
                unique_labels = np.unique(labels)

                # Store separated clusters
                for label in unique_labels:
                  if label != -1:  # Ignore noise (-1)
                      cluster_points = filtered_points[labels == label]
                      cluster_normals = filtered_normals[labels == label]  # Corresponding normals for the cluster

                      cluster_file = f"{file}_surface_{label}"  # Unique name for the cluster

                      # Store in cleaned_surfaces
                      cleaned_surfaces.append((cluster_file, cluster_points))
                      # Also store the corresponding normals in cleaned_normals
                      cleaned_normals.append((cluster_file, cluster_normals))

        print(f"Reference surfaces are located at indices: {ref_surface_indices}", len(cleaned_surfaces))


        # Call the plotting function after processing cleaned_surfaces
        #### plot_cleaned_surfaces_with_normals(cleaned_surfaces, cleaned_normals)

        # Find intersection points, add them to surfaces, and extrude them
        nearest_points_dict = {}
        intersection_points=[]
        extruded_points_all_surfaces = []

        for i, (file, points) in enumerate(cleaned_surfaces):

          if i not in ref_surface_indices.values():
            all_extruded_points_for_current_surface = []

            # Check against each reference surface
            for ref_file, ref_points in ref_surfaces:
              # Build KDTree for ref_points
              ref_tree = cKDTree(ref_points)
              # For each point, find the nearest ref_point within PROJECTION_THRESHOLD
              distances_ref, indices_ref = ref_tree.query(points, distance_upper_bound=PROJECTION_THRESHOLD)

              # Create a k-d tree from the current surface points
              points_tree = cKDTree(points)
              # Find all current surface points that are within DISTANCE_THRESHOLD of any reference point
              indices_list = points_tree.query_ball_point(ref_points, r=DISTANCE_THRESHOLD)
              # Flatten list and get unique indices
              indices = np.unique(np.hstack(indices_list))

              # Extract points within the distance
              nearest_to_ref = points[indices.astype(int)]

              if nearest_to_ref.size > 0:

                close_ref_points_mask = distances_ref <= PROJECTION_THRESHOLD
                nearest_ref_points = ref_points[indices_ref[close_ref_points_mask]]  # Get closest points from Surface 0
                # Create a k-d tree from the current surface points
                near_point_tree = cKDTree(points)
                distances_point, indices_point = near_point_tree.query(ref_points, distance_upper_bound=PROJECTION_THRESHOLD)

                close_points_mask = distances_point <= PROJECTION_THRESHOLD
                nearest_or_points = np.unique(points[indices_point[close_points_mask]], axis=0)  # Get closest points from currect Surface

                nearest_points_dict[file] = nearest_or_points  # Store points associated with the surface file
               # Check if fault plane is within the surface (meaning the surface cutting the faukt has almost the same z and it is not clustered)
                x_points = [points_sur[0] for points_sur in points]
                x_ref = [points_reference[0] for points_reference in nearest_ref_points]

                x_mean_ref= np.mean(x_ref)
                x_max=max(x_points)
                x_min=min(x_points)
                x_range_min = min(x_min, x_max)
                x_range_max = max(x_min, x_max)
                if  nearest_or_points.size > 0 :
                  # If the surface is not clustered ignore it
                  if x_range_min <= x_mean_ref <= x_range_max:
                    continue
                  else:
                    # Calculate normals for the current surface
                    surface_normals = calculate_normals(nearest_or_points, cleaned_surfaces, cleaned_normals, file)
                    ###  plot_surface_with_normals(nearest_or_points, surface_normals, title=f"Normals for {file}")

                    # Store intersection points separately for extrusion
                    intersection_points.extend(nearest_ref_points)

                # Extrude the intersection points and check the direction
                if nearest_ref_points.size > 0:
                    # If the surface is not clustered ignore it
                    if x_range_min <= x_mean_ref <= x_range_max:
                      continue
                    else:
                      # Extrude intersection points along the direction of perpendicular to their normals, and check the direction
                      extruded_intersection_points = correct_extrusion_direction(nearest_or_points, surface_normals, file, intersection_points, nearest_points_dict, EXTRUSION_FACTOR)
                      # Add extruded points to the surface
                      extruded_points_all_surfaces.append(extruded_intersection_points)
                      # Add extruded points to the surface
                      all_extruded_points_for_current_surface.append(extruded_intersection_points)

            if all_extruded_points_for_current_surface:
              # Stack all extruded points
              extruded_combined = np.vstack(all_extruded_points_for_current_surface)
              updated_points = np.vstack([points, extruded_combined])
              cleaned_surfaces[i] = (file, updated_points)

        # Optionally, visualize all surfaces including extruded points
        ####plotter = pv.Plotter()
        ####colors = cc.glasbey[:len(cleaned_surfaces)]  # Get distinct colors

        ##### Plot the original surfaces
        ###for idx, (file, points) in enumerate(cleaned_surfaces):
        ###  point_cloud = pv.PolyData(points)
        ###  plotter.add_mesh(point_cloud, color=colors[idx], point_size=5, render_points_as_spheres=True, opacity=0.7)

        #### Plot the intersection points (black)
        ###if (len(intersection_points)) > 0:
        ###    intersection_points_all = np.vstack(intersection_points)
        ###    intersection_cloud = pv.PolyData(intersection_points_all)
        ###    plotter.add_mesh(intersection_cloud, color='black', point_size=5, render_points_as_spheres=True)

            # Plot the extruded points (red stars)
        ###    extruded_points_all_surfaces = np.vstack(extruded_points_all_surfaces)
        ###    extruded_cloud = pv.PolyData(extruded_points_all_surfaces)
        ###    plotter.add_mesh(extruded_cloud, color='darkgreen', point_size=10, render_points_as_spheres=True)

        #### Show the plot
        ###plotter.show()


    else:
      cleaned_surfaces =surfaces

    return cleaned_surfaces, ref_surface_indices, grid_litho, wells, extra_planes, mine_shafts, sources
