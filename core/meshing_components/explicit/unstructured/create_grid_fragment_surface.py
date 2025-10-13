import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import colorcet as cc
import meshio
import gmsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from scipy.stats import zscore

from scipy.interpolate import Rbf
from sklearn.cluster import HDBSCAN


def create_surface_grid(cleaned_surfaces, buffer_dist = 0, smooth= 1e-5):
    """
    Sort the surface vertices into a grid and interpolate the z values
    based on the sorted grid.

    Args:
        cleaned_surfaces (list): List of tuples where the second element contains surface points.

    Returns:
        interpolated_surfaces: A list of arrays containing interpolated grids for each surface.
    """

    interpolated_surfaces = []
    max_n_gx=250
    max_n_gy=100
    for id, points in cleaned_surfaces:  # Extract points directly
        # Create DataFrame and remove duplicates
        df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}).drop_duplicates()
        x_cleaned = df['x'].values
        y_cleaned = df['y'].values
        z_cleaned = df['z'].values

        #x_min, x_max = np.min(x_cleaned), np.max(x_cleaned)
        #y_min, y_max = np.min(y_cleaned), np.max(y_cleaned)
        # Extend grid bounds by 20 units in all directions


        x_min, x_max = np.min(x_cleaned) - buffer_dist, np.max(x_cleaned) + buffer_dist
        y_min, y_max = np.min(y_cleaned) - buffer_dist, np.max(y_cleaned) + buffer_dist


        # Ensure grid size constraints
        unique_x = np.unique(x_cleaned)
        unique_y = np.unique(y_cleaned)
        # Number of grid points in x and y directions
        n_gx = len(unique_x)
        n_gy = len(unique_y)
        if n_gx > max_n_gx:
          n_gx = max_n_gx
        if n_gy > max_n_gy:
          n_gy = max_n_gy
        print(n_gx,n_gy,'n_gx, n_gy')

        # Create grid
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, n_gx), np.linspace(y_min, y_max, n_gy))

        # Interpolation using RBF
        rbf = Rbf(x_cleaned, y_cleaned, z_cleaned, function='multiquadric', epsilon=2, smooth=smooth)

        z_interpolated = (rbf(grid_x, grid_y))

        # Combine into final interpolated surface
        interpolated_grid = np.column_stack((grid_x.flatten(), grid_y.flatten(), z_interpolated.flatten()))
        interpolated_surfaces.append(interpolated_grid)

    print('Interpolation is done!')
    return interpolated_surfaces


def plot_surfaces_individually(interpolated_surfaces):
    for i, surface in enumerate(interpolated_surfaces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(surface[:, 0], surface[:, 1], surface[:, 2],
                        cmap='plasma', edgecolor='none', alpha=0.9)

        ax.set_title(f"Interpolated Surface {i+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show()


def import_surfaces(interpolated_s, extent=None, tolerance=50):
    """
    Gets a list of interpolated surface grids and creates B-spline surfaces using GMSH.

    Args:
        interpolated_s (list of np.ndarray): List of surface points (x, y, z) for each surface.
        extent (tuple): Optional bounding box (x_b_min, x_b_max, y_b_min, y_b_max, z_b_min, z_b_max).
        tolerance (float): Acceptable deviation from the extent values.

    Returns:
        surfaces (list): List of GMSH B-spline surface IDs.
        bounds (tuple): Adjusted bounding box based on overlap and extent constraints.
    """

    surfaces = []

    min_x_list, max_x_list = [], []
    min_y_list, max_y_list = [], []
    min_z_list, max_z_list = [], []

    for surface_points in interpolated_s:
        if not isinstance(surface_points, np.ndarray) or surface_points.shape[1] != 3:
            print("Invalid surface points format")
            continue

        min_x_list.append(np.min(surface_points[:, 0]))
        max_x_list.append(np.max(surface_points[:, 0]))
        min_y_list.append(np.min(surface_points[:, 1]))
        max_y_list.append(np.max(surface_points[:, 1]))
        min_z_list.append(np.min(surface_points[:, 2]))
        max_z_list.append(np.max(surface_points[:, 2]))
        x = surface_points[:, 0]
        y = surface_points[:, 1]

        unique_x = np.unique(x)
        unique_y = np.unique(y)

        numPointsU = len(unique_x)
        numPointsV = len(unique_y)

        ps = []
        for i in range(numPointsU):
            for j in range(numPointsV):
                index = i * numPointsV + j
                if index < len(surface_points):
                    point = surface_points[index]
                    ps.append(gmsh.model.occ.addPoint(point[0], point[1], point[2]))

        if len(ps) != numPointsU * numPointsV:
            print(f"Warning: Skipping B-spline surface due to mismatch in control points: {len(ps)} != {numPointsU * numPointsV}")
            continue

        s = gmsh.model.occ.addBSplineSurface(ps, numPointsU=numPointsU)
        surfaces.append(s)

    def find_adjusted_bound(bound_list, target_value, mode='max'):
        """
        Return the closest bound within tolerance, trying the next best if the closest is out of range.
        """
        sorted_list = sorted(bound_list, reverse=(mode == 'max'))

        for val in sorted_list:
            if abs(val - target_value) <= tolerance:
                if (mode == 'min' and val > target_value) or (mode == 'max' and val < target_value):
                    return target_value
                else:
                    return val

        print(f"Warning: No bounds found within ±{tolerance} of {target_value}")
        return target_value

    bounds = None
    extent = tuple(extent)

    if surfaces and extent:
        print(extent)
        x_b_min, x_b_max, y_b_min, y_b_max, z_b_min, z_b_max = extent

        bounds = (
            find_adjusted_bound(min_x_list, x_b_min, mode='max'),
            find_adjusted_bound(max_x_list, x_b_max, mode='min'),
            find_adjusted_bound(min_y_list, y_b_min, mode='max'),
            find_adjusted_bound(max_y_list, y_b_max, mode='min'),
            find_adjusted_bound(min_z_list, z_b_min, mode='max'),
            find_adjusted_bound(max_z_list, z_b_max, mode='min')
        )

    print('B-spline surfaces have been imported!')
    return surfaces, bounds



def fragment_surfaces(surfaces, extent, ref_surface_indices, wells, extra_planes, source_points,mine_shafts,  mesh_size=20, curve_mesh_size=5):
    """
    The function uses GMSH for creation of a bounding box, fragmentation of
    surfaces, and generation of physical groups based on the fragmented surfaces. It filters out surfaces that fall
    outside of the adjusted bounding box defined by `extent`.

    Args:
        surfaces (list): A list of B-spline surfaces or surface tags (e.g., GMSH handles or surface IDs) to be fragmented.
        extent (tuple): A 6-tuple defining the bounding box of the model domain in the form (x_min, x_max, y_min, y_max, z_min, z_max).
        ref_surface_indices (dict): Dictionary indicating reference surfaces (usually fault surfaces) and their indices in the `surfaces` list, used to guide fragmentation logic.
        wells (list of tuples): List of well geometries. Each tuple should contain the coordinates of the well bore (x1, y1, z1, x2, y2, z2).
        extra_planes (list of tuples): List of planes defined by 4 corner points. Each plane is a tuple of 12 floats (x1, y1, z1, ..., x4, y4, z4).
        source_points (list of tuples): List of source points, each defined by (x, y, z), to be included in the model geometry.
        mine_shafts (list of dicts): List of mine shafts where each shaft is represented as a dictionary with keys: `'center'`, `'axis'`, and `'radius'`.
        mesh_size (int, optional): Target mesh size. Default is 20.
        curve_mesh_size (int, optional): Finer mesh size to be applied to curves/edges for better resolution. Default is 5.

    Returns:
        ov (list): List of original GMSH surfaces before fragmentation.
        ovv (list): List of resulting fragmented surfaces or volumes after Boolean operations.
        tagsss (list): List of GMSH physical group tags for the main surfaces (faults, layers, wells, etc.).
        well_tags (list): List of physical group tags associated specifically with wells.
        shaft_tags (list): List of physical group tags for mine shafts.
        shaft_to_child_fragments (dict): Mapping from shaft tags to the IDs of intersecting or resulting child fragments (used to track mesh regions influenced by mine shafts).
        source_tag (int): GMSH physical group tag assigned to the source points.
    """
    outside_threshold = 0.1  # Define the threshold for coordinates of points outside the model domain

    # Create a box for fragmenting
    x_min, x_max, y_min, y_max, z_min, z_max = extent
    v = gmsh.model.occ.addBox(x_min , y_min , z_min, x_max - x_min, y_max - y_min, z_max - z_min)
    def clamp_to_nearest_boundary(val, min_val, max_val):
      if val < min_val:
        # outside below min, snap to min
        return min_val
      elif val > max_val:
        # outside above max, snap to max
        return max_val
      else:
        # inside, keep original
       return val


    if source_points:
        source=[]
        for i in range(len(source_points)):
            x, y, z = source_points[i][0], source_points[i][1], source_points[i][2]
            p_s = gmsh.model.occ.addPoint(x,y,z)
            source.append(p_s)
        gmsh.model.occ.synchronize()

    if wells:
        well_lines = []  # This will be a list of lists: one list per well
        for coords in wells:
            points = []
            for j in range(0, len(coords), 3):
                x, y, z = coords[j], coords[j+1], coords[j+2]
                x = clamp_to_nearest_boundary(x, x_min, x_max)
                y = clamp_to_nearest_boundary(y, y_min, y_max)
                z = clamp_to_nearest_boundary(z, z_min, z_max)
                p = gmsh.model.occ.addPoint(x, y, z)
                points.append(p)

            lines = []  # Lines for the current well
            for k in range(len(points) - 1):
                line = gmsh.model.occ.addLine(points[k], points[k+1])
                lines.append(line)

            well_lines.append(lines)  # Append the current well's line list

        gmsh.model.occ.synchronize()

    if extra_planes:
        for i_layer in range(len(extra_planes)):
          (x1_m, y1_m, z1_m, x2_m, y2_m, z2_m, x3_m, y3_m, z3_m, x4_m, y4_m, z4_m) = extra_planes[i_layer]

          # Add points (you don’t need tags in occ)
          p1 = gmsh.model.occ.addPoint(x1_m, y1_m, z1_m)
          p2 = gmsh.model.occ.addPoint(x2_m, y2_m, z2_m)
          p3 = gmsh.model.occ.addPoint(x3_m, y3_m, z3_m)
          p4 = gmsh.model.occ.addPoint(x4_m, y4_m, z4_m)

          # Create lines
          l1 = gmsh.model.occ.addLine(p1, p2)
          l2 = gmsh.model.occ.addLine(p2, p3)
          l3 = gmsh.model.occ.addLine(p3, p4)
          l4 = gmsh.model.occ.addLine(p4, p1)

          # Create line loop and surface
          loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
          surface_mine = gmsh.model.occ.addPlaneSurface([loop])

          surfaces.append(surface_mine)
          ref_surface_indices[len(surfaces) - 1] = len(surfaces) - 1
          gmsh.model.occ.synchronize()

    mine_shaft_volumes = []  # NEW: store shaft volume tags
    if mine_shafts:
        mine_shaft_volumes = []  # Ensure this list is defined
        for i, shaft in enumerate(mine_shafts):
            x, y, z = shaft["center"]
            x = clamp_to_nearest_boundary(x, x_min, x_max)
            y = clamp_to_nearest_boundary(y, y_min, y_max)
            z = clamp_to_nearest_boundary(z, z_min, z_max)
            dx, dy, dz = shaft["axis"]
            x2 = x + dx
            y2 = y + dy
            z2 = z + dz
            x2 = clamp_to_nearest_boundary(x2, x_min, x_max)
            y2 = clamp_to_nearest_boundary(y2, y_min, y_max)
            z2 = clamp_to_nearest_boundary(z2, z_min, z_max)
            dx = x2 - x
            dy = y2 - y
            dz = z2 - z

            # Replace the original shaft center and axis with clamped values
            shaft["center"] = (x, y, z)
            shaft["axis"] = (dx, dy, dz)
            r = shaft["radius"]
            tag = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r, 2500 + i + 1)
            mine_shaft_volumes.append(tag)
        gmsh.model.occ.synchronize()

    print("Number of surfaces:", len(surfaces))
    print("Surface tags:", surfaces)

    tool_entities = [(2, s) for s in surfaces]  # start with surfaces

    if wells:
        all_well_lines = [l for sublist in well_lines for l in sublist]
        tool_entities += [(1, l) for l in all_well_lines]

    if source_points:
        tool_entities += [(0, p) for p in source]


    if mine_shafts:
        tool_entities += [(3, tag) for tag in mine_shaft_volumes]
    #gmsh.write("model.brep")  # Saves full geometry
    # or
    #gmsh.write("model.geo_unrolled")  # For readable Gmsh .geo

    ov, ovv = gmsh.model.occ.fragment(
        [(3, v)] + tool_entities, [],
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()
    #gmsh.write("model1.brep")  # Saves full geometry

    # Filter to get only 3D volumes from ov
    fragmented_volumes = [entity for entity in ov if entity[0] == 3]
    print(f"Number of volumes created: {len(fragmented_volumes)}")
    print("Volume tags:", [tag for dim, tag in fragmented_volumes])
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-4)

    gmsh.model.occ.synchronize()
    #gmsh.write("fragmented_model.brep")  # Saves full geometry
    # or
    #gmsh.write("fragmented_model.geo_unrolled")  # For readable Gmsh .geo

    tagsss = []  # List to store physical groups

    # Handle the case when there are ref_surface_indices
    if ref_surface_indices:
        # Create 2D meshes
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)
        gmsh.model.mesh.generate(2)
        mesh_file = "mesh.msh"
        gmsh.write(mesh_file)
        gmsh.model.occ.synchronize()
        ######## gmsh.fltk.initialize()
        ######## while gmsh.fltk.isAvailable():
        ########   gmsh.fltk.wait()
        zip_inputs = [v] + surfaces  # always include v and surfaces

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
                        if len(nodeTags) == 0:
                            continue

                        # Convert flattened list to (x, y, z) tuples
                        node_coords = [(coords[ii], coords[ii + 1], coords[ii + 2]) for ii in range(0, len(coords), 3)]

                        # Check if any node is outside the adjusted bounding box
                        for x, y, z in node_coords:
                            if (x < x_min - outside_threshold or x > x_max + outside_threshold or
                                y < y_min - outside_threshold or y > y_max + outside_threshold or
                                z < z_min - outside_threshold or z > z_max + outside_threshold):
                                filtered_surfaces.append(surface)
                                break
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

                    # Add physical group for each fault surface
                    tagsss.append(gmsh.model.addPhysicalGroup(2, values, 1000 * i))


    ###gmsh.fltk.initialize()
    ###while gmsh.fltk.isAvailable():
    ###  gmsh.fltk.wait()

    if wells and not mine_shafts:
      if source_points:
        tool_entities_total = surfaces+ all_well_lines + [(0, p) for p in source]
      else:
        tool_entities_total = surfaces+ all_well_lines
      # Extract new surfaces and lines from `ovv`
      for e in zip([v] + tool_entities_total, ovv):
            print("parent " + str(e[0]) + " -> child " + str(e[1]))


      line_fragment_dict = {}

      for parent, children in zip([v] + tool_entities_total, ovv):
            # Check if the parent is a line (dimension 1)
            if isinstance(parent, int) or (isinstance(parent, tuple) and parent[0] == 1):
                # Ensure children is a list of tuples and filter only 1D entities
                line_children = [child[1] for child in children if child[0] == 1]
                parent_id = parent if isinstance(parent, int) else parent[1]
                if line_children:
                    line_fragment_dict[parent_id] = line_children



      flattened_ovv = [item for sublist in ovv for item in sublist]

      new_surfaces = [e[1] for e in flattened_ovv if e[0] == 2]
      new_lines = [e[1] for e in flattened_ovv if e[0] == 1]
      gmsh.model.occ.synchronize()


      # Fragment surfaces with the well lines
      ov_l, ovv_l = gmsh.model.occ.fragment(
            [(1, l) for l in new_lines],
            [(2, s) for s in new_surfaces]
        )
      gmsh.model.occ.synchronize()


      # Create mapping from input to fragmented outputs
      zipped_l = list(zip([new_surfaces] + new_lines, ovv_l))  # Convert zip object to a list

      print("Mapping of well lines before/after fragment:")

      for el in zip([new_surfaces] + new_lines, ovv_l):
            print("parent " + str(el[0]) + " -> child " + str(el[1]))

      # Collect well tags for new fragmented lines

      well_tags = []
      n = 1

      # well_lines = [[29, 30], [31]]
      for well in well_lines:
            all_child_lines = []

            for line_id in well:
                fragments = line_fragment_dict.get(line_id, [])
                all_child_lines.extend(fragments)

            if all_child_lines:
                tag = gmsh.model.addPhysicalGroup(1, all_child_lines, 1000 + n)
                print(f"Created physical group {tag} for well with original lines {well} and child lines {all_child_lines}")
                well_tags.append(tag)
                n += 1


    elif wells and mine_shafts:
      if source_points:
        tool_entities_total = surfaces+ all_well_lines + [(0, p) for p in source] + mine_shaft_volumes
      else:
        tool_entities_total = surfaces+ all_well_lines + mine_shaft_volumes
      # Extract new surfaces and lines from `ovv`
      for e in zip([v] + tool_entities_total, ovv):
            print("parent " + str(e[0]) + " -> child " + str(e[1]))


      line_fragment_dict = {}

      for parent, children in zip([v] + tool_entities_total, ovv):
            # Check if the parent is a line (dimension 1)
            if isinstance(parent, int) or (isinstance(parent, tuple) and parent[0] == 1):
                # Ensure children is a list of tuples and filter only 1D entities
                line_children = [child[1] for child in children if child[0] == 1]
                parent_id = parent if isinstance(parent, int) else parent[1]
                if line_children:
                    line_fragment_dict[parent_id] = line_children



      flattened_ovv = [item for sublist in ovv for item in sublist]

      new_surfaces = [e[1] for e in flattened_ovv if e[0] == 2]
      new_lines = [e[1] for e in flattened_ovv if e[0] == 1]
      gmsh.model.occ.synchronize()


      # Fragment surfaces with the well lines
      ov_l, ovv_l = gmsh.model.occ.fragment(
            [(1, l) for l in new_lines],
            [(2, s) for s in new_surfaces]
        )
      gmsh.model.occ.synchronize()


      # Create mapping from input to fragmented outputs
      zipped_l = list(zip([new_surfaces] + new_lines, ovv_l))  # Convert zip object to a list

      print("Mapping of well lines before/after fragment:")

      for el in zip([new_surfaces] + new_lines, ovv_l):
            print("parent " + str(el[0]) + " -> child " + str(el[1]))

      # Collect well tags for new fragmented lines

      well_tags = []
      n = 1

      # well_lines = [[29, 30], [31]]
      for well in well_lines:
            all_child_lines = []

            for line_id in well:
                fragments = line_fragment_dict.get(line_id, [])
                all_child_lines.extend(fragments)

            if all_child_lines:
                tag = gmsh.model.addPhysicalGroup(1, all_child_lines, 1000 + n)
                print(f"Created physical group {tag} for well with original lines {well} and child lines {all_child_lines}")
                well_tags.append(tag)
                n += 1


      shaft_fragment_dict = {}
      if source_points:
        tool_entities_total = surfaces+ all_well_lines + [(0, p) for p in source] + mine_shaft_volumes
      else:
        tool_entities_total = surfaces+ all_well_lines + mine_shaft_volumes

      all_parents = [v] + tool_entities_total

      for parent, children in zip(all_parents, ovv):
            if parent in mine_shaft_volumes:
                # Normalize parent ID (in case it's a tuple)
                parent_id = parent if isinstance(parent, int) else parent[1]

                # Filter 3D (volume) children
                volume_children = [child[1] for child in children if child[0] == 3]
                if volume_children:
                    shaft_fragment_dict[parent_id] = volume_children





      shaft_tags = []
      n = 1
      shaft_to_child_fragments = {}


      for shaft in mine_shaft_volumes:
            all_child_shafts = []


            # Normalize to a list: ensures we always can iterate
            shaft_ids = shaft if isinstance(shaft, list) else [shaft]

            for shaft_id in shaft_ids:

                fragments_sh = shaft_fragment_dict.get(shaft_id, [])
                all_child_shafts.extend(fragments_sh)



            if all_child_shafts :
                shaft_to_child_fragments[shaft_id] = all_child_shafts.copy()

                tag_sh = gmsh.model.addPhysicalGroup(3, all_child_shafts , 3500 + n)
                print(f"Created physical group {tag} for well with original shaft {shaft_id} and child lines {all_child_shafts}")
                shaft_tags.append(tag_sh)

                n += 1


                # Remove shaft fragments from ov
                ov = [entry for entry in ov if not (entry[0] == 3 and entry[1] in all_child_shafts)]


      gmsh.model.occ.synchronize()



    elif mine_shafts and not wells:
      if source_points:
        tool_entities_total = surfaces + [(0, p) for p in source] + mine_shaft_volumes
      else:
        tool_entities_total = surfaces + mine_shaft_volumes

      shaft_fragment_dict = {}

      all_parents = [v] + tool_entities_total

      for parent, children in zip(all_parents, ovv):
            if parent in mine_shaft_volumes:
                # Normalize parent ID (in case it's a tuple)
                parent_id = parent if isinstance(parent, int) else parent[1]

                # Filter 3D (volume) children
                volume_children = [child[1] for child in children if child[0] == 3]
                if volume_children:
                    shaft_fragment_dict[parent_id] = volume_children




      shaft_tags = []
      n = 1


      shaft_to_child_fragments = {}

      for shaft in mine_shaft_volumes:
            all_child_shafts = []


            shaft_ids = shaft if isinstance(shaft, list) else [shaft]

            for shaft_id in shaft_ids:

                fragments_sh = shaft_fragment_dict.get(shaft_id, [])
                all_child_shafts.extend(fragments_sh)


            if all_child_shafts:
                shaft_to_child_fragments[shaft_id] = all_child_shafts.copy()


                tag_sh = gmsh.model.addPhysicalGroup(3, all_child_shafts , 3500 + n)
                print(f"Created physical group {tag_sh} for well with original shaft {shaft_id} and child lines {all_child_shafts}")
                shaft_tags.append(tag_sh)
                n += 1


                # Remove shaft fragments from ov
                ov = [entry for entry in ov if not (entry[0] == 3 and entry[1] in all_child_shafts)]

    if not mine_shafts:
        shaft_to_child_fragments ={}

    if source_points:
          source_tag=[]
          for i in range(len(source)):
            tag_source= gmsh.model.addPhysicalGroup(0, [source[i]], 2000+i+1)
            source_tag.append(tag_source)
    else:
        source_tag=[]


    if not wells:
      well_tags=[]
    if not mine_shafts:
      shaft_tags = []

    # Return updated entities
    return ov, ovv, tagsss, well_tags, shaft_tags, shaft_to_child_fragments, source_tag

