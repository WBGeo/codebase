import numpy as np
import gmsh
import os
import glob
import pyvista as pv
from scipy.interpolate import Rbf
import pandas as pd
from core.object_components import MeshResults


def create_surface_grid(geomodel_result):
    """
    Sort the surface vertices into a grid, and then interpolate the z values based on
    the sorted grid. Also ensures the extent values are included in the grid.

    Args:
        geomodel_result (GeomodelResults): Geological model results containing surfaces.
    Returns:
        list: A list of arrays containing interpolated grids for each surface.
    """

    # Get the extent from the results instance
    extent = geomodel_result.extent
    x_min, x_max, y_min, y_max, z_min, z_max = extent
    interpolated_surfaces = []

    # Perform interpolation for each surface
    for i, vertices in enumerate(geomodel_result.surface_meshes_vertices):
        # Get the x, y, z coordinates of the current surface
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        # Remove duplicates to avoid interpolation issues
        df = pd.DataFrame({'x': x, 'y': y, 'z': z}).drop_duplicates()
        x_cleaned = df['x'].values
        y_cleaned = df['y'].values
        z_cleaned = df['z'].values

        # Sort the vertices by x and y to create a structured grid
        sorted_indices = np.lexsort((x_cleaned, y_cleaned))
        x_sorted = x_cleaned[sorted_indices]
        y_sorted = y_cleaned[sorted_indices]

        unique_x = np.unique(x_sorted)
        unique_y = np.unique(y_sorted)

        # Number of grid points in x and y directions
        n_gx = len(unique_x)
        n_gy = len(unique_y)
        # Create grid
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, n_gx), np.linspace(y_min, y_max, n_gy))

        # Define the radial basis function interpolator
        rbf = Rbf(x_cleaned, y_cleaned, z_cleaned, function='multiquadric', epsilon=2, smooth=1e-5)

        # Perform interpolation on the grid
        z_interpolated = rbf(grid_x, grid_y)
        z_interpolated = np.round(z_interpolated)

        # Combine the grid with interpolated z values
        interpolated_grid = np.column_stack((grid_x.flatten(), grid_y.flatten(), z_interpolated.flatten()))

        # Store the interpolated grid in the list
        interpolated_surfaces.append(interpolated_grid)
    print('Interpolation is done!')
    return interpolated_surfaces

def import_surfaces(interpolated_s):
  """
    Gets a list of interpolated surface grids and creates B-spline surfaces using GMSH.

    This function initializes GMSH, processes the provided surface grids, defines control points,
    and generates smooth B-spline surfaces.

    Args:
        interpolated_s (list of np.ndarray):
            A list where each element is a NumPy array containing (x, y, z) coordinates of an interpolated surface.

    Returns:
        list: A list of GMSH surface IDs representing the created B-spline surfaces.
  """
  # List to store the B-spline surfaces
  surfaces = []
  # Initialize GMSH
  gmsh.initialize()

  # Loop through each surface file and create the corresponding B-spline surface
  for surface_points in interpolated_s:
      # Ensure surface_points is in the correct format (a 2D grid of points)
      if not isinstance(surface_points, np.ndarray) or surface_points.shape[1] != 3:
          print("Invalid surface points format")
          continue

      # Extract x, y, and z columns
      x = surface_points[:, 0]
      y = surface_points[:, 1]
      z = surface_points[:, 2]


      # Find unique values for x and y
      unique_x = np.unique(x)
      unique_y = np.unique(y)

      # Count the number of unique values
      numPointsU = len(unique_x)
      numPointsV = len(unique_y)

      # Create the control points list (ps) for this surface
      ps = surface_points.tolist()

      # Create the control points list (ps) for this surface
      ps = []

      # Create the control points for the current surface
      for i in range(numPointsU):
          for j in range(numPointsV):
              index = i * numPointsV + j
              if index < len(surface_points):
                  point = surface_points[index]
                  ps.append(gmsh.model.occ.addPoint(point[0], point[1], point[2]))

      # Check if the number of control points matches the expected size
      assert len(ps) == numPointsU * numPointsV, f"Number of control points for {surface_file} doesn't match the expected size."

      # Create the B-spline surface for the current surface
      s = gmsh.model.occ.addBSplineSurface(ps, numPointsU=numPointsU)
      surfaces.append(s)
  print('B-spline surfaces have been imported!')
  return surfaces


def fragment_surfaces(surfaces, extent):
    """
    Fragments a set of B-spline surfaces within a defined bounding box using GMSH.
    This function creates a 3D bounding box and fragments it with the provided B-spline surfaces.
    The surfaces are sequentially fragmented to ensure they are correctly incorporated.
    After fragmentation, GMSH synchronizes the changes to update the model.

    Args:
        surfaces (list):
            A list of surface IDs representing B-spline surfaces in GMSH.
        extent (tuple):
            A tuple (x_min, x_max, y_min, y_max, z_min, z_max) defining the bounding box for fragmentation.

    Returns:
        tuple: Two lists (ov, ovv) containing the fragmented volume and surface entities.
    """

    # Create a box for fragmenting
    x_min, x_max, y_min, y_max, z_min, z_max = extent
    v = gmsh.model.occ.addBox(x_min, y_min, z_min, x_max, y_max, z_max)

    # Fragment the box with all the B-spline surfaces
    fragmented_surfaces = [(2, surfaces[0])]
    for surface in surfaces[1:]:
        try:
            fragmented_surfaces, _ = gmsh.model.occ.fragment(
                fragmented_surfaces, [(2, surface)]
            )
            gmsh.model.occ.synchronize()
            print(f"Fragmented with surface {surface}")
        except Exception as e:
            print(f"Fragmentation failed with surface {surface}: {e}")

    # Final fragmentation of the bounding box with the surfaces
    ov, ovv = gmsh.model.occ.fragment(
        [(3, v)], [(2, s[1]) for s in fragmented_surfaces],
        removeObject=True,
        removeTool=True
    )

    gmsh.model.occ.synchronize()
    return ov, ovv


def mesh_generator(ov, mesh_size=50):
  # Extract volumes tags
  volumes = [tag for dim, tag in ov if dim == 3]

  print("Volumes (3D):", volumes)

  # Add physical groups for volumes only
  for i, tag in enumerate(volumes):
      gmsh.model.addPhysicalGroup(3, [tag], i + 1)
      gmsh.model.setPhysicalName(3, i + 1, f"Volume {i + 1}")
      print((3, i + 1, f"Volume {i + 1}"), 'physical')

  # Assign a mesh size to all the points

  gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

  # Generate 3D mesh
  gmsh.model.mesh.generate(3)
  # Visualize the result (if needed)
  gmsh.fltk.initialize()
  while gmsh.fltk.isAvailable():
      gmsh.fltk.wait()

  ########################## <get nodes and elements for creating different mesh formats #######################
  # Get nodes Tag and their coordinates
  nodeTags,coord, n = gmsh.model.mesh.getNodesByElementType(4,-1,  False)
  # Make it unique
  uniqueNodeTags = set(nodeTags)
  # Convert back to a list
  uniqueNodeTagsList = list(uniqueNodeTags)
  # Make it an integer array
  uniqueNodeTagsList = np.array(uniqueNodeTagsList, dtype=int)
  # Create an empty list to store the results
  node_data = []

  for n in uniqueNodeTagsList:
      coord = gmsh.model.mesh.getNode(n)
      node_data.append([n-1, coord[0][0], coord[0][1], coord[0][2]])

  # Convert to NumPy array
  unique_nodes_with_coords = np.array(node_data)

  # Get elements grouped by volume
  elements_by_volume_array = []
  for volume in volumes:
      dim = 3
      elementTypes, element_tags, nodeTags = gmsh.model.mesh.getElements(dim, volume)
      elements_by_volume_array.append((element_tags[0], volume))

  # Create a volume lookup table
  volume_map = {}
  for  elements, vol in elements_by_volume_array:
      print(f"Volume {vol} contains elements: {elements}", len(elements))
      for el in elements:
          volume_map[el] = vol

  # Get element ids and respective nodes
  elementType = gmsh.model.mesh.getElementType("tetrahedron", 1)
  elementTags, elementNodeTags= gmsh.model.mesh.getElementsByType(elementType)
  element_ids = elementTags
  # Reshape node data in order to have 4 nodes (since elements are tetrahedrons) in each row
  nodes_reshaped = elementNodeTags.reshape(-1, 4)

  # Find the volume for each element id
  volumes = np.array([volume_map[el] for el in element_ids])
  # Stack element ids, their respective nodes, and volume info
  elements = np.column_stack((element_ids-1, nodes_reshaped-1, volumes))


  # Finalize GMSH
  gmsh.finalize()
  return elements, unique_nodes_with_coords


def create_unstructed_mesh_data(geomodel_result, mesh_size=20):
    """
    Generates a geological mesh and returns a MeshData object.

    Args:
        results_test (object): Object containing data to create the grid.
        refinement_data (list): list of refinement values.
        z_threshold (float): Threshold for Z-value adjustment.
        tolerance (float): Distance tolerance for Z-value adjustment.

    Returns:
        MeshResults: An instance of the MeshResults class.
    """


    interpolated_s = create_surface_grid(geomodel_result)
    surfaces= import_surfaces(interpolated_s)

    # fragment
    # Extract extent values
    extent = geomodel_result.extent
    ov,ovv = fragment_surfaces(surfaces, extent)
    elements, nodes = mesh_generator(ov, mesh_size)
    # Change the format
    nodes = np.array(nodes, dtype=float)
    elements = elements.astype(int)
    # surface id stats from 1. Changing it to start from zero
    # Subtract 1 from the last column
    elements[:, -1] -= 1
    # Since in implicit mesh the numbering is reversed, here I also reverse them
    # Find unique values in the last column
    unique_values = np.unique(elements[:, -1])
    # Create a mapping: max value → 0, min value → max, etc.
    mapping = {val: i for i, val in enumerate(unique_values[::-1])}
    # Apply the mapping to the last column
    elements[:, -1] = np.vectorize(mapping.get)(elements[:, -1])
    print("Shape of nodes_array:", elements.shape)
    print("First few rows:", elements[:5])
    # Create and return a MeshData instance
    return MeshResults(elements=elements,
                       nodes=nodes,
                       )
