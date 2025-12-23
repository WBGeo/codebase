import typing

import gmsh
import meshio
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

from typing import List, Tuple, Union, Mapping
from core.object_components import InputData, GeomodelResults
from core.object_components import MeshResults
from core.meshing_components.explicit.unstructured.create_grid_fragment_surface import create_surface_grid, import_surfaces, fragment_surfaces, plot_surfaces_individually
from core.meshing_components.explicit.unstructured.create_clean_surface import data_prepration

from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType


def point_on_line_segment(pt, p1, p2, tol=1e-6):
    """
    Check if a point lies on the infinite line defined by p1 and p2 using symmetric line equation.
    """
    pt = np.array(pt, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    # Avoid division by zero: check each coordinate separately
    ratios = []
    for i in range(3):
        delta = p2[i] - p1[i]
        if abs(delta) > tol:
            ratio = (pt[i] - p1[i]) / delta
            ratios.append(ratio)
        else:
            # If delta is nearly zero, pt must match p1 in this coordinate
            if abs(pt[i] - p1[i]) > tol:
                return False

    # Now check all computed ratios are (nearly) the same
    if len(ratios) <= 1:
        return True  # Point lies on a degenerate or axis-aligned line
    return max(ratios) - min(ratios) < tol



def mesh_generator(ov, tagsss,  wells, well_tags, source_tag, shaft_tags,shaft_to_child_fragments, grid_litho, mesh_size=20, curve_mesh_size=5):
  """
    The function uses GMSH to create and export a 3D tetrahedral mesh from given volumes. It samples 150 random
    nodes from each tetrahedral block and assigns a dominant lithology based on proximity to known lithology
    points (`grid_litho`). If no lithology dominates, the block is ignored. Tetrahedral blocks with the same
    lithology are merged into single blocks for export. The function returns the final cleaned-up mesh with
    tetrahedra grouped by lithology and surfaces preserved.

    Args:
    ov : List of GMSH entities where each tuple contains the dimension and tag of an entity. Only 3D volumes (dimension == 3) will be considered for meshing.
    tagsss : List of GMSH physical group tags for fault surfaces (triangles) that should be preserved and re-associated with corresponding triangle mesh elements in the final mesh.
    grid_litho : A DataFrame containing 4 columns: the first three represent x, y, z coordinates, and the fourth column contains lithology numbers. These are used to assign lithology to mesh blocks via nearest-neighbor classification.
    number_random_sample : number of samples selected from mesh blocks which should be compared with geological grid points to assign the correct physical group for each layer
    mesh_size : Mesh element size to be used by GMSH during mesh generation (default is 20).

    Returns:
    nodes : Array of mesh node coordinates (shape: [n_nodes, 3]).
    new_cells : List of meshio CellBlock objects representing the cells (elements) of the mesh. Tetrahedral elements are grouped by lithology, and triangle elements are preserved and re-associated with their physical tags.

  """
  # Extract volumes tags
  volumes = [tag for dim, tag in ov if dim == 3]

  # Add physical groups for volumes only
  for i, tag in enumerate(volumes):
      gmsh.model.addPhysicalGroup(3, [tag], i + 1)
      gmsh.model.setPhysicalName(3, i + 1, f"Volume {i + 1}")
      print((3, i + 1, f"Volume {i + 1}"), 'physical')
  gmsh.model.occ.synchronize()
  gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)

  gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
  gmsh.model.mesh.removeDuplicateNodes()



  # Generate 3D mesh
  if source_tag:
      for volume_tag in volumes:
          for s_tag in source_tag:
              gmsh.model.mesh.embed(0, [s_tag], 3, volume_tag)

  # === Show geometry before meshing ===
  #gmsh.fltk.initialize()  # Start the Gmsh GUI
  #gmsh.fltk.run()         # Keep GUI open until you close it manually
  #gmsh.fltk.finalize()    # Close GUI cleanly

  # Finally, let's specify a global mesh size and mesh the partitioned model:
  gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curve_mesh_size)

  gmsh.option.setNumber("General.Verbosity", 4)

  gmsh.model.mesh.generate(3)


  gmsh.model.occ.synchronize()
  ####gmsh.fltk.initialize()
  ###while gmsh.fltk.isAvailable():
  ###      gmsh.fltk.wait()
  # Save the mesh
  mesh_file = "generated_mesh.msh"
  gmsh.write(mesh_file)
  # Read the mesh and get nodes and elements information
  mesh_model=meshio.read(mesh_file)
  nodes = mesh_model.points  # Coordinates of the nodes
  cells = mesh_model.cells  # Elements (cells)


  # Access the cells and their types
  for block in mesh_model.cells:
      print(f"Cell type: {block.type}, Number of cells: {len(block.data)}")
  # Get tetra blocks (only 3D elements)
  tetra_blocks = [block for block in cells if block.type == "tetra"]
  print(f"Number of tetrahedral blocks: {len(tetra_blocks)}")


  if shaft_tags:
      tetra_blocks_orgin=tetra_blocks
      # Reconstruct tag → block mapping using the known volumes and shaft fragments
      tag_dict = {}

      # Step 1: Reconstruct tag → block mapping
      tag_list_all = volumes.copy()

      for shaft_tag, frag_list in shaft_to_child_fragments.items():
        tag_list_all.extend(frag_list)
        tag_dict[shaft_tag] = frag_list


      assert len(tag_list_all) == len(tetra_blocks_orgin), \
          f"Mismatch: {len(tag_list_all)} tags vs {len(tetra_blocks_orgin)} tetra blocks"
      tag_list_all.sort()
      tag_to_block = dict(zip(tag_list_all, tetra_blocks_orgin))

      # Build full set of shaft child tags
      shaft_child_tags = set()
      for frag_list in shaft_to_child_fragments.values():
          shaft_child_tags.update(frag_list)


      # Use original tag order to separate blocks
      regular_blocks = []
      shaft_blocks_dict = {}  # key = shaft_tag, value = list of blocks

      # 1. First, create shaft block lists using tag_dict
      for shaft_tag, frag_list in tag_dict.items():
        shaft_blocks = []
        for tag in frag_list:
            block = tag_to_block[tag]

            shaft_blocks.append(block)
        shaft_blocks_dict[shaft_tag] = shaft_blocks  # Save blocks per shaft


      # 2. Then, build regular blocks (those not in any shaft)
      all_shaft_child_tags = set(tag for frag_list in tag_dict.values() for tag in frag_list)

      for tag in tag_list_all:
        if tag not in all_shaft_child_tags:
            block = tag_to_block[tag]
            regular_blocks.append(block)

      # Update the main tetra_blocks with only the regular ones
      tetra_blocks = regular_blocks

      # Optionally, print the results
      print(tetra_blocks, "Regular tetra blocks after removing shafts")
      print(f"Number of shafts: {len(shaft_blocks_dict)}")
      for shaft_tag, blocks in shaft_blocks_dict.items():
        print(f"Shaft {shaft_tag} has {len(blocks)} blocks")


  # Precompute KD-tree
  grid_coords = grid_litho.iloc[:, :3].to_numpy()
  grid_litho_values = grid_litho.iloc[:, 3].to_numpy()
  tree = cKDTree(grid_coords)

  threshold_ratio = 0.7  # fraction of nodes needed to assign a lithology
  lithology_numbers = []

  for block in tetra_blocks:
    node_ids = np.unique(block.data)
    node_coords = nodes[node_ids]

    # Query nearest lithology point for all nodes in block
    _, nearest_idx = tree.query(node_coords, k=1)
    node_litho = grid_litho_values[nearest_idx]

    # Majority vote
    unique_vals, counts = np.unique(node_litho, return_counts=True)
    max_count_idx = np.argmax(counts)

    if counts[max_count_idx] / len(node_ids) >= threshold_ratio:
        lithology_numbers.append(int(unique_vals[max_count_idx]))
    else:
        # fallback: assign the lithology of the centroid
        centroid = np.mean(node_coords, axis=0)
        _, nearest_idx = tree.query(centroid)
        lithology_numbers.append(int(grid_litho_values[nearest_idx]))

  # Group blocks by lithology
  litho_to_blocks = defaultdict(list)
  for block_index, (lith, block) in enumerate(zip(lithology_numbers, tetra_blocks)):
    litho_to_blocks[lith].append(block_index)

  # Merge blocks per lithology
  merged_tetra_blocks = []
  for lith, block_indices in litho_to_blocks.items():
    merged_nodes = np.concatenate([tetra_blocks[idx].data for idx in block_indices])
    merged_tetra_blocks.append(meshio.CellBlock(cell_type="tetra", data=merged_nodes))

  print("Number of merged lithology blocks:", len(merged_tetra_blocks))





  # === Add shaft blocks as a separate CellBlock ===

  if shaft_tags:
      for shaft_tag, shaft_blocks in shaft_blocks_dict.items():
          all_shaft_data = []
          for block in shaft_blocks:
              all_shaft_data.extend(block.data)
          merged_tetra_blocks.append(
              meshio.CellBlock(cell_type="tetra", data=np.array(all_shaft_data))
          )

  # === Now merged_tetra_blocks includes lithology-based and shaft_child CellBlocks ===

  print(f"Total merged meshio blocks: {len(merged_tetra_blocks)}")
  print(f"Lithologies: {list(litho_to_blocks.keys())}, + shaft_child group")

  # Print details of merged blocks
  for i, block in enumerate(merged_tetra_blocks):
      print(f"Merged Block {i}: {block.data.shape[0]} tetrahedra")
  # Save the merged_blocks
  if merged_tetra_blocks:
    cells_n = [block for block in cells if block.type != "tetra"]  # Keep non-triangle blocks
    cells_n.extend(merged_tetra_blocks)  # Add merged triangle block
  else:
    cells_n= cells

  new_cells = [block for block in cells_n if (block.type != "triangle") and (block.type != "line") and (block.type != "vertex")]


  if tagsss:
      # Extract only triangle elements from cells
      triangle_blocks = [block for block in cells_n if block.type == "triangle"]
      other_blocks = [block for block in cells_n if block.type not in {"triangle", "line", "vertex"}]



      # Get GMSH global nodes and reshape coordinates
      all_tags, all_coords_flat, _ = gmsh.model.mesh.getNodes()
      all_coords = np.array(all_coords_flat).reshape(-1, 3)

      # Match meshio nodes to GMSH tags via KDTree
      tree = cKDTree(all_coords)
      dist, idx = tree.query(nodes, distance_upper_bound=1e-1)


      # Build mapping between meshio indices <-> GMSH tags
      # meshio index → GMSH tag ;
      # idx: the result of cKDTree.query(...), gives for each meshio point the index j of the closest GMSH point (within the distance_upper_bound).
      # all_tags[j]: gives the GMSH tag corresponding to that matched node.
      # i: is the meshio local index.
      local_to_gmsh_tag = {i: all_tags[j] for i, j in enumerate(idx) if j < len(all_tags)}
      # Just inverts the previous mapping.
      # GMSH tag → meshio index
      gmsh_tag_to_local = {v: k for k, v in local_to_gmsh_tag.items()}

      # For each tag (surface), find triangles that belong to it
      tag_to_triangles = {}

      for tag in tagsss:
          # Get GMSH node tags on this surface
          surface_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, tag)

          # Convert to local meshio indices
          local_surface_nodes = {gmsh_tag_to_local[t] for t in surface_tags if t in gmsh_tag_to_local}

          # Collect triangles with ≥2 nodes on the surface
          matched_triangles = []

          for block in triangle_blocks:
              for tri in block.data:
                  if sum(n in local_surface_nodes for n in tri) == 3:
                      matched_triangles.append(tri)

          if matched_triangles:
              tag_to_triangles[tag] = np.array(matched_triangles)

      # Build new_cells with surface triangles grouped by tag
      new_cells = other_blocks.copy()

      for tag, triangles in tag_to_triangles.items():
          new_cells.append(meshio.CellBlock(cell_type="triangle", data=triangles))



  if well_tags:
          well_lines = []  # Each element is a list of segments for one well

          for well_flat in wells:
              # Group every 3 values into a 3D point
              well_points = [(well_flat[i], well_flat[i + 1], well_flat[i + 2]) for i in range(0, len(well_flat), 3)]

              if len(well_points) < 2:
                  continue  # Not enough points to make a line

              segments = []
              for i in range(len(well_points) - 1):
                  p1 = well_points[i]
                  p2 = well_points[i + 1]
                  segments.append([p1, p2])  # Each segment is a list of 2 points

              well_lines.append(segments)  # Add this well's segments to the main list



          line_blocks = [block for block in cells_n if block.type == "line"]
          points_by_well = {i: [] for i, well in enumerate(well_tags)}



          for block in line_blocks:
              for j in range(len(block.data)):
                  point1 = nodes[int(block.data[j][0])]
                  point2 = nodes[int(block.data[j][1])]
                  for well_id, segments in enumerate(well_lines):
                    for p1, p2 in segments:
                        if point_on_line_segment(point1, p1, p2) or point_on_line_segment(point2, p1, p2):
                            points_by_well[well_id].append(block.data)
                            break  # Match found for this segment, break inner loop
                    else:
                        continue  # No match in this well, continue to next well
                    break  # Match found in this well, skip to next line


          for line_tag, line_points in points_by_well.items():

            if len(line_points) > 0:  # skip empty
                new_cells.append(meshio.CellBlock(cell_type="line", data=np.vstack(line_points)))

          #new_cells.append(meshio.CellBlock(cell_type="line", data=points_by_well))


  if source_tag:
        vertex_blocks = [block for block in cells_n if block.type == "vertex"]
        # Original vertex tags (point indices)
        vertex_indices = np.concatenate([block.data.flatten() for block in vertex_blocks])



        for idx in (vertex_indices):
            new_cells.append(meshio.CellBlock(cell_type="vertex", data=np.array([[idx]])))



  if new_cells:
    return nodes, new_cells

  else:
    print('No tags found')
    return nodes, cells_n

WellData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='well_list', color='aqua', identifier='mesh::WellListData', controlled='Table|x3')]
SourcesData = typing.Annotated[List[Tuple[float, float, float]], AnnotatedScriptType(name='sources', color='aqua', identifier='mesh::SourcesData', controlled='Table|3|X|Y|Z')]

CenterData = typing.Annotated[List[Tuple[float, float, float]], AnnotatedScriptType(name='sources', color='aqua', identifier='mesh::SourcesData', controlled='Table|3|X|Y|Z')]
AxesData = typing.Annotated[List[Tuple[float, float, float]], AnnotatedScriptType(name='axes', color='aqua', identifier='mesh::AxesData', controlled='Table|3|X|Y|Z')]
RadiData = typing.Annotated[List[float], AnnotatedScriptType(name='radi', color='aqua', identifier='mesh::RadiData', controlled='Table|1|Radius')]

PlanesData = typing.Annotated[List[Tuple[float, ...]], AnnotatedScriptType(name='planes', color='aqua', identifier='mesh::PlanesData')]
RadiiData = typing.Annotated[List[float], AnnotatedScriptType(name='radii', color='aqua', identifier='mesh::RadiiData')]
ExtentData = typing.Annotated[List[float], AnnotatedScriptType(name='extent', color='aqua', identifier='mesh::ExtentData')]


# the file must end with "wells.csv", e.g., "example_wells.csv", etc.
WellCSVDataType = typing.Annotated[str, AnnotatedScriptType(name='path', color='aqua', identifier='wbgeo::well_csv', controlled='RemoteFile|endswith=wells.csv')]

@wbgeo_component(description='Loads a well from a wells.CSV file',
                 title='Load Well',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='wbgeo::meshing_load_well_from_csv',  # a unique identifier
                 return_name='Wells',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def load_wells_from_csv(well_file: WellCSVDataType) -> WellData:
  # format of the csv is: id, x,y,z\n
  named_well_data: Mapping[str, List[Tuple[float]]] = {}
  # load csv file
  with open(well_file, 'r') as f:
    for line in f.readlines():
      if line.startswith('#'): continue
      well_id, well_x, well_y, well_z = [s.strip() for s in line.split(",")]
      if well_id not in named_well_data:
        named_well_data[well_id] = []
      named_well_data[well_id].append([float(well_x), float(well_y), float(well_z)])
  # ensure that we have at least 2 points per well
  if any(True for well in named_well_data.values() if len(well) < 2):
    incorrect_wells = [well_id for well_id, well_data in named_well_data.items() if
                       len(well_data) < 2]
    raise ValueError(f"Some well(s) {incorrect_wells} are missing their second point")
  # format right now is {key: [(x,y,z)]} -> map it to [x1, y1, z1, ..., xi, yi, zi] for each well
  return [tuple([coordinate for well_group in well_data for coordinate in well_group]) for well_data
          in named_well_data.values()]

# Register this function as a component
@wbgeo_component(description='Provides unstructured mesh',
                 title='Create Unstructured Mesh',  # The title shown in the GUI
                 color='#cc9999',  # the color of the components
                 border_color='#000000',  # and its border color
                 group='Meshing',
                 identifier='create_unstructured_mesh_data',  # a unique identifier
                 return_name='Mesh',  # the name for the returned-port
                 )  # inputs are handled via the method signature
def create_unstructured_mesh_data_showcase( # for the demo: Only show a limited amount of inputs
    geomodel_result: GeomodelResults,
    wells: WellData,
    tolerance: float = 50,
    mesh_size: float = 30,
    curve_mesh_size: float = 5,
    DISTANCE_THRESHOLD: float = 50,
    PROJECTION_THRESHOLD: float = 60,
    EXTRUSION_FACTOR: float = 100,
    z_threshold: float = 10,
    buffer_dist: float = 0,
    smooth: float = 1e-5 ) -> MeshResults:
  return create_unstructured_mesh_data(**locals())

def create_unstructured_mesh_data(
    geomodel_result: GeomodelResults,
    wells: WellData = [],
    sources: SourcesData = [],
    centers: CenterData = [],
    axes: AxesData = [],
    radii: RadiiData = [],
    extra_planes: PlanesData = [],
    tolerance: float = 50,
    mesh_size: float = 30,
    curve_mesh_size: float = 5,
    DISTANCE_THRESHOLD: float = 50,
    PROJECTION_THRESHOLD: float = 60,
    EXTRUSION_FACTOR: float = 100,
    z_threshold: float = 10,
    extent: ExtentData = [],
    buffer_dist: float = 0,
    smooth: float = 1e-5 ) -> MeshResults:
    """
    Generates an unstructured geological mesh using a geomodel and additional structures
    such as wells, sources, shafts, and extra planes. It performs surface cleaning,
    fragmentation, and meshing using GMSH and returns the final MeshData object.

    Args:
        input_data (InputData): Input data object containing surface points, orientations, mapping, faults, and extent.
        geomodel_result (object): Output object from the geomodel interpolation, e.g. from `universal_cokriging_interpolator`.
        wells (list of tuples): Each tuple contains coordinates defining the top (, middel) and bottom of a well (x1, y1, z1, x2, y2, z2).
        sources (list of tuples): Each tuple contains coordinates (x, y, z) of a point source.
        centers (list of tuples): List of coordinates for the centers of mine shaft cylinders (x, y, z).
        axes (list of tuples): List of direction vectors (dx, dy, dz) for the axes of mine shaft cylinders.
        radii (list of floats): List of radii for the mine shaft cylinders.
        extra_planes (list of tuples): Each tuple contains coordinates of 4 corners (12 values) defining an extra plane.
        tolerance (float): Distance threshold to identify boarder of mesh.
        mesh_size (int): Default mesh size for surface and volume meshing (default is 30).
        curve_mesh_size (int): Mesh size applied to curves (default is 5).
        DISTANCE_THRESHOLD (float): Maximum distance used to filter overlapping points between surfaces.
        PROJECTION_THRESHOLD (float): Distance threshold for projecting points when calculating extrusion.
        EXTRUSION_FACTOR (float): Factor that scales extrusion distance.
        z_threshold (float): Threshold for determining whether two surfaces on either side of a fault are close in elevation.
        extent (list): extent of mesh (min_x, max_x, min_y,max_y, min_z, max_z)
        buffer_dist (float): extent of interpolated surfaces (extrapolation)
        smooth (float): smoothness factor for interpolation of surfaces
    Returns:
        MeshResults: An instance of the MeshResults class.
    """

    # Validate that wells is a list of tuples with at least 6 coordinates and length is a multiple of 3
    if not isinstance(wells, list) or not all(isinstance(w, tuple) and len(w) >= 6 and len(w) % 3 == 0 for w in wells):
      raise Exception("❌ 'wells' must be a list of tuples, each containing 2 or more 3D coordinate points (e.g., 6, 9, 12 values, etc.).")

    # Validate that sources is a list of tuples
    if not all(isinstance(s, tuple) and len(s) == 3 for s in sources):
      raise Exception("❌ 'sources' must be a list of 3D coordinate tuples like [(x, y, z), ...].")
    # Validate that centers is a list of 3D tuples
    if not all(isinstance(center, tuple) and len(center) == 3 for center in centers):
      raise Exception("❌ 'centers' must be a list of 3D coordinate tuples like [(x, y, z), ...].")
    # If shafts are specified, check the number of centers
    # Validate that axes is a list of tuples
    if not all(isinstance(axis, tuple) and len(axis) == 3 for axis in axes):
      raise Exception("❌ 'axes' must be a list of 3D coordinate tuples like [(x, y, z), ...].")
    # Check number of axes matches num_shafts if specified
    if len(axes) != len(radii):
        raise Exception(f"❌ Number of axes ({len(axes)}) does not match 'radii' ({len(radii)}).")
    if len(axes) != len(centers):
      raise Exception(f"❌ Number of axes ({len(axes)}) does not match 'center' ({len(center)}).")
    # Validate that radiis is a list of int
    if not isinstance(radii, list) or not all(isinstance(r, int) for r in radii):
        raise Exception("❌ 'radii' must be a list of integers like [10, 20, 30].")
    num_shafts = len(radii)

    # Validate that sources is a list of tuples
    if not all(isinstance(extra, tuple) and len(extra) == 12 for extra in extra_planes):
        raise Exception("❌ 'extra_planes' must be a list of four sets of 3D coordinate tuples like [(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4), ...].")
    num_planes = len(extra_planes)

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


    gmsh.initialize()  # Initialize GMSH once
    cleaned_surfaces, ref_surface_indices , grid_litho = data_prepration(geomodel_result, DISTANCE_THRESHOLD = DISTANCE_THRESHOLD, PROJECTION_THRESHOLD = PROJECTION_THRESHOLD,
                                                                                                                EXTRUSION_FACTOR = EXTRUSION_FACTOR, z_threshold = z_threshold)

    interpolated_s = create_surface_grid(cleaned_surfaces, buffer_dist = buffer_dist, smooth=smooth)

    # fragment
    if extent ==[]:
        extent = geomodel_result.extent
    else:
        extent= np.array(extent)


    surfaces_orginal, bounds = import_surfaces(interpolated_s, extent, tolerance=tolerance)
    print(bounds, 'biii')

    gmsh.model.occ.synchronize()
    #gmsh.fltk.initialize()
    #while gmsh.fltk.isAvailable():
    #    gmsh.fltk.wait()





    surfaces=surfaces_orginal.copy()
    ov,ovv, tagssss, well_tags, shaft_tags,shaft_to_child_fragments,  source_tag = fragment_surfaces(surfaces, bounds, ref_surface_indices,wells, extra_planes, sources, mine_shafts, mesh_size=mesh_size,curve_mesh_size=curve_mesh_size )
    nodes, cells = mesh_generator(ov, tagssss, wells, well_tags, source_tag, shaft_tags, shaft_to_child_fragments,  grid_litho, curve_mesh_size=curve_mesh_size )


    # Create and return a MeshData instance
    return MeshResults(elements_unstructured=cells,
                       nodes=nodes,
                       )
