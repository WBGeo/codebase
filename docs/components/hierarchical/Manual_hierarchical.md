# Hierarchical Meshing

Hierarchical meshing creates two nested meshes of the same geological
model: a larger **outer mesh** (`mesh1`) and a smaller **inner mesh**
(`mesh2`). The inner mesh represents a selected subregion of the outer
domain and can use a different mesh resolution. The workflow also
identifies nodes in the outer mesh that correspond most closely to the
six boundary faces of the inner mesh. These node sets can be used when
coupling the two meshes in a hierarchical simulation.

The current implementation supports **unstructured**, **structured**,
and **implicit structured** hierarchical meshes.

> **Note:** `extent2` must lie completely inside `extent1`. The
> hierarchical workflow always creates two meshes.

------------------------------------------------------------------------

## Concepts

The hierarchical workflow is based on two spatial levels:

-   **Mesh 1** is the larger or outer mesh. Its extent is defined by
    `extent1`.
-   **Mesh 2** is the smaller or inner mesh. Its extent is defined by
    `extent2`.

The two meshes are generated independently from the same structural
model. After generation, the six boundary faces of `mesh2` are detected
from its minimum and maximum x, y, and z coordinates. A
nearest-neighbour search is then used to find the corresponding nodes in
`mesh1`.

The resulting boundary-node groups are:

-   `left`
-   `right`
-   `front`
-   `back`
-   `bottom`
-   `top`

For each matching node in `mesh1`, the stored boundary data contain the
node index followed by its `X`, `Y`, and `Z` coordinates.

------------------------------------------------------------------------

## Workflow

### 1. Create the hierarchical meshes

The main function is:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(...)
```

Hierarchical meshing can use either a WBGeo structural-model result or, for unstructured meshing, geological surfaces stored as CSV files.

The geological input options are:

-   **`geomodel_result`**: Optional WBGeo structural-model result used to generate both meshes. Provide this for the normal structural-model workflow. It may be omitted when `surface_csv_dir` is used.
-   **`surface_csv_dir`**: Optional directory containing geological surface CSV files. This input is supported for `mesh_type="Un"`. When this is supplied, `geomodel_result=None` does not need to be written explicitly because `None` is the default.

At least one geological input, `geomodel_result` or `surface_csv_dir`, must be supplied.

The common hierarchy inputs are:

-   **`mesh_type`**: Mesh type used at both hierarchy levels. Supported
    values are `"Un"` for unstructured, `"Str"` for structured, and
    `"Im"` for implicit structured meshes.
-   **`extent1`**: Extent of the outer mesh. With `geomodel_result`, `"whole"` uses the complete structural-model extent. For CSV-only input, an explicit extent must be supplied because there is no structural model from which to obtain the `"whole"` extent.
-   **`extent2`**: Extent of the inner mesh. This must be completely
    contained within `extent1`.

### Unstructured hierarchical mesh

For `mesh_type="Un"`, the principal resolution controls are:

-   **`mesh_size1`**: Target mesh size for the outer mesh. This is
    required.
-   **`mesh_size2`**: Target mesh size for the inner mesh. If omitted,
    it uses the value of `mesh_size1`.
-   **`unstructured_params`**: Common additional keyword arguments
    passed to both unstructured hierarchy levels.
-   **`unstructured_params1`**: Optional parameters that override
    `unstructured_params` for the outer mesh (`mesh1`).
-   **`unstructured_params2`**: Optional parameters that override
    `unstructured_params` for the inner mesh (`mesh2`). For example,
    this can be used to give `mesh2` a different `rotation_angle`.

### Geological CSV surfaces instead of `geomodel_result`

For an unstructured hierarchical mesh, geological surfaces can be read from the same CSV-surface directory supported by the normal unstructured mesher. The CSV processing remains inside `create_unstructured_mesh_data()`; `hierarchical_mesh()` passes the directory to both hierarchy levels.

`geomodel_result=None` does **not** need to be specified explicitly. For example:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(
    mesh_type="Un",
    surface_csv_dir=surface_csv_dir,

    extent1=[0, 1200, 0, 1000, 0, 1000],
    extent2=[300, 900, 200, 900, 100, 800],

    mesh_size1=30,
    mesh_size2=20,

    unstructured_params={
        "curve_mesh_size": 5,
        "mapping_litho": "none",
    },
    unstructured_params2={
        "rotation_angle": 45,
    },
)
```

For CSV-only input, use an explicit `extent1`; `extent1="whole"` requires `geomodel_result`. Engineering objects and refinement options can be supplied in the same way as in the structural-model workflow.

Common options such as `curve_mesh_size` and `refinement` can therefore
be defined once in `unstructured_params`, while level-specific options
can be defined separately.

The hierarchical unstructured workflow can also receive the same
engineering and geometric objects used by the standard unstructured
mesher:

-   `wells`
-   `sources`
-   `shafts`
-   `extra_planes`
-   `ellipses`
-   `triangulations`

Engineering objects are clipped to the extent of each hierarchy level
before that mesh is generated. This allows the same physical object to
cross the boundary of the smaller inner mesh.

Example with hierarchical engineering objects, refinement, and a
separate rotation for the inner mesh:

``` python
engineering_objects_dir = (
    cwd + "/examples/synthetic_examples/model1/input_data/engineering_objects/"
)

# Load engineering objects in hierarchical mode.
hierarchical_wells = load_wells_from_csv(
    engineering_objects_dir + "model_1_wells_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_sources = load_sources_from_csv(
    engineering_objects_dir + "model_1_sources_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_mining_shafts = load_shafts_from_csv(
    engineering_objects_dir + "model_1_shafts_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_ellipses = load_ellipses_from_csv(
    engineering_objects_dir + "model_1_ellipses_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_extra_planes = load_planes_from_csv(
    engineering_objects_dir + "model_1_planes_hierarchical.csv",
    key_hierarchical=True,
)

hierarchical_triangulation_files = (
    engineering_objects_dir + "seismic_plane_new_offset_0_hierarchical.csv",
    engineering_objects_dir + "seismic_plane_new_offset_1_hierarchical.csv",
)

hierarchical_triangulations = load_triangulations_planes_from_csv(
    hierarchical_triangulation_files,
    key_hierarchical=True,
)

# Refinement.
refinement = Refinement()
refinement.wells = LinearWellRefinement(
    SizeMin=10.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.sources = LinearSourceRefinement(
    SizeMin=10.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.shafts = LinearShaftRefinement(
    SizeMin=10.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.ellipses = EllipseRefinement(
    SizeMin=15.0, SizeMax=80.0, DistMin=30.0, DistMax=80.0
)
refinement.extraplanes = ExtraPlaneRefinement(
    SizeMin=15.0, SizeMax=80.0, DistMin=20.0, DistMax=60.0
)
refinement.triangulations = TriangulationRefinement(
    SizeMin=15.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0
)

mesh1, mesh2, nodes_bcs = hierarchical_mesh(
    geomodel_result=structural_model_result,
    mesh_type="Un",

    extent1="whole",
    extent2=[300, 900, 200, 900, 100, 800],

    wells=hierarchical_wells,
    sources=hierarchical_sources,
    shafts=hierarchical_mining_shafts,
    ellipses=hierarchical_ellipses,
    extra_planes=hierarchical_extra_planes,
    triangulations=hierarchical_triangulations,

    mesh_size1=30,
    mesh_size2=20,

    # Common options for both hierarchy levels.
    unstructured_params={
        "curve_mesh_size": 5,
        "refinement": refinement,
    },

    # Override only the inner-mesh rotation.
    unstructured_params2={
        "rotation_angle": 45,
    },
)
```

Here, `mesh1` covers the complete structural-model extent, while `mesh2`
covers only the region specified by `extent2`. The smaller value of
`mesh_size2` gives the inner mesh a finer target resolution. Both meshes
inherit `curve_mesh_size` and `refinement`, while `mesh2` additionally
uses a 45-degree rotation.

Different rotations can be assigned to both hierarchy levels, for
example:

``` python
unstructured_params1={
    "rotation_angle": 20,
},
unstructured_params2={
    "rotation_angle": 45,
},
```

The physical boundary nodes are still obtained from `mesh2.point_sets`,
and the nearest-node search is performed in a common coordinate frame.

### Structured hierarchical mesh

For `mesh_type="Str"`, use the structured-mesh controls instead of
`mesh_size1` and `mesh_size2`:

-   **`refinement1`**: Vertical refinement definition for mesh 1.
-   **`refinement2`**: Vertical refinement definition for mesh 2. If
    omitted, `refinement1` is reused.
-   **`mesh_devision1`**: Horizontal mesh-division definition for mesh
    1.  
-   **`mesh_devision2`**: Horizontal mesh-division definition for
    mesh 2. If omitted, `mesh_devision1` is reused.
-   **`structured_params`**: Additional parameters forwarded to the
    normal structured-mesh generator.

`refinement1` and `mesh_devision1` are required for a structured
hierarchy. Unstructured `mesh_size` parameters must not be supplied.

### Implicit hierarchical mesh

For `mesh_type="Im"`, the two meshes are generated using the implicit
structured-mesh workflow.

-   **`implicit_params`**: Additional keyword arguments passed to the
    implicit mesh generator.

Implicit hierarchical meshes do not use `mesh_size`, `refinement`, or
`mesh_devision` parameters.

------------------------------------------------------------------------

## Boundary-node matching

For an unstructured hierarchical mesh, WBGeo uses the boundary
`point_sets` already generated by the unstructured mesher. The
inner-mesh boundary is therefore not detected again from `xmin`, `xmax`,
`ymin`, `ymax`, `zmin`, or `zmax`.

The relevant node sets are:

``` python
mesh2.point_sets["left"]
mesh2.point_sets["right"]
mesh2.point_sets["front"]
mesh2.point_sets["back"]
mesh2.point_sets["bottom"]
mesh2.point_sets["top"]
```

Each point set contains the IDs of nodes belonging to that physical face
of `mesh2`. Their coordinates are obtained from `mesh2.nodes`.

This is particularly important for rotated or oriented inner domains
because the physical faces do not need to be parallel to the global
coordinate axes.

For every boundary node in `mesh2`, the hierarchical workflow searches
for the nearest node in `mesh1`. If `mesh1` and `mesh2` use different
`rotation_angle` values, their node coordinates are first compared in
the same original/world coordinate frame. Boundary identification
therefore comes from `point_sets`, while rotation/translation handling
is applied only to the nearest-node matching.

The third output is a dictionary:

``` text
nodes_bcs = {
    "left": ...,
    "right": ...,
    "front": ...,
    "back": ...,
    "bottom": ...,
    "top": ...,
}
```

Each array contains:

``` text
NodeID, X, Y, Z
```

These are the matched nodes of the **outer mesh (`mesh1`)**
corresponding to the physical boundary-node sets of the inner mesh
(`mesh2`).

------------------------------------------------------------------------

## Export

Hierarchical meshes can be exported to Exodus using
`export_meshes_exodus()`. The function exports both meshes and writes
the matched boundary nodes into separate CSV files.

Example:

``` python
mesh1_exodus = "mesh1r.exo"
mesh2_exodus = "mesh2r.exo"
boundary_nodes_file = "boundary_nodesr"

export_meshes_exodus(
    mesh1,
    mesh2,
    nodes_bcs,
    mesh1_exodus,
    mesh2_exodus,
    boundary_nodes_file,
)
```

This creates:

``` text
mesh1r.exo
mesh2r.exo
boundary_nodesr/
    left_boundary_nodes.csv
    right_boundary_nodes.csv
    front_boundary_nodes.csv
    back_boundary_nodes.csv
    bottom_boundary_nodes.csv
    top_boundary_nodes.csv
```

Each boundary CSV contains:

``` text
NodeID,X,Y,Z
```

The Exodus files can subsequently be used by external simulation and
post-processing software.

------------------------------------------------------------------------

## Visualisation and Inspection

### WBGeo/PyVista visualisation

`plot_hierarchical_meshes_3d()` displays the two hierarchy levels side
by side. Mesh 1 is shown as the larger mesh and Mesh 2 as the smaller
mesh.

Example:

``` python
plot_hierarchical_meshes_3d(
    mesh1,
    mesh2,
    structural_model_result,
    show_plotter=True,
    link_views=True,
    volume_opacity=1.0,
    show_block_controls=True,
)
```

The viewer can link the cameras of the two views and provides individual
block visibility controls. This is useful for comparing lithological and
geometrical features between the outer and inner meshes.

The corresponding Workbench inspector is **Inspect Hierarchical Meshes
3D**.

### ParaView visualisation

After exporting the two meshes to Exodus, both files can be opened in
the same ParaView session:

``` python
open_hierarchical_exodus_in_paraview(
    mesh1_exodus=mesh1_exodus,
    mesh2_exodus=mesh2_exodus,
    paraview_executable=(
        "/Applications/ParaView-5.12.0.app/"
        "Contents/MacOS/paraview"
    ),
)
```

The two meshes appear as separate sources in ParaView's Pipeline
Browser, where their blocks and visibility can be controlled
independently.

ParaView must be installed and accessible through the system `PATH`.
Alternatively, its executable path can be supplied explicitly.

------------------------------------------------------------------------

## Complete Model 1 hierarchical workflow

The practical sequence is:

``` python
# 1. Load hierarchical engineering objects.
hierarchical_wells = load_wells_from_csv(
    engineering_objects_dir + "model_1_wells_hierarchical.csv",
    key_hierarchical=True,
)

# 2. Configure refinement.
refinement = Refinement()
refinement.wells = LinearWellRefinement(
    SizeMin=10.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)

# 3. Generate the outer and inner meshes.
mesh1, mesh2, nodes_bcs = hierarchical_mesh(
    geomodel_result=structural_model_result,
    mesh_type="Un",
    extent1="whole",
    extent2=[300, 900, 200, 900, 100, 800],
    wells=hierarchical_wells,
    mesh_size1=30,
    mesh_size2=20,
    unstructured_params={
        "curve_mesh_size": 5,
        "refinement": refinement,
    },
    unstructured_params2={
        "rotation_angle": 45,
    },
)

# 4. Export both meshes and the matched boundary nodes.
mesh1_exodus = "mesh1r.exo"
mesh2_exodus = "mesh2r.exo"
boundary_nodes_file = "boundary_nodesr"

export_meshes_exodus(
    mesh1,
    mesh2,
    nodes_bcs,
    mesh1_exodus,
    mesh2_exodus,
    boundary_nodes_file,
)

# 5. Open the exported meshes in ParaView.
open_hierarchical_exodus_in_paraview(
    mesh1_exodus=mesh1_exodus,
    mesh2_exodus=mesh2_exodus,
    paraview_executable=(
        "/Applications/ParaView-5.12.0.app/"
        "Contents/MacOS/paraview"
    ),
)

# 6. Inspect both hierarchy levels with WBGeo/PyVista.
plot_hierarchical_meshes_3d(
    mesh1,
    mesh2,
    structural_model_result,
    show_plotter=True,
    link_views=True,
    volume_opacity=1.0,
    show_block_controls=True,
)
```

The ParaView executable shown above is the macOS path used in the Model
1 example. On another system, use the corresponding ParaView executable
or a ParaView installation available through the system `PATH`.

------------------------------------------------------------------------

## Output

`hierarchical_mesh()` returns three objects:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(...)
```

-   **`mesh1`**: `MeshResults` for the larger outer mesh.
-   **`mesh2`**: `MeshResults` for the smaller inner mesh.
-   **`nodes_bcs`**: Dictionary containing the matching outer-mesh nodes
    for the six faces of the inner mesh.

The overall workflow is:

``` text
Structural Model
      |
      v
Hierarchical Mesh
      |
      +----> Mesh 1 (outer)
      |
      +----> Mesh 2 (inner)
      |
      +----> Matching boundary nodes
                 |
                 v
        Exodus + boundary CSV export
                 |
                 v
        WBGeo / ParaView inspection
```

The hierarchy therefore preserves two independently generated mesh
levels while providing the boundary-node information needed to relate
the smaller inner domain to the larger outer domain.
