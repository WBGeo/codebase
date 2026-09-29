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

The two meshes are generated independently from the same geological
input. For all supported hierarchical mesh types (`Un`, `Str`, and `Im`),
the six physical boundary faces of `mesh2` are taken from its existing
`point_sets`. A nearest-neighbour search then finds the corresponding
nodes in `mesh1`.

For unstructured hierarchical meshing, both meshes are first generated
in the original/world coordinate frame without applying the requested
output rotation or translation. Requested unstructured output transforms
are applied only after boundary-node matching. Structured and implicit
structured hierarchical meshes use the same boundary-matching procedure,
but currently do not apply separate hierarchy-level output rotations or
translations.

For each matching node in `mesh1`, the stored boundary data contain the
mesh1 node index followed by `X`, `Y`, and `Z`. The same selected mesh1
node IDs are stored in two coordinate representations: the final `mesh1`
coordinate frame (`big_frame`) and the final `mesh2` coordinate frame
(`small_frame`).

------------------------------------------------------------------------

## Workflow

### 1. Create the hierarchical meshes

The main function is:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(...)
```

Hierarchical meshing can use either a WBGeo structural-model result or,
for unstructured meshing, geological surfaces stored as CSV files.

The geological input options are:

-   **`geomodel_result`**: Optional WBGeo structural-model result used
    to generate both meshes. Provide this for the normal
    structural-model workflow. It may be omitted when `surface_csv_dir`
    is used.
-   **`surface_csv_dir`**: Optional directory containing geological
    surface CSV files. This input is supported for `mesh_type="Un"`.
    When this is supplied, `geomodel_result=None` does not need to be
    written explicitly because `None` is the default.

At least one geological input, `geomodel_result` or `surface_csv_dir`,
must be supplied.

The common hierarchy inputs are:

-   **`mesh_type`**: Mesh type used at both hierarchy levels. Supported
    values are `"Un"` for unstructured, `"Str"` for structured, and
    `"Im"` for implicit structured meshes.
-   **`extent1`**: Extent of the outer mesh. With `geomodel_result`,
    `"whole"` uses the complete structural-model extent. With
    `surface_csv_dir`, `"whole"` uses the combined minimum and maximum
    X, Y, and Z coordinates from the geological CSV surfaces. An explicit
    `extent1` always takes precedence over the inferred whole-domain extent.
-   **`extent_z`**: Optional absolute Z limits `[z_min, z_max]` for an
    inferred outer extent. It is applied when `extent1="whole"` (or when
    the outer extent is otherwise inferred). The X and Y limits continue
    to come from the structural model or the combined CSV surfaces, while
    the inferred Z limits are replaced exactly by `extent_z`. For example,
    if the inferred range is `zmin=-7`, `zmax=10`, then
    `extent_z=[-9, 5]` gives `zmin=-9`, `zmax=5`. These values are model
    coordinates, not extension lengths. They must be finite and satisfy
    `z_min < z_max`. If an explicit `extent1` is supplied, it takes
    precedence and is used unchanged.
-   **`extent2`**: Extent of the inner mesh. This must be completely
    contained within the final resolved `extent1`, including any Z limits
    supplied through `extent_z`.

> **Note:** `extent1` and `extent2` may be supplied either
>    as six values `[xmin, xmax, ymin, ymax, zmin, zmax]` or as eight
>    three-dimensional corner points defining an oriented box. 



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

For an unstructured hierarchical mesh, geological surfaces can be read
from the same CSV-surface directory supported by the normal unstructured
mesher. The CSV processing remains inside
`create_unstructured_mesh_data()`; `hierarchical_mesh()` passes the
directory to both hierarchy levels.

`geomodel_result=None` does **not** need to be specified explicitly. For
example:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(
    mesh_type="Un",
    surface_csv_dir=surface_csv_dir,

    extent1="whole",
    extent_z=[-1000, 500],
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

For CSV-only input, `extent1="whole"` derives the outer X, Y, and Z
limits from the combined minimum and maximum coordinates of all CSV
surfaces. `extent_z` can optionally replace only the inferred Z limits
with absolute model coordinates; X and Y remain inferred from the CSV
surfaces.

For example, if the CSV-derived outer extent is
`[xmin, xmax, ymin, ymax, -7, 10]`, then `extent_z=[-9, 5]` produces
`[xmin, xmax, ymin, ymax, -9, 5]`. The values are not enlargement
distances. They are the final `zmin` and `zmax`, and must satisfy
`z_min < z_max`. If an explicit `extent1` is supplied instead of
`"whole"`, the explicit extent is used as given and `extent_z` does not
modify it. Engineering objects and refinement options can be supplied in
the same way as in the structural-model workflow.

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

#### Engineering-object file formats for the two hierarchy levels

There are two ways to provide engineering objects in hierarchical
unstructured meshing, depending on whether the two meshes should contain
the same objects or different objects.

**Same engineering objects in both meshes.** If `mesh1` and `mesh2` should
use the same wells, sources, shafts, extra planes, ellipses, or
triangulations, use the same engineering-object CSV formats described for
the standard unstructured mesh. No hierarchy-assignment column is needed.
Load the objects in the normal way and pass them to `hierarchical_mesh()`;
the hierarchical workflow uses the supplied objects for both mesh levels
and clips them to the corresponding mesh extent where required.

**Different engineering objects in the two meshes.** If an engineering
object should belong only to one hierarchy level, use the hierarchical
form of the engineering-object input. Add one final column to the normal
engineering-object data to specify the mesh to which the object belongs:

- `1` assigns the object to the outer mesh (`mesh1`).
- `2` assigns the object to the inner mesh (`mesh2`).

The ordinary columns for the object remain unchanged; the hierarchy value
is appended as the last column. For example, the standard well format is
`wellid, x, y, z`. A hierarchical well file adds the mesh assignment as a
fifth value:

``` text
# wellid, x, y, z, mesh
1, 400, 400, 400, 1
1, 400, 400, 500, 1
1, 400, 400, 400, 2
1, 400, 400, 500, 2
```

Here, the first two rows define well `1` for `mesh1`, while the last two
rows define well `1` for `mesh2`. The geometry may therefore be different
at the two hierarchy levels even when the same object ID is reused.

When this hierarchy-assignment column is present, load the file with
`key_hierarchical=True`, for example:

``` python
hierarchical_wells = load_wells_from_csv(
    engineering_objects_dir + "model_1_wells_hierarchical.csv",
    key_hierarchical=True,
)
```

The same principle applies to the other supported engineering-object
loaders: retain the normal unstructured-mesh object columns and append the
hierarchy assignment when different objects are required for `mesh1` and
`mesh2`. If both meshes should receive the same objects, the additional
column and `key_hierarchical=True` are unnecessary.

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
covers only the region specified by `extent2`. If desired, `extent_z` can
replace the Z limits of an inferred structural-model `"whole"` extent with
absolute `[z_min, z_max]` coordinates while retaining the model-derived
X/Y limits. The smaller value of `mesh_size2` gives the inner mesh a finer
target resolution. Both meshes
inherit `curve_mesh_size` and `refinement`, while the final coordinates
of `mesh2` additionally use a 45-degree rotation. Boundary matching is
still performed before this rotation, in the common world frame.

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

For all supported hierarchical mesh types—unstructured (`Un`), structured
(`Str`), and implicit structured (`Im`)—WBGeo first identifies the
boundary nodes of the inner mesh (`mesh2`) from the physical `point_sets`
generated by the corresponding mesher.

The relevant node sets are:

``` python
mesh2.point_sets["left"]
mesh2.point_sets["right"]
mesh2.point_sets["front"]
mesh2.point_sets["back"]
mesh2.point_sets["bottom"]
mesh2.point_sets["top"]
```

Each point set contains the node IDs belonging to one physical boundary
face of `mesh2`. The coordinates of these `mesh2` boundary nodes are then
used as target locations for finding corresponding nodes in the outer
mesh (`mesh1`).

For each boundary node of `mesh2`, WBGeo performs a KD-tree
nearest-neighbour search among the nodes of `mesh1` and selects the
closest `mesh1` node. Therefore, `mesh2` defines the physical boundary
locations, while the final coupling-node IDs are node IDs from `mesh1`.

After the nearest `mesh1` nodes have been identified, the same selected
`mesh1` node IDs are represented in **two coordinate systems**. Their
coordinates are stored once in the final coordinate system of the large
mesh (`big_frame`) and once in the final coordinate system of the small
mesh (`small_frame`). Thus, the two boundary datasets do not represent
two different sets of nodes: they contain the same selected `mesh1` node
IDs, but their `X`, `Y`, and `Z` coordinates are expressed in the two
different mesh coordinate frames.

This two-frame output is used for all three hierarchical mesh types. For
unstructured meshes, `big_frame` and `small_frame` can contain different
coordinates when different output rotations or translations are requested
for the two hierarchy levels. For structured and implicit structured
meshes, separate hierarchy-level output transformations are currently not
applied. Consequently, `big_frame` and `small_frame` normally contain the
same coordinates for those mesh types, while retaining the same common
output structure and the same selected `mesh1` node IDs.


### Boundary output structure

`hierarchical_mesh()` still returns exactly three objects:

``` python
mesh1, mesh2, nodes_bcs = hierarchical_mesh(...)
```

The third output now contains two coordinate frames:

``` text
nodes_bcs = {
    "big_frame": {
        "left": ...,
        "right": ...,
        "front": ...,
        "back": ...,
        "bottom": ...,
        "top": ...,
    },
    "small_frame": {
        "left": ...,
        "right": ...,
        "front": ...,
        "back": ...,
        "bottom": ...,
        "top": ...,
    },
}
```

Each face array has four columns:

``` text
NodeID, X, Y, Z
```

The `NodeID` values in both frames are the **same mesh1 node IDs** selected
by the nearest-neighbour search described above. In other words, after
the closest nodes have been found in `mesh1`, WBGeo reports the
coordinates of those same selected `mesh1` nodes in both mesh coordinate
systems.

-   **`big_frame`** stores those selected `mesh1` nodes using the final
    coordinate system of `mesh1`, including any output transform
    requested for `mesh1`.
-   **`small_frame`** stores the same selected `mesh1` world-coordinate
    points expressed in the final coordinate system of `mesh2`. 

For unstructured meshes, `rotation_angle` centres XY coordinates using
the mesh bounding-box centre and then rotates them; the original Z
coordinates are preserved. For `rotation_angle_translate`, the same centred XY rotation
is used and the mesh centre is placed at `(xt, yt, zt)`. If the target
coordinates are omitted, the translated centre defaults to `(0, 0, 0)`.

------------------------------------------------------------------------

## Export

Hierarchical meshes can be exported in all hierarchical export formats
provided by the package:

-   ABAQUS
-   ANSYS
-   Exodus
-   FEFLOW
-   Gmsh
-   VTK
-   VTM
-   VTU

Each exporter writes both hierarchy meshes and supports the current
two-frame boundary-node structure. 

Example with Exodus:

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

With the current `nodes_bcs` structure, the boundary files are written
into two subdirectories:

``` text
mesh1r.exo
mesh2r.exo
boundary_nodesr/
    big_frame/
        left_boundary_nodes.csv
        right_boundary_nodes.csv
        front_boundary_nodes.csv
        back_boundary_nodes.csv
        bottom_boundary_nodes.csv
        top_boundary_nodes.csv
    small_frame/
        left_boundary_nodes.csv
        right_boundary_nodes.csv
        front_boundary_nodes.csv
        back_boundary_nodes.csv
        bottom_boundary_nodes.csv
        top_boundary_nodes.csv
```


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


The corresponding Workbench inspector is **Inspect Hierarchical Meshes
3D**.

### ParaView visualisation (only in codebase)

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

## Complete hierarchical workflow

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

# 4. Export both meshes and both boundary-node coordinate frames.
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
-   **`nodes_bcs`**: Nested dictionary containing the matched outer-mesh
    node IDs in both the final big-mesh and final small-mesh coordinate
    frames.

The boundary object has the form:

``` text
nodes_bcs
    |
    +----> big_frame
    |         +----> left/right/front/back/bottom/top
    |
    +----> small_frame
              +----> left/right/front/back/bottom/top
```

Both frames use the same `mesh1` node IDs. The difference is only the
coordinate system in which those selected nodes are expressed.


The hierarchy therefore preserves two independently generated mesh
levels while keeping boundary-node selection independent of output
rotation and providing both coordinate representations required to
relate the inner and outer domains.
