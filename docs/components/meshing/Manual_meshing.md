# Meshing

Meshing turns the output of the structural modeling component into a
mesh suitable for numerical simulation. This is non-trivial: geological
models tend to produce complex geometries --- overlapping surfaces,
small gaps, cavities, unconformities --- that make generating a valid,
water-tight mesh difficult. A valid mesh must be topologically
consistent, without gaps or overlaps, to give reliable simulation
results. Preserving geological realism while keeping the mesh
numerically stable and computationally efficient requires careful
treatment of faults, stratigraphic boundaries, and unconformities.

The Workbench addresses this with two complementary approaches ---
**explicit** meshing (structured and unstructured) and **implicit**
meshing --- that turn structural model output into water-tight 3D meshes
for geoscientific simulation. The Workbench focuses on 3D models
throughout, to better capture the Earth's three-dimensional structure,
so all meshing methods produce 3D meshes.

> **Note:** Interfering or embedded objects --- wells, point sources,
> mining shafts, triangulated surfaces, user-defined planes, and faults
> --- are only supported in explicit **unstructured** mesh generation.
> These features are not yet implemented for structured or implicit
> meshing.

------------------------------------------------------------------------

## Concepts

**Explicit meshes** discretise the domain directly into elements whose
geometry conforms to the structural model's surfaces. **Implicit
meshes** instead sample lithology from the structural model's continuous
scalar fields onto a regular grid, without explicitly conforming to the
surfaces.

**Structured meshes** arrange elements in a regular, grid-like pattern
with predictable node connectivity --- in 3D this means hexahedral
elements, which are efficient to index and numerically stable. They are
well suited to relatively simple, laterally continuous geometries, such
as stratified sedimentary sequences, but are less flexible for faults,
folds, or discontinuous layers.

**Unstructured meshes** arrange elements --- tetrahedral, prismatic, or
polyhedral --- in an irregular pattern defined by the geometry and
topology of the model. They conform more easily to complex or irregular
features and support localised refinement, at the cost of higher memory
and processing requirements.

------------------------------------------------------------------------

## Workflow

Steps 1--3 are alternative meshing methods --- pick the one that fits
your model, rather than running them in sequence. Step 4 (export)
applies to the result of whichever method was used.

### 1. Explicit structured mesh

> **Workbench component:** `Create Structured Mesh`

Structured meshes are composed of hexahedral elements arranged on a
regular grid, which provides good numerical stability and accuracy and
allows simpler indexing and faster computation than unstructured
meshing.

> **Note:** Structured mesh generation is designed for geological models
> in which all layers extend continuously across the entire model
> domain. It performs best when the layers are approximately horizontal
> and laterally continuous, covering nearly the full extent of the model
> in the x and y directions. It is **not suitable** for models
> containing faults or discontinuous layers that do not span the entire
> domain.

The inputs are:

-   **`refinement_data`**: A list of integers defining the vertical
    resolution (number of divisions in the z-direction) between
    geological surfaces. The list has length `n + 1`, where `n` is the
    number of geological surfaces: the first value defines the divisions
    between the bottom boundary and the first surface, the last value
    defines the divisions between the top boundary and the last surface,
    and the intermediate values define the divisions between consecutive
    surfaces. Each integer produces `m = divisions + 1` elements between
    the corresponding pair of surfaces.
-   **`z_threshold`** *(optional)*: A small float used to slightly
    adjust point positions in the vertical direction when points are
    very close to one another, preventing surface overlap and ensuring
    proper mesh separation.
-   **`mesh_division`** *(optional)*: A list of two values defining the
    mesh resolution in the x and y directions. The generated mesh has
    one fewer element than each specified value --- for example,
    `mesh_division = [5, 8]` creates a mesh with 4 elements in x and 7
    elements in y.
-   **`tolerance`** *(optional)*: A float threshold defining the minimum
    vertical separation between points. If two points are closer than
    this value in z, one of them is shifted vertically by `z_threshold`
    to prevent overlap.
-   **`extent`** *(optional)*: The model bounding box
    `(min_x, max_x, min_y, max_y, min_z, max_z)`. If not defined, the
    extent of the structural model given in `geomodel_result` is used.

In the codebase, an additional argument is required as the first input:

-   **`geomodel_result`**: Output of the structural model, containing
    geological surfaces and their vertices, faults, normal vectors,
    model extent, and grid points with material identifiers for each
    geological layer.

The output is a structured 3D hexahedral mesh with consistent node
ordering, producing a valid, right-handed (counter-clockwise) hexahedral
orientation.

Example call in the codebase:

``` python
mesh_explicit_structured = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[10, 10, 10],
    mesh_division=[12, 12],
    tolerance=0.5,
    extent=(0, 1000, 0, 1000, 0, 950),
)
```

Here, `structural_model_result` is the output of a previously generated
structural model. `refinement_data=[10, 10, 10]` subdivides each
interval between consecutive surfaces into 10 vertical layers.
`mesh_division=[12, 12]` sets the mesh resolution to 12 divisions in x
and y. `tolerance=0.5` sets the minimum allowed vertical separation
between neighbouring points before adjustment, and `extent` defines the
spatial limits of the mesh.

### 2. Explicit unstructured mesh

> **Workbench component:** `Create Unstructured Mesh`

Unstructured meshes are composed of tetrahedral elements arranged
irregularly, generated using the open-source package **Gmsh**. They
conform more easily to complex geological features --- faults, folds,
cavities, erosional surfaces --- and support local refinement where
higher resolution is needed, at the cost of higher memory and processing
requirements than structured meshing.

This is the only meshing method that supports embedding engineering
objects and faults.

#### Engineering objects

All engineering objects below are optional, loaded from CSV files, and
passed into `Create Unstructured Mesh` alongside the structural model
output.

**`wells`** --- well trajectories, each a straight line or a polyline (a
sequence of points) between the top and bottom of the well.

> **Workbench component:** `Load Well`

Each row of the CSV file corresponds to a single point on a well and has
four columns: `id`, `x`, `y`, `z`. Rows sharing an `id` belong to the
same well; each well needs at least two points (top and bottom), with
additional intermediate points allowed for piecewise-linear
trajectories. Lines starting with `#` are ignored.

A CSV file defining two wells --- one a single straight segment, the
other a polyline of two connected segments --- looks like:

    # id, x, y, z
    1, 100, 200, 400
    1, 100, 200, 600
    2, 300, 500, 300
    2, 300, 500, 400
    2, 400, 500, 400

This defines well 1 as `(100, 200, 400) → (100, 200, 600)`, and well 2
as `(300, 500, 300) → (300, 500, 400) → (400, 500, 400)`.

``` python
wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_wells.csv")
```

**`sources`** --- point sources, each a single `(x, y, z)` point.

> **Workbench component:** `Load Source`

Each row has four columns: `id`, `x`, `y`, `z`. Each `id` must appear
exactly once --- unlike wells, sources cannot span multiple rows. Lines
starting with `#` are ignored.

    # id, x, y, z
    1, 100, 200, 400
    2, 300, 500, 300

``` python
sources = load_sources_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_sources.csv")
```

**`shafts`** --- mining shafts, each a cylinder with a center
`(x, y, z)`, an axis direction `(dx, dy, dz)`, and a radius.

> **Workbench component:** `Load Shaft`

Each row has eight columns: `id, cx, cy, cz, ax, ay, az, radius`, where
`cx/cy/cz` is the center, `ax/ay/az` is the axis direction vector, and
`radius` is the shaft radius. Each `id` must appear exactly once. Lines
starting with `#` are ignored.

    # id, cx, cy, cz, ax, ay, az, radius
    1, 100, 200, 300, 0, 0, 1, 50
    2, 400, 500, 600, 1, 0, 0, 75

This defines shaft 1 with center `(100, 200, 300)`, axis `(0, 0, 1)`,
radius 50, and shaft 2 with center `(400, 500, 600)`, axis `(1, 0, 0)`,
radius 75.

``` python
shafts = load_shafts_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_shafts.csv")
```

**`extra_planes`** --- additional planes, each defined by the
coordinates of its four corners (12 values). Useful for adding missing
geological layers or faults. Only in-plane extra planes can be generated
in the Workbench; for out-of-plane surfaces, use `triangulations`
instead.

> **Workbench component:** `Load Plane`

Each row defines a single point belonging to a plane: `id, x, y, z`.
Rows sharing an `id` belong to the same plane, and each plane needs at
least four points. Lines starting with `#` are ignored.

    # id, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4
    1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0
    2, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1

``` python
planes = load_planes_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_planes.csv")
```

**`ellipses`** --- elliptical surfaces, each a dictionary with keys:

-   `center`: `(x, y, z)` coordinates of the ellipse center
-   `radii`: `(r1, r2)` semi-major and semi-minor axes
-   `angle1` *(optional)*: start angle in radians (default `0.0`)
-   `angle2` *(optional)*: end angle in radians (default `2π`)
-   `zAxis` *(optional)*: normal vector of the ellipse plane (default
    `[0, 0, 1]`)
-   `xAxis` *(optional)*: x-axis vector of the ellipse plane (default
    `[1, 0, 0]`)

> **Workbench component:** `Load Ellipse`

Each row defines one ellipse. The first column is an ID, followed by the
required fields `cx, cy, cz, r1, r2` and the optional fields
`angle1, angle2, zx, zy, zz, xx, xy, xz`. Lines starting with `#` are
ignored.

    # id, cx, cy, cz, r1, r2, angle1, angle2, zx, zy, zz, xx, xy, xz
    1, 100, 200, 0, 50, 30
    2, 400, 500, 10, 75, 50, 0.0, 3.14159, 0, 0, 1, 1, 0, 0
    3, 0, 0, 0, 25, 10, , , 0, 0, 1, 1, 0, 0

``` python
ellipses = load_ellipses_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_ellipses.csv")
```

**`triangulations`** --- additional 3D points `[(x, y, z), ...]` used to
generate a Delaunay triangulation for a surface.

> **Workbench component:** `Load Triangulations Planes`

Each row defines a single 3D point with three columns: `x, y, z`. Lines
starting with `#`, and empty lines, are ignored.

    # x, y, z
    0.0, 0.0, 0.0
    1.0, 0.0, 0.0
    0.0, 1.0, 0.0

``` python
triangulations = load_triangulations_planes_from_csv(cwd + "/examples/synthetic_examples/model2/input_data/engineering_objects/model_2_triangulations.csv")
```

#### Mesh generation parameters

-   **`tolerance`**: The maximum allowable deviation (in model
    coordinate units) when adjusting surface boundaries to match the
    model extent. During interpolation, surface coordinates can shift
    slightly due to numerical precision or grid discretisation, so the
    outermost points may no longer perfectly coincide with the model
    boundary; `tolerance` allows small deviations up to this limit to be
    corrected so the surfaces align with the extent.

-   **`mesh_size`**: The target element size assigned to all geometry
    points (the background mesh), giving a uniform meshing resolution
    throughout the domain. Where refinement is active, the Workbench
    automatically uses the minimum size among all active refinement
    contributions in the refined regions.

-   **`curve_mesh_size`**: The target number of mesh elements per 2π
    radians (360°) of curvature. Gmsh reduces the element size in curved
    regions so that approximately `curve_mesh_size` elements are used to
    discretise a full circle --- a larger value gives finer meshing
    along curved regions and coarser meshing where the geometry is
    nearly flat.

-   **`DISTANCE_THRESHOLD`**: A distance threshold used to identify and
    remove geological surface points that lie close to fault surfaces.
    For each geological surface, points within `DISTANCE_THRESHOLD` of
    any fault surface are treated as overlapping the fault and are
    removed. This prevents overlaps and keeps faults cleanly separated
    from surrounding geological units --- the geological model's output
    typically contains overlapping or duplicated points near fault
    surfaces. A small value may leave overlapping regions incompletely
    removed; a large value can remove valid geometry near faults.

-   **`PROJECTION_THRESHOLD`**: The search radius around a fault surface
    used to find nearby layer points for extrusion. After points near
    faults are removed (see `DISTANCE_THRESHOLD`), gaps appear between
    the geological surfaces and the fault planes; nearby points are
    extruded --- perpendicular to their surface normals --- until they
    intersect the fault plane, closing these gaps.
    `PROJECTION_THRESHOLD` should generally be larger than
    `DISTANCE_THRESHOLD`, but not so large that points with different
    curvature far from the fault get selected, which can distort the
    geometry near the fault.

-   **`EXTRUSION_FACTOR`**: How far each point found within
    `PROJECTION_THRESHOLD` is pushed along the direction perpendicular
    to its surface normal, in model units.

-   **`z_threshold`**: The vertical tolerance used to identify surfaces
    that are effectively flat across a fault (i.e. the fault produces no
    meaningful vertical offset there). If a surface's total vertical
    variation is smaller than `z_threshold`, the fault is treated as
    non-displacing for that surface, and no overlap removal or extrusion
    is applied near it.

-   **`extent`** *(optional)*: Defines the spatial domain of the
    unstructured mesh. Two representations are supported:

    1.  **Axis-aligned extent** --- provide the six values
        `(xmin, xmax, ymin, ymax, zmin, zmax)`. This creates a
        box-shaped domain whose boundaries are parallel to the global
        coordinate axes.

    2.  **Eight-corner extent** --- provide the `(x, y, z)` coordinates
        of the eight corners of the box. This allows the meshing domain
        itself to be defined by a box whose boundaries are not
        necessarily parallel to the global coordinate axes.

    If `extent` is not supplied, the extent from `geomodel_result` is
    used.

-   **`smooth`**: The smoothing parameter for the Radial Basis Function
    (RBF) interpolation used to rebuild surfaces. Larger values produce
    smoother surfaces by reducing small-scale noise; smaller values
    preserve more detail from the original surface geometry.

-   **`gmsh_flag`** (optional): If `True`, saves the original
    Gmsh-generated mesh as `mesh.msh`. Default is `False`.

-   **`mapping_litho`** *(optional)*: Controls how lithological units
    are assigned to the raw Gmsh blocks. The supported modes are:

    -   `"automatic_centers"` *(default)* --- assigns lithology by
        voting on raw-block cell centroids. This is generally the more
        reliable automatic option near fault planes because centroids
        are interior points.
    -   `"automatic_corners"` --- assigns lithology using raw-block node
        coordinates. This is retained as an alternative and for
        backwards comparison.
    -   `"none"` --- skips lithological mapping so the fragmented Gmsh
        blocks can be inspected directly.
    -   `"manual"` --- merges blocks according to the user-supplied
        `merge_file`.

    When automatic mapping is ambiguous, for example for very thin
    layers or coarse meshes, `"none"` can be used to inspect the
    fragmented blocks before defining an explicit `"manual"` mapping.

-   **`merge_file`** *(required only when `mapping_litho="manual"`)*:
    Path to a file listing how mesh blocks should be merged into
    physical groups representing geological units. Every mesh block must
    be listed, including blocks that are not merged with any other ---
    one physical group per line, with block IDs on the same line merged
    together:

```{=html}
<!-- -->
```
    # block ids to be merged
    # all blocks must be listed here, even if not merged with another

    27, 29, 31
    21, 26
    20, 24, 25, 30, 18, 17
    22, 23, 28, 19, 16, 13, 11
    12, 14, 15
    6, 7, 8, 9, 10
    1
    2
    3
    4
    5

Here, blocks 1--5 are each listed alone and so are assigned their own
physical group, while blocks 27, 29, and 31 are listed together and
merged into a single physical group. - **`refinement`** *(optional)*:
Local mesh-refinement settings. Refinement can be applied around wells,
sources, shafts, triangulated surfaces, ellipses, extra planes, faults,
and user-defined general points or line segments. See [Mesh
refinement](#mesh-refinement) below.

-   **`rotation_angle`** *(optional)*: A parameter that results in rotation of the generated
    unstructured mesh about the z-axis. The angle is specified in
    degrees. Before rotation, the x-y coordinates are translated so that
    the centre of the generated mesh is at the origin; the mesh is then
    rotated by `rotation_angle`, while z coordinates remain unchanged.
    This parameter is useful when the final mesh coordinate system
    should be aligned with a preferred direction, for example principal
    stress directions. 

In the codebase, an additional argument is required as the first input:

-   **`geomodel_result`**: Output of the structural model, containing
    geological surfaces and their vertices, faults, normal vectors,
    model extent, and grid points with material identifiers for each
    geological layer.

Example call in the codebase:

``` python
mesh_unstructured_with_objects = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    wells=wells,
    sources=sources,
    shafts=shafts,
    extra_planes=planes,
    triangulations=triangulations,
    tolerance=50,
    mesh_size=30,
    curve_mesh_size=5,
    DISTANCE_THRESHOLD=40,
    PROJECTION_THRESHOLD=60,
    EXTRUSION_FACTOR=80,
    z_threshold=10,
    gmsh_flag=True,
    mapping_litho="automatic_centers",
    refinement=refinement,
    rotation_angle=30.0,
)
```

This generates an unstructured tetrahedral mesh from
`structural_model_result` and includes the supplied engineering and
additional geometric objects. `mesh_size` and `curve_mesh_size` define
the background and curvature-based resolution. `DISTANCE_THRESHOLD`,
`PROJECTION_THRESHOLD`, and `EXTRUSION_FACTOR` control geometric
processing near faults, while `z_threshold` identifies surfaces for
which a fault produces no meaningful vertical offset. `gmsh_flag=True`
saves the original Gmsh `.msh` file, `mapping_litho="automatic_centers"`
applies centroid-based automatic lithological mapping, and `refinement`
applies the local refinement definitions. `rotation_angle=30.0` is an
independent input parameter of the unstructured mesh generator. It
recentres the x-y coordinates on the mesh centre and rotates the
generated mesh by 30° about the z-axis.

#### Mesh refinement

The Workbench provides local control of mesh resolution around
geological, engineering, and user-defined features. A single
`Refinement` object can combine several refinement definitions. Where
multiple refinements overlap, the minimum local mesh size
controls the generated mesh.

The current `Refinement` object supports:

-   **`wells`** --- refinement around well trajectories.
-   **`sources`** --- refinement around point sources.
-   **`shafts`** --- refinement according to perpendicular distance from
    mine-shaft axes.
-   **`triangulations`** --- refinement around triangulated surfaces.
-   **`ellipses`** --- refinement around elliptical surfaces.
-   **`extraplanes`** --- refinement around the additional planes
    supplied through `extra_planes`.
-   **`faults`** --- refinement around fault surfaces.
-   **`general`** --- refinement around one  user-defined point or a finite line defined by two endpoints

##### Linear refinement around wells, sources, and shafts

Wells, sources, and shafts support a linear distance-based formulation
using `SizeMin`, `SizeMax`, `DistMin`, and `DistMax`:

-   **`SizeMin`** --- minimum target mesh size close to the feature.
-   **`SizeMax`** --- target mesh size away from the feature.
-   **`DistMin`** --- distance up to which `SizeMin` is applied.
-   **`DistMax`** --- distance at which the refinement reaches
    `SizeMax`.

Between `DistMin` and `DistMax`, the requested mesh size increases from
`SizeMin` to `SizeMax`.

``` python
refinement = Refinement()

refinement.wells = LinearWellRefinement(
    SizeMin=5.0,
    SizeMax=90.0,
    DistMin=30.0,
    DistMax=100.0,
)

refinement.sources = LinearSourceRefinement(
    SizeMin=5.0,
    SizeMax=90.0,
    DistMin=30.0,
    DistMax=100.0,
)

refinement.shafts = LinearShaftRefinement(
    SizeMin=8.0,
    SizeMax=90.0,
    DistMin=25.0,
    DistMax=100.0,
)
```

For shafts, the distance is measured from the shaft axis.

Wells, sources, and shafts also support function-based refinement. In
this case, a Gmsh mathematical expression defines the requested mesh
size as a function of `DIST`, the distance from the relevant feature.

``` python
refinement.wells = FunctionWellRefinement(
    expression="5 + 75 * (1 - exp(-DIST / 80))"
)
```

##### Refinement around surfaces

Refinement around Triangulations, ellipses, faults, and extra planes can be done only using a linear refinement with the same
parameter names as the other linear refinement definitions: `SizeMin`,
`SizeMax`, `DistMin`, and `DistMax`. 

``` python
refinement = Refinement()

refinement.triangulations = TriangulationRefinement(
    SizeMin=8.0,
    SizeMax=90.0,
    DistMin=50.0,
    DistMax=100.0,
)

refinement.faults = FaultRefinement(
    SizeMin=5.0,
    SizeMax=90.0,
    DistMin=30.0,
    DistMax=100.0,
)

refinement.extraplanes = ExtraPlaneRefinement(
    SizeMin=8.0,
    SizeMax=90.0,
    DistMin=40.0,
    DistMax=100.0,
)

refinement.ellipses = ExtraPlaneRefinement(
    SizeMin=8.0,
    SizeMax=90.0,
    DistMin=40.0,
    DistMax=100.0,
)
```


##### General point and line refinement

`GeneralRefinement` provides refinement that is independent of the
predefined geological and engineering objects. It can be used for a
single point or for a finite line segment.

For a point, provide one coordinate:

``` python
refinement.general = GeneralRefinement(
    coordinates=[(500.0, 500.0, 400.0)],
    SizeMin=5.0,
    DistMax=100.0,
)
```

For a line segment, provide its two endpoints:

``` python
refinement.general = GeneralRefinement(
    coordinates=[
        (100.0, 200.0, 300.0),
        (900.0, 200.0, 300.0),
    ],
    SizeMin=5.0,
    DistMax=100.0,
)
```

For linear general refinement, `SizeMin` is applied at the point or line
and the mesh size increases toward the background `mesh_size` at
`DistMax`. Alternatively, `expression` can be supplied and `DIST` used
in the expression. The `general` field can also receive a list of
`GeneralRefinement` objects when several independent point or line
refinements are required.

Multiple refinement types can be combined:

``` python
refinement = Refinement()

refinement.wells = LinearWellRefinement(
    SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.shafts = LinearShaftRefinement(
    SizeMin=8.0, SizeMax=90.0, DistMin=25.0, DistMax=100.0
)
refinement.faults = FaultRefinement(
    SizeMin=5.0, SizeMax=90.0, DistMin=30.0, DistMax=100.0
)
refinement.triangulations = TriangulationRefinement(
    SizeMin=8.0, SizeMax=90.0, DistMin=50.0, DistMax=100.0
)
refinement.extraplanes = ExtraPlaneRefinement(
    SizeMin=8.0, SizeMax=90.0, DistMin=40.0, DistMax=100.0
)
refinement.general = GeneralRefinement(
    coordinates=[(500.0, 500.0, 400.0)],
    SizeMin=5.0,
    DistMax=100.0,
)
```

### 3. Implicit structured mesh

> **Workbench component:** `Create Structured Implicit Mesh`

Implicit structured meshes are composed of hexahedral elements arranged
on a regular, logically ordered grid inherited directly from the
structural model's grid --- lithology and material properties are
sampled from the model's continuous scalar fields rather than built from
explicit surfaces. This makes implicit meshes less flexible for highly
irregular geometries than unstructured meshes, but simple, robust, and
cheap to compute --- well suited to large-scale domains, sensitivity
studies, and workflows that require many repeated model evaluations.

The input is:

-   **`extent`** *(optional)*: The meshing domain. It can be given
    either as the legacy six-value axis-aligned extent
    `[xmin, xmax, ymin, ymax, zmin, zmax]`, or as eight ordered
    `(x, y, z)` corners defining an oriented parallelepiped. The
    oriented form is useful when the model domain itself is not aligned
    with the global coordinate axes. If `extent` is omitted, the
    structural-model extent is used.

In the codebase, an additional argument is required as the first input:

-   **`geomodel_result`**: Output of the structural model, containing
    geological surfaces and their vertices, faults, normal vectors,
    model extent, and grid points with material identifiers for each
    geological layer.

The output is a structured 3D hexahedral mesh.

Example call in the codebase:

``` python
mesh_implicit_structured = create_implicit_structured_mesh(
    geomodel_result=structural_model_result,
    extent=(0, 1000, 0, 1000, 0, 950),
)
```

### 4. Export the mesh

> **Workbench component:** `Download Mesh`

The Workbench provides a unified export interface for meshes generated
by any of the three methods above. Meshes can be exported to **Exodus**,
**Gmsh**, **Abaqus**, **Ansys**, **FeFlow**, **STL**, **VTK**, **VTU**,
and **VTM** formats --- in the visual interface (visual DSL),
`Download Mesh` exposes all of these through a single format selector.

> **Note:** The Gmsh, STL, and FeFlow exports currently support
> unstructured meshes only; they are not yet implemented for structured
> mesh types.

In the codebase, each format has its own export function, e.g. for Gmsh:

``` python
buf = export_mesh_results_to_gmsh(mesh_unstructured_with_objects)
with open("filename_example_mesh.msh", "wb") as f:
    f.write(buf.getvalue())
```

`export_mesh_results_to_gmsh()` takes a mesh object and returns a binary
buffer, which is then written to disk. Exporting to other formats
follows the same pattern, only the function changes --- for example:

``` python
export_mesh_results_to_vtk(...)
export_mesh_results_to_vtu(...)
```

The Exodus export additionally requires the mesh type:

``` python
buf = export_mesh_results_to_exodus(mesh_unstructured_with_objects, type="unstr")
```

Here, `type` is `"unstr"`, `"str"`, or `"imp"` depending on which of the
three meshing methods produced the mesh, ensuring correct formatting
during export.

------------------------------------------------------------------------

## Visualisation and Inspection

> **Workbench:** Where available, these are exposed as inspector
> functions with limited functionality in the visual interface (visual
> DSL) --- parameters are fixed; the codebase functions below expose the
> full set of options.

> **Workbench component:** `Plot Mesh in 3D`

`plot_mesh_3d()` gives a quick 3D preview of a generated mesh, colored
by lithology using the same element colors as the structural model.

-   **`mesh_results`**: The `MeshResults` object to plot.
-   **`geomodel_result`**: The structural model result that provides
    element labels and colors.
-   **`style`** *(optional)*: PyVista mesh style --- `"surface"`,
    `"wireframe"`, or `"points"`. Default `"surface"`.
-   **`show_plotter`** *(optional)*: If `True`, displays the interactive
    plot window. Default `True`.

Example call in the codebase:

``` python
plot_mesh_3d(mesh_unstructured_with_objects, structural_model_result, style="surface")
```

------------------------------------------------------------------------

## Output

> **Workbench:** The `Mesh` output port of `Create Structured Mesh`,
> `Create Unstructured Mesh`, or `Create Structured Implicit Mesh`
> carries the result. It can be connected directly to `Download Mesh`
> for export, or inspected in-place using `Plot Mesh in 3D`.

Each meshing method returns a `MeshResults` object:

-   **`nodes`** --- node coordinates, shaped `(N, 3)`.
-   **`elements`** --- mesh connectivity, stored as a list of MeshIO
    `CellBlock`s.
-   **`cell_data`** *(optional)* --- per-cell data arrays aligned with
    `elements`, such as material/lithology IDs.
-   **`point_sets`** *(optional)* --- per-node data arrays.
-   **`mesh`** --- a PyVista `MultiBlock` view of the same data, built
    on first access. Block 0 is the basement (lithology 0); subsequent
    blocks follow oldest to youngest.

`MeshResults` is the direct input to the mesh export components
described above, and to `Plot Mesh in 3D` for quick in-place inspection.
