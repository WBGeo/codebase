# Structural Modeling

The structural modeling component reconstructs 3D geological structure from sparse field observations. Structural modeling in general can be approached in many ways; this Workbench uses **implicit methods**, where the geometry is represented as the isosurfaces of a continuously interpolated scalar field rather than as explicit surfaces built by hand. The scalar field is fitted to surface contact points and (optionally) orientation measurements, and the result is a volumetric lithology model together with surface meshes that represent the boundaries between lithological units.

---

## Concepts

**Surface points** are locations where a geological interface has been observed — e.g. a formation contact in a borehole, an outcrop pick, or a digitised section. Each point carries an X, Y, Z coordinate and the name of the formation it belongs to. In the implicit framework used here, surface points define the **base** of the named geological element: the scalar field is fitted so that the isovalue for a given formation passes through its surface points from below.

**Orientations** are dip/azimuth measurements at known locations, represented as a gradient vector (G_x, G_y, G_z). They are optional, but some interpolation methods require them to produce a result while others do not consider them at all — see the interpolation methods table for details.

**Stratigraphic groups** organise formations into an unconformity-bounded sequence. Each group is interpolated independently, and groups are ordered youngest to oldest. Within a group, formations are also listed youngest to oldest — this ordering is used when assembling the final lithology block.

**The mapping** is a dictionary that assigns formations to groups:

```json
{
  "Strat_Series2": ["rock3", "rock2"],
  "Strat_Series1": ["rock1", "basement"]
}
```

The key is the group name; the list contains the formations in that group, youngest first.

**The grid** is where the scalar fields are evaluated and the lithology is assigned. Two grids are available and can be used interchangeably for the fault and the structural model:

- A **Regular Grid** divides the domain into voxels of equal size. Surfaces are extracted with marching cubes.
- An **Octree Grid** starts from coarse blocks and refines them only where a surface or fault passes through, so the cells are fine near contacts and faults and coarse elsewhere. Surfaces are extracted with dual contouring. At its finest level, an Octree Grid gives the same model as a Regular Grid of the same resolution, with fewer cells.

All interpolation methods, fault modeling, unconformities and surface mesh types work on both grids.

---

## Workflow

### 1. Define the grid

> **Workbench components:** `Regular Grid` or `Octree Grid`

Either grid can be connected to `Compute Fault Model` and `Compute Structural Model`. When a fault model is used, the fault model and the structural model must be computed on the same grid.

#### Regular Grid

A `RegularGrid` defines the 3D computation domain as a regular axis-aligned voxel grid:

- **Extent** — bounding box `(xmin, xmax, ymin, ymax, zmin, zmax)` in world coordinates. Should cover all input data with a small margin.
- **Resolution** — number of voxels `(nx, ny, nz)`. Higher resolution gives more geometric detail but increases computation time significantly.

#### Octree Grid

An `OctreeGrid` defines the same kind of domain, but with cells of different sizes. The extent is first divided into base blocks, the coarsest cells of the grid. During modeling, a block is halved along every axis wherever a surface or fault passes through it, and again inside the halves, up to the maximum level. Cells that lie entirely within one formation stay coarse.

- **Extent** — bounding box `(xmin, xmax, ymin, ymax, zmin, zmax)`, as for the Regular Grid.
- **Base resolution** — number of base blocks `(bx, by, bz)`. Choose them so that the blocks are close to cubic, e.g. `(10, 4, 4)` for an extent of 2500 × 1000 × 1000.
- **Max level** — how many times a base block can be halved. The finest cells have the size of a base block divided by `2**max_level`, so the finest resolution is `base_resolution * 2**max_level` (e.g. `(10, 4, 4)` at level 4 gives `160 × 64 × 64`). Level `0` gives a plain regular grid of base blocks.
- **Gap mitigation** *(optional)* — where a surface crosses cells of different sizes, dual contouring can leave small gaps. `balance_mesh_depth_adjacency` (default) refines the cells next to the finest ones as well, which closes these gaps at the cost of extra cells. `none` skips this step.
- **Cell balance** *(optional)* — `none` (default) lets a finest cell sit next to a much coarser one. `face_2to1` adds cells so that cells sharing a face differ by at most one level, for a more gradual transition between fine and coarse cells.

```python
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid

grid = OctreeGrid(extent=(0, 2500, 0, 1000, 0, 1000), base_resolution=(10, 4, 4), max_level=4)
```

The rest of the workflow is the same as with a Regular Grid.

### 2. Load input data

> **Workbench component:** `Structural Input Data`

Surface points are required. Orientations are optional — whether they are needed depends on the interpolation method chosen later (see the interpolation methods table). The mapping described above is provided as a Python dictionary or configured directly in the visual interface (visual DSL). Together they form the `InputData_StructuralElements` object that feeds the model.

The required columns are:
- Surface points: `X`, `Y`, `Z`, `formation`
- Orientations (if provided): `X`, `Y`, `Z`, `G_x`, `G_y`, `G_z`, `formation`

### 3. (Optional) Build the fault model

> **Workbench components:** `Fault Input Data` → `Compute Fault Model`

If the geological model includes faults, they must be modelled before the stratigraphic interpolation. The fault workflow is:

1. Load fault surface points and orientations into `InputData_FaultElements`.
2. Call `build_fault_frame` to create the `FaultFrame`.
3. Call `compute_fault_domains` to interpolate each fault scalar field and partition the grid into fault domains. This step also extracts fault surface meshes.

Fault interpolation currently uses Universal Co-Kriging exclusively, so **both surface points and orientations are required for every fault**. Orientation vectors are scaled to unit length first, so only their direction matters. The result is a `FaultModelResults` object containing:

- A **surface mesh** for each fault (the boundary between hanging wall and footwall).
- A **domain map** that partitions every cell of the grid (voxel or octree cell) into strictly separate, non-overlapping fault blocks. Stratigraphic groups that are active for a given fault will be interpolated independently in each of its domains; groups that post-date the fault are not affected by the domain split. The domain map is only computed if no faults cross-cut (see limitations below).

**Limitations of the fault model:**

- **Cross-cutting faults are not supported for structural modeling.** Each pair of faults is tested for isovalue-band overlap after interpolation. If two fault surfaces intersect within the model domain, the fault model is still computed and can be inspected (the 3D plot highlights the line where the faults cross), but the result is flagged `cross_cutting=True`, has no domain map, and is rejected by `Compute Structural Model` / `build_structural_frame` with an error. Fix the fault geometry and recompute the fault model.
- **Every fault must split the grid.** A fault whose scalar field does not actually separate the domain into two non-empty sides will raise an error. Faults that are entirely outside the grid extent, or whose surface does not reach across the domain, cannot be modelled.

### 4. Build the structural model

> **Workbench component:** `Compute Structural Model`

This step combines configuration and computation. The input data, grid, and — optionally — the fault model are assembled into a `StructuralFrame`. The following configuration options are available before running the computation:

- **Interpolation method per group** (optional) — defaults to Radial Basis Function. Set per group with `frame["GroupName"].set_interpolation_method("Universal Co-Kriging")`.
- **Interpolation parameters** (optional) — default values are estimated from the input data. Override with `frame["GroupName"].configure_interpolation_params(nelements=20000)`.
- **Fault activity** (only relevant when a fault model is attached) — controls which stratigraphic groups are offset by each fault. `frame.set_fault_activity_by_group(fault_name="Fault-1", group_name="Strat_Series2")` marks the specified group as the youngest one affected. Groups older than or equal to it are interpolated separately in each fault domain; groups younger than it are treated as post-fault and interpolated across the full grid.

The computation runs the following steps in sequence:

1. **Validation** — checks that every active fault domain contains sufficient input data for the chosen interpolation method in each group (for example, that orientation data is present in each domain for methods that require it). All issues are reported together before any expensive computation begins.
2. **Interpolation** — each stratigraphic group is interpolated independently. When faults are active for a group, the interpolation is run completely separately in each fault domain using only the data points that fall within that domain. The resulting scalar fields are then stitched together.
3. **Age masks** — a boolean mask is computed per group that represents its depositional extent: the oldest group underlies the entire domain (mask is True everywhere), while younger groups are only present where they were deposited. When assembling the lithology block, younger groups take precedence over older ones wherever their mask is True — this is what produces unconformable contacts between groups.
4. **Lithology block** — the scalar fields and age masks are combined into a single integer array where each cell is assigned to a formation.
5. **Surface meshes** — isosurfaces are extracted from the scalar fields using marching cubes on a Regular Grid, or dual contouring on an Octree Grid. Five mesh types are stored per element: `masked` (respects unconformities and fault domains), `unmasked` (full isosurface through the entire grid), `combined` (derived from the lithology block), `extended` (extrapolated beyond the grid boundary, used as input for 3D mesh generation), and `extended_masked` (extrapolated beyond the grid boundary like `extended`, but one separate piece per fault domain, cut at the faults and at the base of younger groups; see [Output](#output)).

**Limitations of the structural model:**

- **Group interactions are limited to unconformities.** The only geometric relationship supported between stratigraphic groups is that a younger group erodes the top of an older one. There is no support for intrusions, cross-cutting bodies, or any other non-erosional contact geometry between groups.
- **Sufficient data required in every active fault domain.** When a fault is active for a group, the interpolation in each fault domain uses only the data points that fall within that domain. If a domain contains too few surface points — or none at all — the result in that domain will be unreliable or empty. Methods that additionally require orientations (UCK, FDI, PLI, GeoINR) need orientation data in every active domain as well.

---

## Interpolation Methods

All methods produce the same output format (scalar field + per-element isovalues) and can be set per stratigraphic group. The exposed parameters listed below are a subset of the full configuration space — each underlying library (SciPy, PyKrige, GemPy, LoopStructural, PyTorch) has many additional options that are not yet surfaced.

| Method | Orientations required | Library                                                          | Currently adjustable parameters |
|---|---|------------------------------------------------------------------|---|
| **Radial Basis Function (RBF)** | No | [SciPy](https://scipy.org/)                                      | `kernel`, `smoothing`, `epsilon`, `neighbors` |
| **Ordinary Kriging (OK)** | No | [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/) | `variogram_model`, `range`, `sill`, `nugget`, `anisotropy_scaling_x/y/z`, `neighbors` |
| **Universal Kriging (UK)** | No | [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/) | Same as OK plus `drift_terms` |
| **Universal Co-Kriging (UCK)** | Yes | [GemPy](https://www.gempy.org/)                                  | None currently exposed |
| **Finite Differences (FDI)** | Yes | [LoopStructural](https://loop3d.github.io/LoopStructural/)       | `nelements`, `solver`, `damp`, `tol` |
| **Piecewise Linear (PLI)** | Yes | [LoopStructural](https://loop3d.github.io/LoopStructural/)       | `nelements`, `solver`, `damp`, `tol` |
| **GeoINR** | Yes | [GeoINR](https://github.com/MichaelHillier/GeoINR) (custom)                               | `hidden_dim`, `n_hidden_layers`, `epochs`, `lr`, `alpha`, `beta` |

For UCK and GeoINR, orientation vectors are scaled to unit length, so only their direction affects the result: UCK normalises them before passing them to GemPy, and GeoINR normalises them in its orientation loss.

Three of these parameters take one of a fixed set of options. An unknown option is rejected when it is set.

| Parameter | Methods | Options (default first) |
|---|---|---|
| `kernel` | RBF | `thin_plate_spline`, `linear`, `cubic`, `quintic`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic`, `gaussian` |
| `variogram_model` | OK, UK | `gaussian`, `spherical`, `exponential`, `hole-effect` |
| `solver` | FDI, PLI | `cg`, `lsmr`, `admm` |

The RBF shape parameter `epsilon` is only used by the kernels `multiquadric`, `inverse_multiquadric`, `inverse_quadratic` and `gaussian`; the other kernels ignore it. It is an inverse length (larger values make the kernel more local) and always has a default, so these kernels can be selected without setting it.

Default parameters are estimated automatically from the spatial distribution of the input points (data scale, point count, nearest-neighbour distance) and can be overridden via `configure_interpolation_params`. Parameter choice has a large influence on the resulting model — the default values are a reasonable starting point but should be treated as a first guess rather than an optimal configuration. In particular, variogram range and sill for kriging methods, `nelements` for FDI and PLI, and the number of epochs and network capacity for GeoINR can all significantly change the shape and smoothness of the interpolated surfaces.

---

## Visualisation and Inspection

> **Workbench:** Where available, these are exposed as inspector functions with limited functionality in the visual interface (visual DSL) — parameters are fixed; the codebase functions below expose the full set of options.

The following components allow quick inspection of input data and model results without leaving the Workbench.

### Preview raw input data

> **Workbench components:** `Plot Input Data 3D`, `Plot Fault Input Data 3D`

Before running the model, surface points and orientations can be previewed directly — no `StructuralFrame` required.

- **`plot_input_data_3D(input_data)`** — 3D plot of structural surface points and orientations, colored by formation.
    - **`input_data`**: An `InputData_StructuralElements` object.
    - **`show_orientations`** *(optional)*: If `True`, draws orientation arrows. Default `True`.
    - **`arrow_scale`** *(optional)*: Length of the orientation arrows. Defaults to 5% of the data's bounding box diagonal.
    - **`point_size`** *(optional)*: Size of the surface point markers. Default `8`.
    - **`notebook`** *(optional)*: Use PyVista's notebook rendering mode. Default `False`.
    - **`show`** *(optional)*: If `True`, opens the interactive window immediately. Default `True`.
- **`plot_fault_input_data_3D(input_data)`** — the same, for fault input data (`InputData_FaultElements`), colored by fault name instead of formation. Same parameters as above.

```python
plot_input_data_3D(data_structural_elements)
plot_fault_input_data_3D(data_faults)
```

### Inspect the structural model

> **Workbench components:** `Detailed Report`, `Plot Model Result 2D`, `Plot Model Result 3D`

These act on a `StructuralModelResults` object (see [Output](#output) below).

`Detailed Report` prints a text summary of the structural frame — grid, group count, surface point and orientation counts, and per-group interpolation method and parameters. On an Octree Grid, it also lists the grid settings and, once the model is computed, the number of cells per level and their share of the finest cells:

```python
structural_model_result.structural_frame.detailed_report()
```

`Plot Model Result 3D` — `plot_structural_model_3D(frame, ...)`:

- **`frame`**: The `StructuralFrame` to visualise, e.g. `structural_model_result.structural_frame`.
- **`mesh_type`** *(optional)*: Which geological surface mesh to render — `"masked"`, `"unmasked"`, `"combined"`, `"extended"`, or `"extended_masked"` (see [Output](#output)). Default `"masked"`.
- **`fault_mesh_type`** *(optional)*: Mesh type for fault surfaces. Defaults to `mesh_type` if not set.
- **`show_surface_meshes`** *(optional)*: Render the per-element surface meshes. Default `True`.
- **`show_points`** *(optional)*: Render input surface points. Default `True`.
- **`show_orientations`** *(optional)*: Render orientation arrows. Default `True`.
- **`fault_opacity`** *(optional)*: Opacity of fault surface meshes. Default `0.35`.
- **`show_lith_block`** *(optional)*: Show the lithology block as the outside of the model — the cells on the six walls of the extent, colored by formation and with their edges drawn, so the grid's cells (uniform voxels, or octree cells of different sizes) are visible. The block hides what lies inside it. Default `False`.
- **`notebook`** *(optional)*: Use PyVista's notebook rendering mode. Default `False`.
- **`off_screen`** *(optional)*: Render off-screen with no window — required before calling `.screenshot()` without `show()`. Default `False`.
- **`show`** *(optional)*: If `True`, opens the interactive window immediately; set `False` to add further actors before displaying. Default `True`.

The Workbench version calls this with `mesh_type="masked"` and `show_surface_meshes=True`.

```python
plot_structural_model_3D(structural_model_result.structural_frame, mesh_type="masked", fault_opacity=0.5)
```

`Plot Model Result 2D` — `plot_structural_model_2D(frame, ...)`:

- **`frame`**: The `StructuralFrame` to visualise.
- **`axis`** *(optional)*: Axis to slice along — `"x"`, `"y"`, or `"z"`. Default `"y"`.
- **`index`** *(optional)*: Slice index along the chosen axis, counted in the finest cells on an Octree Grid. Defaults to the mid-point.
- **`show_result`** *(optional)*: Overlay the lithology block as a colored background. Default `True`.
- **`show_fault_contours`** *(optional)*: Overlay fault isolines. Default `True`.
- **`show_input_data`** *(optional)*: Overlay input surface points and orientation arrows. Default `True`.
- **`show_grid`** *(optional)*: Outline the grid cells the slice cuts — uniform voxels on a Regular Grid, cells of different sizes on an Octree Grid. Default `False`.
- **`title_suffix`** *(optional)*: Text appended to the plot title. Default `""`.

```python
plot_structural_model_2D(structural_model_result.structural_frame, axis="x", index=20)
```

### Inspect the fault model

> **Workbench components:** `Detailed Fault Report`, `Plot Fault Model Result 2D`, `Plot Fault Model Result 3D`

These act on a `FaultModelResults` object.

`Detailed Fault Report`:

```python
fault_model_result.fault_frame.detailed_report()
```

`Plot Fault Model Result 3D` — `plot_fault_model_3D(fault_frame, ...)`:

- **`fault_frame`**: The `FaultFrame` to visualise, e.g. `fault_model_result.fault_frame`.
- **`mesh_type`** *(optional)*: Which fault surface mesh to render — `"masked"`, `"unmasked"`, `"extended"`, or `"extended_masked"` (see [Output](#output)). Default `"unmasked"`.
- **`show_surface_meshes`** *(optional)*: Render fault surface meshes. Default `True`.
- **`show_input_data`** *(optional)*: Render input surface points and orientation arrows. Default `True`.
- **`show_domain_map`** *(optional)*: Render translucent fault domain blocks. Default `True`.
- **`domain_opacity`** *(optional)*: Opacity of the domain blocks. Default `0.15`.
- **`highlight_crosscuts`** *(optional)*: Highlight the line where two fault surfaces cross. Has no effect if no faults cross or if `show_surface_meshes` is `False`. Default `True`.
- **`crosscut_color`** *(optional)*: Color of the highlighted crosscut lines. Default `"red"`.
- **`crosscut_line_width`** *(optional)*: Line width of the highlighted crosscut lines. Default `6.0`.
- **`notebook`** *(optional)*: Use PyVista's notebook rendering mode. Default `False`.
- **`show`** *(optional)*: If `True`, opens the interactive window immediately. Default `True`.

```python
plot_fault_model_3D(fault_model_result.fault_frame, show_domain_map=True, domain_opacity=0.2)
```

`Plot Fault Model Result 2D` — `plot_fault_model_2D(fault_frame, ...)`:

- **`fault_frame`**: The `FaultFrame` to visualise.
- **`axis`** *(optional)*: Axis to slice along — `"x"`, `"y"`, or `"z"`. Default `"y"`.
- **`index`** *(optional)*: Slice index along the chosen axis, counted in the finest cells on an Octree Grid. Defaults to the mid-point.
- **`show_results`** *(optional)*: Plot the fault domain map and isolines, if present. Default `True`.
- **`show_input_data`** *(optional)*: Plot input surface points, if available. Default `True`.
- **`show_fault_contours`** *(optional)*: Overlay fault isolines. Default `True`.
- **`show_grid`** *(optional)*: Outline the grid cells the slice cuts, as in `plot_structural_model_2D`. Default `False`.

```python
plot_fault_model_2D(fault_model_result.fault_frame, axis="y", index=15)
```

---

## Output

> **Workbench:** The `structural_model` output port of `Compute Structural Model` carries the result. It can be connected directly to meshing components or inspected in-place using `Detailed Report`, `Plot Model Result 2D`, and `Plot Model Result 3D`.

`compute_structural_model` returns a `StructuralModelResults` object containing a deep copy of the structural frame with all computed results attached:

- **Scalar fields** — one float array per stratigraphic group, shaped `(nx, ny, nz)` on a Regular Grid.
- **Lithology block** — an integer array shaped `(nx, ny, nz)` on a Regular Grid, where each voxel holds the formation ID of the youngest formation present at that location (0 = basement / no formation).
- **Octree cells** *(Octree Grid only)* — the cells the grid ended up with, stored on the frame as `cells` (level and integer index of each cell). On an Octree Grid, the scalar fields, age masks, lithology block and fault domain map hold one value per cell, shaped `(n_cells,)` and in the order of `cells`.
- **Surface meshes** — vertices and triangle faces per formation, stored under the mesh type keys `masked`, `unmasked`, `combined`, `extended`, `extended_masked`.
  - `extended_masked` is meant as input for 3D mesh generation of faulted models. Each fault domain contributes its own piece, extracted from the scalar field interpolated in that domain alone, so the pieces on both sides of a fault never have to be separated afterwards. The pieces are stored together in one mesh but share no vertices, so they can be split into connected components. Each piece is cut at the faults and at the base of all younger groups, and reaches about one grid cell beyond both, so it reliably intersects the fault and unconformity surfaces. Like `extended`, the pieces reach beyond the model boundary.
- **Fault surface meshes** — `masked`, `unmasked`, `extended` and `extended_masked` meshes for each fault element. The fault's `extended_masked` mesh is cut at the base of all groups younger than the youngest group the fault offsets, and reaches about one grid cell beyond it.
- **All intermediate results** — scalar values per element, age masks per group, and the fault domain map are stored on the structural frame within the result object and can be accessed programmatically for further analysis or visualisation.

The `StructuralModelResults` object is the direct input to the meshing components. The meshing components currently need a structural model computed on a Regular Grid and refuse a model computed on an Octree Grid (see the [meshing manual](../meshing/Manual_meshing.md)).
