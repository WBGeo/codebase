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

Fault interpolation uses Universal Co-Kriging exclusively, so **both surface points and orientations are required for every fault**. Orientation vectors are scaled to unit length first, so only their direction matters. The result is a `FaultModelResults` object containing:

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

1. **Validation** — checks that every active fault domain contains sufficient input data for the chosen interpolation method in each group (for example, that orientation data is present in each domain for methods that require it, or that there are enough surface points for the chosen method). The same check runs in the Workbench as a pre-check when the inputs are connected. All issues are reported together before any expensive computation begins.
2. **Interpolation** — each stratigraphic group is interpolated independently. When faults are active for a group, the interpolation is run completely separately in each fault domain using only the data points that fall within that domain. The resulting scalar fields are then stitched together.
3. **Age masks** — a boolean mask is computed per group that represents its depositional extent: the oldest group underlies the entire domain (mask is True everywhere), while younger groups are only present where they were deposited. When assembling the lithology block, younger groups take precedence over older ones wherever their mask is True — this is what produces unconformable contacts between groups.
4. **Lithology block** — the scalar fields and age masks are combined into a single integer array where each cell is assigned to a formation.
5. **Surface meshes** — isosurfaces are extracted from the scalar fields using marching cubes on a Regular Grid, or dual contouring on an Octree Grid. Five mesh types are stored per element: `masked` (respects unconformities and fault domains), `unmasked` (full isosurface through the entire grid), `combined` (derived from the lithology block), `extended` (extrapolated beyond the grid boundary, used as input for 3D mesh generation), and `extended_masked` (extrapolated beyond the grid boundary like `extended`, but one separate piece per fault domain, cut at the faults and at the base of younger groups; see [Output](#output)).

**Limitations of the structural model:**

- **Group interactions are limited to unconformities.** The only geometric relationship supported between stratigraphic groups is that a younger group erodes the top of an older one. There is no support for intrusions, cross-cutting bodies, or any other non-erosional contact geometry between groups.
- **Sufficient data required in every active fault domain.** When a fault is active for a group, the interpolation in each fault domain uses only the data points that fall within that domain. The validation step rejects domains with too few surface points for the chosen method, or without orientations for methods that need them (UCK, FDI, PLI, GeoINR); the minimum for each method is given in its section below. A domain without any surface points of a group is skipped: the group is not interpolated there. Even above the minimum, sparse data in a domain gives a less reliable result.

---

## Interpolation Methods

All methods produce the same output format (scalar field + per-element isovalues) and can be set per stratigraphic group. The exposed parameters listed below are a subset of the full configuration space — each underlying library (SciPy, PyKrige, GemPy, LoopStructural, PyTorch) has many additional options that are not exposed.

| Method | Orientations required | Library                                                          | Adjustable parameters |
|---|---|------------------------------------------------------------------|---|
| **Radial Basis Function (RBF)** | No | [SciPy](https://scipy.org/)                                      | `kernel`, `smoothing`, `epsilon`, `neighbors`, `anisotropy_scaling_z` (see [below](#radial-basis-function-rbf)) |
| **Ordinary Kriging (OK)** | No | [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/) | `variogram_model`, `range`, `nugget`, `anisotropy_scaling_y/z`, `neighbors` (see [below](#ordinary-kriging-ok)) |
| **Universal Kriging (UK)** | No | [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/) | Same as OK without `neighbors` (see [below](#universal-kriging-uk)) |
| **Universal Co-Kriging (UCK)** | Yes | [GemPy](https://www.gempy.org/)                                  | None (see [below](#universal-co-kriging-uck)) |
| **Finite Differences (FDI)** | Yes | [LoopStructural](https://loop3d.github.io/LoopStructural/)       | `nelements`, `regularisation`, `orientation_weight` (see [below](#finite-differences-fdi-and-piecewise-linear-pli)) |
| **Piecewise Linear (PLI)** | Yes | [LoopStructural](https://loop3d.github.io/LoopStructural/)       | `nelements`, `regularisation`, `orientation_weight` (see [below](#finite-differences-fdi-and-piecewise-linear-pli)) |
| **GeoINR** | Yes | [GeoINR](https://github.com/MichaelHillier/GeoINR) (custom)                               | `orientation_weight`, `hidden_dim`, `n_hidden_layers`, `epochs`, `lr`, `seed` (see [below](#geoinr)) |

For all methods that use orientations, only the direction of the orientation vectors affects the result, not their length: UCK normalises them before passing them to GemPy, LoopStructural normalises them for FDI and PLI, and GeoINR normalises them in its orientation loss. Orientations can lie anywhere, not only on an interface; their formation only assigns them to a group.

Two of these parameters take one of a fixed set of options. An unknown option is rejected when it is set.

| Parameter | Methods | Options (default first) |
|---|---|---|
| `kernel` | RBF | `thin_plate_spline`, `linear`, `cubic`, `quintic`, `multiquadric`, `inverse_multiquadric`, `inverse_quadratic`, `gaussian` |
| `variogram_model` | OK, UK | `gaussian`, `spherical`, `exponential`, `hole-effect` |

Default parameters are estimated automatically from the spatial distribution of the input points (data scale, point count, nearest-neighbour distance) and can be overridden via `configure_interpolation_params`. The estimate uses all points of a group, across all fault domains. Parameter choice has a large influence on the resulting model — the default values are a reasonable starting point but should be treated as a first guess rather than an optimal configuration. In particular, variogram range and anisotropy for kriging methods, `regularisation` for FDI and PLI (and `nelements` for PLI with little data), and the number of epochs and network capacity for GeoINR can all significantly change the shape and smoothness of the interpolated surfaces.

### Radial Basis Function (RBF)

RBF fits a smooth scalar field through the surface points of a group: each formation gets a value (oldest = 1, youngest = n), and the field between the points is built from one radial function per point plus, for some kernels, a polynomial trend. Orientations are not used; if they are given, a warning says so. RBF is the default method.

**A group with only one formation.** RBF needs at least two formations to tell up from down. For a group with a single formation, a copy of its surface points is placed below them (shifted down by 10 % of the point cloud's diagonal) as an artificial older formation, and a warning is shown. This assumes roughly horizontal layers; for steep or overturned layers, use a method with orientations (UCK, FDI, PLI, GeoINR).

**Minimum data.** The kernels `thin_plate_spline`, `cubic` and `quintic` add a polynomial trend, which has to be fitted from the points. They need a minimum number of surface points in each group and fault domain, and the points must not all lie in one plane (for example, a single flat surface on its own). The copied points of a single-formation group count. The other kernels work with a single point.

| Kernel | Trend | Minimum surface points |
|---|---|---|
| `thin_plate_spline` (default), `cubic` | plane | 4, not all in one plane |
| `quintic` | curved surface | 10, not all on one plane or one simple curved surface |
| `linear`, `multiquadric`, `gaussian`, `inverse_multiquadric`, `inverse_quadratic` | none or constant | 1 |

This is checked before the computation (and in the Workbench when the inputs are connected). If a group or fault domain has too few points, add points there or choose `linear` or `multiquadric`.

| Parameter | Meaning | Default | Effect |
|---|---|---|---|
| `kernel` | Shape of the radial function | `thin_plate_spline` | `thin_plate_spline` gives smooth, gently extrapolating surfaces and needs no length scale. Kernels using `epsilon` are more local. |
| `smoothing` | How far the field may deviate from the points (0 = passes exactly through them) | 0.05 | Larger values give smoother surfaces that no longer honour every point; useful for noisy data. The scale is the step between two formations (1). |
| `epsilon` | Shape parameter, an inverse length (only `multiquadric`, `inverse_multiquadric`, `inverse_quadratic`, `gaussian`; ignored by the others) | 1 / (5 × median nearest-neighbour distance) | Larger values make each point's influence more local. Refers to the stretched coordinates when `anisotropy_scaling_z` is not 1. |
| `neighbors` | Number of nearest points used at each grid cell (empty = all points) | all points; for more than 5000 points min(500, 5 · n^(2/3)) | Fewer neighbours are faster and more local, but can produce small steps in the surfaces. Must be at least 4 (10 for `quintic`). |
| `anisotropy_scaling_z` | Vertical stretching factor applied to points and grid before fitting | 1.0 (no anisotropy) | Values above 1 make vertical distances count more than horizontal ones, so points influence each other more along the layers than across them. Values below 1 do the opposite. |

### Ordinary Kriging (OK)

Ordinary Kriging interpolates the scalar field from the surface points with a variogram, which describes how quickly two points become independent with increasing distance. Each formation gets a value (oldest = 1, youngest = n), as for RBF. Orientations are not used; if they are given, a warning says so. A group with only one formation gets the same copied formation below it as described for [RBF](#radial-basis-function-rbf).

**Minimum data.** At least 2 surface points in each group and fault domain (the copied points of a single-formation group count). This is checked before the computation.

The defaults are derived from the spread of the points: per axis, the distance between the 10 % and 90 % quantiles of the point coordinates.

| Parameter | Meaning | Default | Effect |
|---|---|---|---|
| `variogram_model` | Shape of the variogram: `gaussian`, `spherical`, `exponential`, `hole-effect` | `gaussian` | `gaussian` gives the smoothest surfaces. `spherical` and `exponential` give slight kinks at the points. `hole-effect` is meant for periodic structures. |
| `range` | Distance (model units) up to which points influence each other | 20 × median nearest-neighbour distance, limited to between 0.1 × the largest spread and 1.5 × the diagonal of the spread | Longer ranges give smoother, more regional surfaces; shorter ranges follow the points more locally. |
| `nugget` | Fraction of the variance that is not spatially correlated (0 to below 1) | 0 | 0 = surfaces pass exactly through the points. Larger values give smoother surfaces that no longer honour every point; useful for noisy data. |
| `anisotropy_scaling_y` | Factor applied to y coordinates before kriging, relative to x | spread in y / spread in x, limited to 0.05–1 | Values below 1 make distances along y count less, i.e. a longer range along y. |
| `anisotropy_scaling_z` | Factor applied to z coordinates before kriging, relative to x | spread in z / spread in x, limited to 0.05–1 | Values below 1 make vertical distances count less, i.e. a longer vertical range. For layered data the vertical spread is small, so the default reproduces the gradual vertical change between layers smoothly. |
| `neighbors` | Number of nearest points used at each grid cell (moving window; empty = all points) | all points below 20 points; otherwise n / 10, limited to 30–200 | A moving window keeps memory low for many points but is slower. If a fault domain has no more points than `neighbors`, all its points are used. |

The sill (the total variance) is fixed at 1, because it does not change the interpolated field; only the nugget relative to it does. A tiny nugget (10⁻⁶ of the sill) is always added internally to keep the computation numerically stable, which is most important for the Gaussian variogram. It does not visibly change the result.

### Universal Kriging (UK)

Universal Kriging works like Ordinary Kriging, but additionally fits a linear trend (a + b·x + c·y + d·z) together with the variogram. This suits dipping layers whose scalar field changes steadily across the model. Orientations are not used, and a single-formation group gets the copied formation as for RBF. Universal Kriging always uses all points (there is no moving window), so memory grows with the number of points and grid cells.

**Minimum data.** At least 4 surface points in each group and fault domain, not all in one plane (for example, a single flat surface on its own). Otherwise the linear trend cannot be fitted. If a domain has too few points, add points there or use Ordinary Kriging.

The parameters are the same as for Ordinary Kriging, without `neighbors`, and the internal jitter is added in the same way. Two defaults differ:

| Parameter | Default |
|---|---|
| `range` | 30 × median nearest-neighbour distance, limited to between 0.15 × and 0.9 × the largest spread |
| `anisotropy_scaling_y`, `anisotropy_scaling_z` | spread in y (z) / spread in x, limited to 0.05–20 |

### Universal Co-Kriging (UCK)

Universal Co-Kriging (implemented with GemPy) interpolates the scalar field from surface points and orientations together: the surface points of each formation define where its surface lies, the orientations define the dip direction of the layers. Only the direction of the orientation vectors matters; they are scaled to unit length before use. Because the orientations define which way is up, a group with a single formation works without the copied formation used by RBF and Kriging. After the interpolation, the scalar field is rescaled so that the formations get the values 1 … n like in the other methods.

UCK has no adjustable parameters: GemPy's default settings are used, which are derived from the model extent.

**Minimum data.** In each group and fault domain:

- at least one orientation, and
- at least one surface point of every formation of the group. UCK takes each formation's value from that formation's own points, so a formation without points in a fault domain cannot be placed there. The other methods number the formations from the group order and do not need this.

Both are checked before the computation. If a formation is missing in a fault domain, add surface points of it there, or use another method for this group.

### Finite Differences (FDI) and Piecewise Linear (PLI)

FDI and PLI (both implemented with LoopStructural) solve the scalar field on their own internal computation grid, which covers the model extent plus a 20 % margin. FDI uses a regular grid of cells, PLI a mesh of tetrahedra. Each formation gets a value (oldest = 1, youngest = n), as for RBF. The field is the weighted least-squares solution of three conditions:

- it has the formation's value at every surface point (weight 1);
- its layers have the measured dip at every orientation (weight `orientation_weight`);
- it is smooth (weight `regularisation`).

Only the ratios between these weights matter. Orientations set only the dip at their location; they do not pull an interface towards them, so they can lie anywhere, for example in the middle of a layer. If a group (or a fault domain) has surface points of only one formation, LoopStructural uses the orientations to define which way is up, so a group with a single formation works without the copied formation used by RBF and Kriging.

LoopStructural's piecewise-linear interpolator has no gradient constraint. For PLI, each orientation is therefore passed as two tangent constraints: the field's gradient must be perpendicular to the strike vector and to the dip vector, which is equivalent to a gradient constraint without its length. LoopStructural weights tangent constraints by the volume of their mesh element; the weight passed to it is divided by the mean element volume, so `orientation_weight` has the same effect for any `nelements`.

**Minimum data.** At least one orientation in each group and fault domain that contains surface points. This is checked before the computation.

| Parameter | Methods | Meaning | Default | Effect |
|---|---|---|---|---|
| `nelements` | FDI, PLI | Number of elements of the internal computation grid (at least 1000) | FDI: 10 000 × √(points / 1000), limited to 5000–200 000; PLI: 2000 × √(points / 1000) × 0.8–1.4 (denser sampling of the model gives the larger factor), limited to 2000–50 000 | More elements give finer surfaces but take much longer; beyond the default, the result usually changes only slightly. For small or sparse data, PLI's default mesh can be too coarse to follow the orientations; increase `nelements` (e.g. to 10 000) if the layers do not follow the measured dips. |
| `regularisation` | FDI, PLI | Weight of the smoothness term | 0.1 | Larger values give smoother, more planar surfaces that may not honour every data point; smaller values follow the data more closely but can become wavy. LoopStructural's own default for FDI (1.0) gives noticeably smoother surfaces that follow the data less closely. |
| `orientation_weight` | FDI, PLI | Weight of the orientations relative to the surface points | 1.0 | Larger values make the layers follow the measured dips more closely; smaller values let the surfaces follow the surface points more closely. Matters most when there are few surface points. |

The solver (conjugate gradients with LoopStructural's default tolerance) is fixed.

### GeoINR

GeoINR represents the scalar field by a small neural network (a multilayer perceptron, implicit neural representation) that returns the field value for any location; the implementation follows the [GeoINR reference implementation](https://github.com/MichaelHillier/GeoINR). Each formation gets a fixed label, evenly spaced from +1 (youngest) to −1 (oldest). The network is trained by minimising three terms:

- the difference between the field and the formation's label at every surface point;
- the angle between the field's gradient and the measured orientation at every orientation (weight `orientation_weight`);
- the deviation of the gradient magnitude across the model from the gradient magnitude at the surface points (fixed weight). This keeps the field from folding back far away from the data.

Coordinates are scaled by the same factor on all axes, so the measured dips are not distorted. As for FDI and PLI, orientations set only the dip and can lie anywhere, and a group with a single formation works without a copied formation, because the orientations define which way is up.

**Randomness.** Training starts from random network weights, so the result depends on the random start, more so the sparser the data. `seed` fixes the start: the same input and seed give the same model. To judge how robust a model is, compute it with a few different seeds; if the results differ strongly, add data or try a smaller learning rate.

**Runtime.** The network is trained separately for each group and fault domain, which takes from seconds to minutes per group on a CPU. A CUDA-capable GPU is used automatically if PyTorch finds one.

**Minimum data.** At least one orientation in each group and fault domain that contains surface points. This is checked before the computation. Formations without surface points in a fault domain still get their label there.

The defaults depend on the number of surface points of the group (n).

| Parameter | Meaning | Default | Effect |
|---|---|---|---|
| `orientation_weight` | Weight of the orientations relative to the surface points | 0.3 | Larger values make the layers follow the measured dips more closely; smaller values let the surfaces follow the surface points more closely. If the dip is known mainly from orientations, increase it; if orientations disagree with the surface points, decrease it. |
| `hidden_dim` | Number of neurons per hidden layer | 32; 64 from 3000 points | More neurons can represent more complex geometry (e.g. tight folds) but train more slowly and may overfit sparse data. |
| `n_hidden_layers` | Number of hidden layers | 2 | Fewer layers give a less flexible field; 1 to 3 is usually enough. |
| `epochs` | Number of training steps | 4000; 2500 from 5000 points | More steps can fit the data better but take longer. |
| `lr` | Learning rate (step size of the training) | 0.01; 0.005 from 800 points; 0.003 from 5000 points | Smaller values train more steadily but need more steps. |
| `seed` | Start value of the random number generator | 0 | Changes the random start of the training (see above). |

Fixed (not adjustable): Softplus activation, the weight of the gradient-magnitude term (0.05, evaluated at 2000 randomly chosen grid points), and the AdamW optimiser with a learning rate decaying to zero over the training. The network weights of the training step with the lowest loss are used.

---

## Visualisation and Inspection

> **Workbench:** Where available, these are exposed as inspector functions with limited functionality in the visual interface (visual DSL) — only some parameters can be set there (noted per function); the codebase functions below expose the full set of options.

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

The Workbench version exposes `mesh_type` as a drop-down (default `"masked"`); faults use the same type where it exists, otherwise `"unmasked"`. The other options keep their defaults.

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

The Workbench version exposes `axis` (a choice of x, y or z) and `index` (empty = the mid-point); the other options keep their defaults.

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

The Workbench version exposes `mesh_type` as a drop-down (default `"unmasked"`); the other options keep their defaults.

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

The Workbench version exposes `axis` (a choice of x, y or z) and `index` (empty = the mid-point), as in `Plot Model Result 2D`.

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
