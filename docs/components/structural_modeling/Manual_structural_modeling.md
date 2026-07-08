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

---

## Workflow

### 1. Define the grid

> **Workbench component:** `Regular Grid`

A `RegularGrid` defines the 3D computation domain as a regular axis-aligned voxel grid:

- **Extent** — bounding box `(xmin, xmax, ymin, ymax, zmin, zmax)` in world coordinates. Should cover all input data with a small margin.
- **Resolution** — number of voxels `(nx, ny, nz)`. Higher resolution gives more geometric detail but increases computation time significantly.

### 2. Load input data

> **Workbench component:** `Input data for Structural Elements`

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

Fault interpolation currently uses Universal Co-Kriging exclusively, so **both surface points and orientations are required for every fault**. The result is a `FaultModelResults` object containing:

- A **surface mesh** for each fault (the boundary between hanging wall and footwall).
- A **domain map** that partitions every voxel in the grid into strictly separate, non-overlapping fault blocks. Stratigraphic groups that are active for a given fault will be interpolated independently in each of its domains; groups that post-date the fault are not affected by the domain split.

**Limitations of the fault model:**

- **Cross-cutting faults are not supported.** Each pair of faults is tested for isovalue-band overlap after interpolation. If two fault surfaces intersect within the model domain the computation raises an error. Ensure faults are geometrically compatible before running the model.
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
4. **Lithology block** — the scalar fields and age masks are combined into a single 3D integer array where each voxel is assigned to a formation.
5. **Surface meshes** — isosurfaces are extracted from the scalar fields using marching cubes. Four mesh types are stored per element: `masked` (respects unconformities and fault domains), `unmasked` (full isosurface through the entire grid), `combined` (derived from the lithology block), and `extended` (extrapolated beyond the grid boundary, used as input for 3D mesh generation).

**Limitations of the structural model:**

- **Group interactions are limited to unconformities.** The only geometric relationship supported between stratigraphic groups is that a younger group erodes the top of an older one. There is no support for intrusions, cross-cutting bodies, or any other non-erosional contact geometry between groups.
- **Sufficient data required in every active fault domain.** When a fault is active for a group, the interpolation in each fault domain uses only the data points that fall within that domain. If a domain contains too few surface points — or none at all — the result in that domain will be unreliable or empty. Methods that additionally require orientations (UCK, FDI, GeoINR) need orientation data in every active domain as well.

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
| **GeoINR** | Yes | [GeoINR](https://github.com/MichaelHillier/GeoINR) (custom)                               | `hidden_dim`, `n_hidden_layers`, `epochs`, `lr`, `alpha`, `beta` |

Default parameters are estimated automatically from the spatial distribution of the input points (data scale, point count, nearest-neighbour distance) and can be overridden via `configure_interpolation_params`. Parameter choice has a large influence on the resulting model — the default values are a reasonable starting point but should be treated as a first guess rather than an optimal configuration. In particular, variogram range and sill for kriging methods, `nelements` for FDI, and the number of epochs and network capacity for GeoINR can all significantly change the shape and smoothness of the interpolated surfaces.

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

`Detailed Report` prints a text summary of the structural frame — group count, surface point and orientation counts, and per-group interpolation method and parameters:

```python
structural_model_result.structural_frame.detailed_report()
```

`Plot Model Result 3D` — `plot_structural_model_3D(frame, ...)`:

- **`frame`**: The `StructuralFrame` to visualise, e.g. `structural_model_result.structural_frame`.
- **`mesh_type`** *(optional)*: Which geological surface mesh to render — `"masked"`, `"unmasked"`, `"combined"`, or `"extended"` (see [Output](#output)). Default `"masked"`.
- **`fault_mesh_type`** *(optional)*: Mesh type for fault surfaces. Defaults to `mesh_type` if not set.
- **`show_surface_meshes`** *(optional)*: Render the per-element surface meshes. Default `True`.
- **`show_points`** *(optional)*: Render input surface points. Default `True`.
- **`show_orientations`** *(optional)*: Render orientation arrows. Default `True`.
- **`fault_opacity`** *(optional)*: Opacity of fault surface meshes. Default `0.35`.
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
- **`index`** *(optional)*: Slice index along the chosen axis. Defaults to the mid-point.
- **`show_result`** *(optional)*: Overlay the lithology block as a colored background. Default `True`.
- **`show_fault_contours`** *(optional)*: Overlay fault isolines. Default `True`.
- **`show_input_data`** *(optional)*: Overlay input surface points and orientation arrows. Default `True`.
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
- **`notebook`** *(optional)*: Use PyVista's notebook rendering mode. Default `False`.
- **`show`** *(optional)*: If `True`, opens the interactive window immediately. Default `True`.

```python
plot_fault_model_3D(fault_model_result.fault_frame, show_domain_map=True, domain_opacity=0.2)
```

`Plot Fault Model Result 2D` — `plot_fault_model_2D(fault_frame, ...)`:

- **`fault_frame`**: The `FaultFrame` to visualise.
- **`axis`** *(optional)*: Axis to slice along — `"x"`, `"y"`, or `"z"`. Default `"y"`.
- **`index`** *(optional)*: Slice index along the chosen axis. Defaults to the mid-point.
- **`show_results`** *(optional)*: Plot the fault domain map and isolines, if present. Default `True`.
- **`show_input_data`** *(optional)*: Plot input surface points, if available. Default `True`.
- **`show_fault_contours`** *(optional)*: Overlay fault isolines. Default `True`.

```python
plot_fault_model_2D(fault_model_result.fault_frame, axis="y", index=15)
```

---

## Output

> **Workbench:** The `structural_model` output port of `Compute Structural Model` carries the result. It can be connected directly to meshing components or inspected in-place using `Detailed Report`, `Plot Model Result 2D`, and `Plot Model Result 3D`.

`compute_structural_model` returns a `StructuralModelResults` object containing a deep copy of the structural frame with all computed results attached:

- **Scalar fields** — one 3D float array per stratigraphic group, shaped `(nx, ny, nz)`.
- **Lithology block** — a 3D integer array shaped `(nx, ny, nz)`, where each voxel holds the formation ID of the youngest formation present at that location (0 = basement / no formation).
- **Surface meshes** — vertices and triangle faces per formation, stored under the mesh type keys `masked`, `unmasked`, `combined`, `extended`.
- **Fault surface meshes** — `masked`, `unmasked`, `extended` and `extended_masked` meshes for each fault element.
- **All intermediate results** — scalar values per element, age masks per group, and the fault domain map are stored on the structural frame within the result object and can be accessed programmatically for further analysis or visualisation.

The `StructuralModelResults` object is the direct input to the meshing components.
