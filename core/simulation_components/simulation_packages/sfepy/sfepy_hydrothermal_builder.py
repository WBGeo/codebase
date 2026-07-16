"""
Auto-generates SfePy input file(s) for a simplified, single-phase Darcy flow
+ advective-conductive heat transport problem, inferring rock units and
boundary regions directly from a WBGeo MeshResults + StructuralModelResults
pair, instead of a hand-written static file with hardcoded regions.

This is the input-data side only: HydrothermalProblemBuilder builds the
generated SfePy input files and their configuration. Actually running them
(subprocess launch, output loading) lives in sfepy_hydrothermal_run.py,
split out to keep "build the input data" and "run the simulation" as
separate concerns -- matching the Workbench's component/connector model,
where HydrothermalProblemBuilder is the input-data-generator component and
run_simulation_sfepy() (sfepy_hydrothermal_run.py) is the run component.

Solved as two sequential, single-physics stages (pressure, then temperature),
not one monolithic multi-field system -- see run_simulation_sfepy()'s docstring for why:
solving both together in one combined linear system was tried first and
turned out to be badly conditioned (pressure ~1e6 Pa and temperature ~10s of
degC in the same matrix), and no amount of preconditioner tuning fixed it.
Segregating matches the actual physics anyway: temperature depends on the
flow field, but pressure never depends on temperature in this model.

Wired up as Workbench components in simulation_workbench_components.py
(build_hydrothermal_problem, run_hydrothermal_simulation,
export_simulation_results), including a SmartInput sidebar for per-rock-unit,
fluid, and fault-zone property editing (matching
structural_workbench_components.py's interpolation-options pattern).
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyvista as pv
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from py_api_wbgeo.nodesapi import wbgeo_type
from core.object_components import MeshResults, StructuralModelResults, SimulationResults, MeshType

logger = logging.getLogger(__name__)


class FluidProperties(BaseModel):
    """Single global fluid shared across all rock units (single-phase assumption)."""
    mu: float = 1.0e-3            # dynamic viscosity, Pa*s (water at ~20 C)
    k_fluid: float = 0.6          # thermal conductivity, W/(m*K)
    rho_c_fluid: float = 4.186e6  # volumetric heat capacity, J/(m^3*K)


class RockUnitProperties(BaseModel):
    """Per-rock-unit physical properties for the simplified Darcy + heat model."""
    name: str
    porosity: float = 0.15
    permeability: float = 1e-14   # isotropic scalar permeability, m^2
    k_solid: float = 2.5          # thermal conductivity, W/(m*K)
    rho_c_solid: float = 2.2e6    # volumetric heat capacity, J/(m^3*K)

    def effective_conductivity(self, fluid: FluidProperties) -> float:
        """Porosity-weighted thermal conductivity of the saturated rock (solid + pore fluid)."""
        return self.porosity * fluid.k_fluid + (1 - self.porosity) * self.k_solid

    def effective_heat_capacity(self, fluid: FluidProperties) -> float:
        """Porosity-weighted volumetric heat capacity of the saturated rock (solid + pore fluid)."""
        return self.porosity * fluid.rho_c_fluid + (1 - self.porosity) * self.rho_c_solid


#: Valid `linear_solver` keywords for HydrothermalProblemBuilder, and the
#: tradeoff each one represents. Both stages (pressure, temperature) are
#: solved as *separate*, single-physics linear systems (see module
#: docstring) to avoid mixing pressure (~1e6 Pa) and temperature (~10s of
#: degC) magnitudes in one combined system, which is badly conditioned for
#: iterative solvers.
#:
#: The temperature stage's own equation combines rho_c_eff (~1e6) and k_eff
#: (~1-5) terms, so an inappropriately short `t1` relative to the model's
#: actual thermal diffusion timescale (order of years for typical rock
#: properties and domain sizes) makes the transient term's matrix
#: contribution (~rho_c_eff/dt) dwarf the diffusive term's (~k_eff/L^2) by
#: many orders of magnitude -- an internal scale mismatch that defeats
#: preconditioning regardless of solver choice. `t1` defaults to a
#: data-driven diffusion timescale for exactly this reason (see
#: _default_diffusion_timescale()) rather than a fixed constant.
#:
#: - "direct"    -- scipy_direct (sparse LU factorization). Exact (up to
#:                  floating-point precision), no tuning needed. Does not
#:                  scale to very large meshes -- expect multi-minute solves
#:                  at full production resolution. Use for small/debug meshes.
#: - "iterative" -- scipy_iterative (BiCGStab, ILU-preconditioned + diagonal
#:                  scaling). DEFAULT. Scales much better to large meshes.
#:                  As with any iterative method, verify convergence (check
#:                  the printed residual) for new rock/fluid parameter sets
#:                  or a manually-supplied `t1` that deviates a lot from the
#:                  diffusion timescale.
LINEAR_SOLVERS = ("direct", "iterative")


#: MeshResults.mesh_type's readable values, mapped to the short codes
#: export_mesh_results_to_exodus()/sfepy-run actually need ("imp", "str",
#: "unstr"), which read as unexplained abbreviations at a call site. Used by
#: sfepy_hydrothermal_run.py too (imported from here) to translate
#: builder.mesh_type to the short code when actually running SfePy.
MESH_TYPE_CODES = {
    MeshType.IMPLICIT: "imp",
    MeshType.STRUCTURED: "str",
    MeshType.UNSTRUCTURED: "unstr",
}


def check_mesh_has_known_type(mesh_results: MeshResults) -> None:
    """
    Pre-check: mesh_results must carry its own provenance (mesh_type),
    normally set automatically by create_implicit_structured_mesh/
    create_structured_mesh_data/create_unstructured_mesh_data. Used both as
    a Workbench input_checks pre-check (build_hydrothermal_problem) and
    internally in HydrothermalProblemBuilder.__post_init__ -- pre-checks can
    be skipped by a caller (see docs/developers/components.md), so it's
    also enforced there.
    """
    if mesh_results.mesh_type is None:
        raise ValueError(
            "mesh_results.mesh_type is None -- HydrothermalProblemBuilder needs to know "
            "which meshing component produced this mesh (create_implicit_structured_mesh, "
            "create_structured_mesh_data, or create_unstructured_mesh_data), which normally "
            "set this automatically. If mesh_results was constructed manually (e.g. in a "
            "test), set mesh_type explicitly to one of MeshType's members."
        )


def check_mesh_has_lithology_mapping(mesh_results: MeshResults) -> None:
    """
    Pre-check: mesh_results must carry a per-block lithology mapping
    (cell_data["block_id"]). Implicit and structured meshes always have
    this; unstructured meshes only get it when create_unstructured_mesh_data
    was called with mapping_litho="automatic_centers" or "automatic_corners"
    (not "manual" or "none", neither of which produce a block_id tag). Used
    both as a Workbench input_checks pre-check (build_hydrothermal_problem)
    and internally in HydrothermalProblemBuilder._map_mat_id_to_lithology --
    pre-checks can be skipped by a caller (see docs/developers/components.md),
    so it's also enforced there.
    """
    cell_data = mesh_results.cell_data or {}
    if "block_id" not in cell_data:
        raise ValueError(
            "mesh_results has no lithology mapping (cell_data['block_id']) -- "
            "HydrothermalProblemBuilder needs the mesh's per-block lithology mapping. "
            "For unstructured meshes, make sure create_unstructured_mesh_data was called "
            "with mapping_litho='automatic_centers' or 'automatic_corners' (not 'manual' "
            "or 'none', neither of which produce a block_id tag). Implicit and structured "
            "meshes always carry this tag."
        )


def check_mesh_has_no_engineering_objects(mesh_results: MeshResults) -> None:
    """
    Pre-check: mesh_results must not contain any well (dim==1, "line" block)
    or source (dim==0, "vertex" block). CustomSfepyBuilder doesn't support
    these yet -- today's mesh-export pre-filter (_run_sfepy_input_file, for
    mesh_type_code == "unstr") already silently drops them before ever
    reaching SfePy, which is a silent trap; this makes that unsupported
    combination a loud rejection instead. Used both as a Workbench
    input_checks pre-check (build_custom_sfepy_problem) and internally in
    CustomSfepyBuilder.__post_init__ -- pre-checks can be skipped by a
    caller (see docs/developers/components.md), so it's also enforced there.
    """
    bad = [b for b in mesh_results.elements if b.dim < 2]
    if bad:
        raise ValueError(
            f"mesh_results has {len(bad)} well/source block(s) (dim < 2) -- "
            f"CustomSfepyBuilder doesn't support these yet."
        )


def map_mat_id_to_lithology(mesh_results: MeshResults) -> Dict[int, int]:
    """
    Map each mesh element block (mat_id, assigned in the same enumeration
    order as mesh_results.elements) to a lithology ID, trusting
    MeshResults.cell_data["block_id"] directly -- see
    check_mesh_has_lithology_mapping for which mesh types/mapping_litho
    settings provide this tag.

    Non-volume blocks (block_id == -1, meshing's sentinel for a fault
    surface, well, source point, or boundary/"extended" surface block) are
    dropped from the returned mapping rather than raising -- they are
    never part of the actual FEM domain SfePy solves (Exo_format.py's
    Exodus writer already excludes them from the exported volume mesh; see
    core/meshing_components/mesh_format/exodus/Exo_format.py's
    `volume_blocks` split), so there is nothing to solve for them. They
    always trail the real volume blocks positionally (lithology-block
    merging always runs first when meshing builds mesh_results.elements --
    see mesh_data.py's block-append order), so the remaining mapping's
    keys stay a contiguous 0..N-1 range exactly matching mat_id 0..N-1 in
    the exported/solved mesh.

    Standalone module-level function (not just a method on
    HydrothermalProblemBuilder) so CustomSfepyBuilder can reuse the exact
    same mat_id/"cells of group N" numbering to validate a user-supplied
    SfePy file against a mesh, without needing to construct a
    HydrothermalProblemBuilder just to get it. Every place that needs a
    lithology mapping (HydrothermalProblemBuilder._region_lines,
    build_pressure_input_file, build_heat_input_file,
    compute_darcy_velocity, compute_cell_materials,
    check_custom_sfepy_regions) iterates this mapping's keys instead of
    range(len(mesh_results.elements)) for exactly this reason.
    """
    check_mesh_has_lithology_mapping(mesh_results)
    cell_data = mesh_results.cell_data or {}
    mapping = {
        mat_id: int(np.unique(block_ids)[0])
        for mat_id, block_ids in enumerate(cell_data["block_id"])
    }
    return {mat_id: lith_id for mat_id, lith_id in mapping.items() if lith_id != -1}


_CELLS_OF_GROUP_RE = re.compile(r"cells\s+of\s+group\s+(\d+)")


def extract_referenced_groups(input_file_contents: str) -> Set[int]:
    """
    Best-effort scan of a SfePy input file's raw source text for every
    'cells of group N' region-selector literal. Text-based, not
    exec()-based: SfePy input files can use either a module-level
    `regions = {...}` dict (what HydrothermalProblemBuilder generates) or
    a top-level `define()` function -- exec()-then-read-the-`regions`-
    attribute only works for the first style. A plain text search works
    for both, and never executes untrusted uploaded code (unlike
    exec()-ing an uploaded file at construction time, before any explicit
    "Run" action). Known limitation: misses dynamically-built selector
    strings (e.g. an f-string) and can't tell a real selector from a
    similarly-worded comment -- acceptable since this is a soft
    pre-flight sanity check, not the final authority (sfepy-run's own
    region resolution is).
    """
    return {int(m) for m in _CELLS_OF_GROUP_RE.findall(input_file_contents)}


def check_custom_sfepy_regions(
    input_file_contents: str, mesh_results: MeshResults, extra_valid_groups: Set[int] = frozenset(),
) -> None:
    """
    Raises if a custom SfePy input file's text references a
    'cells of group N' that doesn't exist on mesh_results, via
    map_mat_id_to_lithology's mesh-block-index ("mat_id"/group) numbering
    -- NOT enumerate_rock_units' lithology_id numbering, a different
    space: mat_id is mesh block position (what SfePy's "group" actually
    means once _run_sfepy_input_file tags the exported mesh), lithology_id
    is a semantic rock-unit id, and one lithology can span multiple
    mat_ids when e.g. a fault splits it into disconnected mesh blocks.

    extra_valid_groups: additional group ids to accept beyond real
    lithologies -- e.g. CustomSfepyBuilder.fault_group_id, an id
    injected at run time (not present in mesh_results.cell_data at all)
    for a "damage zone" of cells reassigned away from their host
    lithology (see compute_fault_zone_cell_mask).

    Logs a warning (does not raise) if mesh_results has a group the file
    never references -- may be intentional (e.g. a group deliberately
    left unsolved), so not treated as an error.
    """
    valid = set(map_mat_id_to_lithology(mesh_results).keys()) | set(extra_valid_groups)
    referenced = extract_referenced_groups(input_file_contents)
    unknown = referenced - valid
    if unknown:
        raise ValueError(
            f"Custom SfePy input file references group(s) {sorted(unknown)} via "
            f"'cells of group N' that do not exist on mesh_results (valid groups: "
            f"{sorted(valid)}). This usually means the file was written for a "
            f"different mesh."
        )
    unused = valid - referenced
    if unused:
        logger.warning(
            "mesh_results has group(s) %s this custom SfePy file's text never "
            "references via 'cells of group N' -- if unintentional, part of the "
            "mesh may be left unsolved.", sorted(unused),
        )


def enumerate_rock_units(geomodel_result: StructuralModelResults) -> List[Tuple[int, str]]:
    """
    Ordered (lithology_id, formation_name), oldest(1)->youngest(N), plus
    basement(0). A module-level function (not just a method on
    HydrothermalProblemBuilder) so it can also be used to enumerate rock
    unit names for a smart-options form before a builder is constructed
    (see simulation_workbench_components.py).
    """
    frame = geomodel_result.structural_frame
    units: List[Tuple[int, str]] = [(0, "basement")]
    # structural_groups/elements are stored youngest->oldest
    for group in reversed(frame.structural_groups):
        for elem in reversed(group.structural_elements):
            if elem.id is not None:
                units.append((elem.id, elem.name))
    units.sort(key=lambda t: t[0])
    return units


def lithology_to_group_idx(geomodel_result: StructuralModelResults) -> Dict[int, int]:
    """
    Lithology id -> the index into structural_frame.structural_groups it
    belongs to, in the *same* convention frame.fault_activity/
    effective_domain_components_for_group() (general.py) use: index 0 =
    youngest group. Basement (lithology 0) deliberately excluded -- it
    isn't part of any group, and is handled separately in
    compute_fault_zone_cell_mask() (treated as always fault-affected,
    since it sits below/outside every group a fault's activity could
    possibly be restricted to).

    Standalone module-level function so it's reusable by anything that
    wants a fault-zone cell mask without going through
    HydrothermalProblemBuilder (e.g. CustomSfepyBuilder).
    """
    frame = geomodel_result.structural_frame
    mapping: Dict[int, int] = {}
    for group_idx, group in enumerate(frame.structural_groups):
        for elem in group.structural_elements:
            if elem.id is not None:
                mapping[elem.id] = group_idx
    return mapping


def fault_trace_on_grid(fault, active_mask: np.ndarray) -> np.ndarray:
    """
    Boolean array, same shape as the structural model's lith_block,
    marking grid cells adjacent to `fault`'s zero-crossing surface --
    i.e. where `fault.get_domain_mask()` (the "which side" mask, already
    computed as scalar_field > scalar_value when the fault frame was
    built) flips between one neighboring cell and the next along any
    axis -- AND restricted to `active_mask` (see
    compute_fault_zone_cell_mask()): the structural modeling component
    supports *finite* faults that don't cut every stratigraphic group
    (model9 is the clearest example; even model2's single fault could be
    set to only affect its older group via set_fault_activity_by_index()).
    Without this restriction, cells in a group the fault doesn't
    actually reach would still show a "crossing" here -- get_domain_mask()
    is a purely geometric quantity (scalar_field > scalar_value) that
    doesn't know about fault_activity at all, only
    effective_domain_components_for_group() (general.py) applies that
    during the real lith_block computation, merging the fault's domains
    back together (no real offset) for inactive groups.
    """
    domain_mask = fault.get_domain_mask()
    crossing = np.zeros(domain_mask.shape, dtype=bool)
    for axis in range(domain_mask.ndim):
        diff = np.diff(domain_mask, axis=axis)
        idx_lo = [slice(None)] * domain_mask.ndim
        idx_hi = [slice(None)] * domain_mask.ndim
        idx_lo[axis] = slice(0, -1)
        idx_hi[axis] = slice(1, None)
        crossing[tuple(idx_lo)] |= diff
        crossing[tuple(idx_hi)] |= diff
    return crossing & active_mask


def compute_fault_zone_cell_mask(
    mesh_results: MeshResults, geomodel_result: StructuralModelResults, fault_zone_n_voxels: int = 1,
) -> Optional[np.ndarray]:
    """
    Boolean array, one entry per mesh cell (concatenated in
    mesh_results.elements block order), True where that cell falls within
    fault_zone_n_voxels of any fault's trace. None if there are no faults
    in this structural model at all, or if the resulting mask is empty
    (e.g. fault_zone_n_voxels too small relative to mesh resolution, or
    fault_activity restricts every fault away from every group) -- both
    logged as a warning rather than an error, since a caller's config
    might be reused across faulted/unfaulted models.

    Standalone module-level function (not just a method on
    HydrothermalProblemBuilder) so it's reusable by any caller wanting the
    same "damage zone" cell selection without HydrothermalProblemBuilder's
    other machinery (rock-unit enumeration, auto-generated materials,
    etc.) -- e.g. CustomSfepyBuilder, which only needs the cell selection
    itself and leaves what to *do* with it entirely to the custom file.
    """
    frame = geomodel_result.structural_frame
    fault_frame = getattr(frame, "fault_frame", None)
    if fault_frame is None or not fault_frame.fault_elements:
        logger.warning(
            "fault_zone_n_voxels was given but this structural model has no "
            "faults (structural_frame.fault_frame is empty/None) -- ignoring."
        )
        return None

    # Per-voxel group index, in fault_activity's convention (0 =
    # youngest group; basement gets a sentinel index one past the last
    # real group, so `group_idx >= youngest_idx` is always True for it
    # regardless of youngest_idx -- basement sits below/outside every
    # group, so any fault reaching this deep affects it unconditionally).
    lith_to_group = lithology_to_group_idx(geomodel_result)
    n_groups = len(frame.structural_groups)
    group_idx_grid = np.full(frame.lith_block.shape, n_groups, dtype=int)
    for lith_id, group_idx in lith_to_group.items():
        group_idx_grid[frame.lith_block == lith_id] = group_idx

    fault_activity = frame.fault_activity or {}
    combined_trace = np.zeros(frame.lith_block.shape, dtype=bool)
    for fault in fault_frame.fault_elements:
        # Same default (0 = fully active) set_fault_frame() uses when
        # fault_activity isn't explicitly restricted.
        youngest_idx = fault_activity.get(fault.name, 0)
        active_mask = group_idx_grid >= youngest_idx
        combined_trace |= fault_trace_on_grid(fault, active_mask)

    if fault_zone_n_voxels > 0:
        combined_trace = binary_dilation(combined_trace, iterations=fault_zone_n_voxels)

    tree = cKDTree(frame.grid.grid_coordinates)
    flat_trace = combined_trace.ravel()

    cell_mask_chunks = []
    for block in mesh_results.elements:
        centroids = mesh_results.nodes[block.data].mean(axis=1)
        _, idx = tree.query(centroids)
        cell_mask_chunks.append(flat_trace[idx])
    cell_mask = np.concatenate(cell_mask_chunks) if cell_mask_chunks else np.array([], dtype=bool)

    if not cell_mask.any():
        logger.warning(
            "fault_zone_n_voxels was given but no mesh cells fell within "
            "fault_zone_n_voxels=%d of any *active* fault trace -- ignoring (try a "
            "larger fault_zone_n_voxels, check the mesh actually resolves this model's "
            "faults, or check fault_activity isn't restricting every fault away from "
            "every group it's meshed in).",
            fault_zone_n_voxels,
        )
        return None
    return cell_mask


@wbgeo_type(name='Hydrothermal Problem', color='#e07a5f',
            identifier='HydrothermalProblemBuilder')
@dataclass
class HydrothermalProblemBuilder:
    """
    Builds SfePy problem-description files for a coupled Darcy flow + heat
    transport problem, solved as two sequential single-physics stages.
    Material regions and boundary regions are inferred automatically from
    the mesh and structural model.

    Temperature is coupled to the Darcy flow: pressure is solved first, a
    Darcy velocity is computed from its gradient, and that velocity enters
    the temperature equation as a `dw_advect_div_free` advection term.

    Set `include_flow=False` to skip the pressure stage entirely and solve
    pure heat conduction instead -- useful to validate the diffusion term in
    isolation (e.g. against the known analytical steady-state profile) before
    trusting the coupled result. Rock units still use porosity-weighted
    effective conductivity/heat capacity by default in this mode (a static,
    saturated rock); to also remove the fluid's contribution entirely, set
    `porosity=0.0` on the relevant `RockUnitProperties` -- the same formulas
    then reduce exactly to `k_solid`/`rho_c_solid`.

    `mesh_type` (which meshing component produced `mesh_results` -- one of
    `MeshType.IMPLICIT`/`STRUCTURED`/`UNSTRUCTURED`) is read directly from
    `mesh_results.mesh_type`, set automatically by
    create_implicit_structured_mesh/create_structured_mesh_data/
    create_unstructured_mesh_data -- not a separate constructor argument,
    since that would just be a second place the same fact could go stale or
    disagree with the actual mesh. Raises clearly if `mesh_results.mesh_type`
    is `None` (e.g. a manually-constructed `MeshResults` in a test that never
    set it). All three mesh types work end-to-end for unfaulted models; a
    structured mesh paired with a faulted structural model is not supported
    (blocked at the meshing stage, since that combination does not reliably
    converge -- use implicit or unstructured meshing for faulted models
    instead).

    Limited fault support: pass `fault_zone_properties` (a `RockUnitProperties`)
    to give cells near any fault their own separate material, instead of
    silently treating them as ordinary lithology. Off by default (None).
    Faults aren't tagged anywhere on `MeshResults` for implicit/structured
    meshes (only a per-block lithology id); only unstructured meshing
    explicitly meshes fault surfaces, and SfePy itself has no fault/interface element
    either (only per-region material constants, same as lithology already
    uses) -- so this represents a fault as a "damage zone" band of cells
    straddling its trace (`fault_zone_n_voxels` grid cells wide, computed on
    the *structural model's* grid via each fault's `get_domain_mask()`,
    then resampled to mesh cells the same way lithology is resampled for
    unstructured meshes), a standard, simpler proxy for fault hydraulics
    (gouge seal or fracture-enhanced conduit) than an exact geometric split.
    All faults in the structural model are combined into one shared "fault
    zone" material region (`Omega_fault`) -- not per-fault -- deliberately
    the smallest useful increment, not a complete fault model. Respects
    `structural_frame.fault_activity` though: the structural modeling
    component supports *finite* faults that don't cut every stratigraphic
    group (model9 is the clearest example; a single fault can also be
    restricted this way via `set_fault_activity_by_index`/`_by_group`), and
    a fault's raw `get_domain_mask()` doesn't know about that restriction at
    all (it's a purely geometric scalar_field > scalar_value quantity) --
    so the fault-zone trace is masked, per fault, to only the grid cells
    whose lithology's stratigraphic group that fault is actually active for
    (`group_idx >= fault_activity[name]`, the same rule
    `effective_domain_components_for_group` in general.py uses for the real
    lith_block computation; basement is always treated as active, since it
    sits below/outside every group). Setting `fault_zone_properties`
    identical to the surrounding rock's properties is a no-op on the
    result (useful for isolating a suspected fault-zone effect by
    comparison).
    """

    mesh_results: MeshResults
    geomodel_result: StructuralModelResults
    rock_properties: Optional[Dict[str, RockUnitProperties]] = None
    fluid: Optional[FluidProperties] = None
    include_flow: bool = True
    fault_zone_properties: Optional[RockUnitProperties] = None
    fault_zone_n_voxels: int = 1
    t0: float = 0.0
    t1: Optional[float] = None
    num_steps: int = 1
    p_top: float = 0.0
    p_bottom: float = 100e4
    t_top: float = 10.0
    t_bottom: float = 60.0
    linear_solver: str = "iterative"
    linear_solver_i_max: int = 500
    linear_solver_eps_r: float = 1e-8

    def __post_init__(self):
        """
        Validates and fills in every derived/defaulted piece of state from
        the fields above (see the class docstring for the constructor
        argument semantics).

        Raises:
            ValueError: linear_solver isn't a valid LINEAR_SOLVERS entry, or
                mesh_results.mesh_type is None (see that check's message for why).
        """
        if self.linear_solver not in LINEAR_SOLVERS:
            raise ValueError(f"linear_solver must be one of {LINEAR_SOLVERS}, got {self.linear_solver!r}")

        check_mesh_has_known_type(self.mesh_results)
        self.mesh_type = self.mesh_results.mesh_type

        self.fluid = self.fluid or FluidProperties()

        self.rock_units: List[Tuple[int, str]] = self._enumerate_rock_units()
        self.rock_properties: Dict[str, RockUnitProperties] = self._fill_defaults(self.rock_properties or {})
        self.mat_id_to_lithology: Dict[int, int] = self._map_mat_id_to_lithology()

        # Limited fault support: opt-in via fault_zone_properties. Faults
        # aren't tagged anywhere on MeshResults for the mesh types that
        # actually work here (implicit/structured only carry lith_block).
        # So instead of a geometrically-exact fault surface, this treats a
        # fault as a "damage zone" band of cells straddling its trace
        # (n_voxels wide, on the *structural model's* grid) and gives them
        # their own separate material -- a standard, simpler proxy for
        # fault-zone hydraulics (gouge seal or fracture-enhanced conduit)
        # than an exact geometric split, and the only mesh-type-agnostic
        # option since SfePy itself has no fault/interface element either,
        # only per-region material constants.
        self.fault_zone_cell_mask: Optional[np.ndarray] = None
        if self.fault_zone_properties is not None:
            self.fault_zone_cell_mask = self._compute_fault_zone_cell_mask()

        # The extra SfePy/Exodus material group id for the fault zone --
        # just needs to be an id that doesn't collide with any real lithology
        # mat_id (self.mat_id_to_lithology's keys); len(mesh_results.elements)
        # safely exceeds all of those (lithology mat_ids are always a
        # contiguous 0..N-1 prefix -- see _map_mat_id_to_lithology).
        self.fault_group_id: Optional[int] = (
            len(self.mesh_results.elements) if self.fault_zone_cell_mask is not None else None
        )

        # t1 defaults to a data-driven thermal diffusion timescale rather
        # than an arbitrary constant -- see _default_diffusion_timescale()
        # for why an inappropriately short t1 breaks the heat equation's own
        # conditioning.
        self.t1: float = self.t1 if self.t1 is not None else self._default_diffusion_timescale()

    # ------------------------------------------------------------------
    # Rock unit / lithology bookkeeping
    # ------------------------------------------------------------------

    def _enumerate_rock_units(self) -> List[Tuple[int, str]]:
        return enumerate_rock_units(self.geomodel_result)

    def _fill_defaults(self, given: Dict[str, RockUnitProperties]) -> Dict[str, RockUnitProperties]:
        """Fill in `RockUnitProperties(name=<unit>)` defaults for any rock unit in
        `self.rock_units` not already present in `given`, so every real lithology
        has an entry regardless of how much the caller specified explicitly."""
        result = dict(given)
        for _, name in self.rock_units:
            if name not in result:
                result[name] = RockUnitProperties(name=name)
        return result

    def _map_mat_id_to_lithology(self) -> Dict[int, int]:
        return map_mat_id_to_lithology(self.mesh_results)

    # ------------------------------------------------------------------
    # Fault zone (limited support -- see the class docstring)
    # ------------------------------------------------------------------

    def _lithology_to_group_idx(self) -> Dict[int, int]:
        return lithology_to_group_idx(self.geomodel_result)

    def _fault_trace_on_grid(self, fault, active_mask: np.ndarray) -> np.ndarray:
        return fault_trace_on_grid(fault, active_mask)

    def _compute_fault_zone_cell_mask(self) -> Optional[np.ndarray]:
        return compute_fault_zone_cell_mask(self.mesh_results, self.geomodel_result, self.fault_zone_n_voxels)

    def compute_cell_materials(self) -> np.ndarray:
        """
        Per-mesh-cell material name (a lithology's name, or
        fault_zone_properties.name for cells reassigned to the fault zone),
        one entry per cell, concatenated in mesh_results.elements block
        order -- the exact material each cell will actually be solved with
        (same per-cell partition _run_sfepy_input_file() uses to tag each
        cell's mat_id, just resolved to a human name instead of a group id).

        Public and separate from the SfePy input-file generation so it can
        be used for visual QA before running a solve that can take a while
        -- see plot_builder_materials() in
        core.simulation_components.simulation_visualization.simulation_visualization.
        """
        names = []
        offset = 0
        for mat_id, lith_id in self.mat_id_to_lithology.items():
            block = self.mesh_results.elements[mat_id]
            n = len(block.data)
            block_names = np.full(n, self._lithology_name(lith_id), dtype=object)
            if self.fault_zone_cell_mask is not None:
                block_mask = self.fault_zone_cell_mask[offset:offset + n]
                block_names[block_mask] = self.fault_zone_properties.name
            names.append(block_names)
            offset += n
        return np.concatenate(names) if names else np.array([], dtype=object)

    def _lithology_name(self, lith_id: int) -> str:
        """Look up a lithology id's name in `self.rock_units`; raises if not found."""
        for lid, name in self.rock_units:
            if lid == lith_id:
                return name
        raise ValueError(f"Lithology id {lith_id} not found in structural model")

    def _default_diffusion_timescale(self) -> float:
        """
        Rough thermal diffusion timescale (L^2 / alpha) used as the default
        `t1`, so the heat equation's transient term (~rho_c_eff/dt) and
        diffusive term (~k_eff/L^2) land at comparable magnitude in the
        assembled matrix.

        An arbitrarily short t1 relative to this timescale (typically on
        the order of years, given realistic rock properties and domain
        sizes) makes the transient term dominate the matrix by many orders
        of magnitude -- an internal scale mismatch that defeats simple
        preconditioning even though the equation itself is fine.
        """
        extent = self.geomodel_result.structural_frame.grid.extent
        length_scale = min(extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
        alphas = [
            props.effective_conductivity(self.fluid) / props.effective_heat_capacity(self.fluid)
            for props in self.rock_properties.values()
        ]
        alpha = float(np.mean(alphas))
        return length_scale ** 2 / alpha

    # ------------------------------------------------------------------
    # Shared region/solver boilerplate
    # ------------------------------------------------------------------

    def _region_lines(self) -> List[str]:
        """
        SfePy `regions` dict body (as literal source lines), shared by both the
        pressure and heat input files: the whole-domain 'Omega', one 'Omega{mat_id}'
        per real lithology, 'Omega_fault' if a fault zone is active, and the
        vertex-based 'Gamma_Top'/'Gamma_Bottom' boundary regions (from
        mesh_results.point_sets, not Exodus side-sets).
        """
        lines = ["    'Omega': 'all',"]
        for mat_id in self.mat_id_to_lithology:
            lines.append(f"    'Omega{mat_id}': 'cells of group {mat_id}',")
        if self.fault_group_id is not None:
            lines.append(f"    'Omega_fault': 'cells of group {self.fault_group_id}',")
        top_ids = list(map(int, self.mesh_results.point_sets["top"]))
        bottom_ids = list(map(int, self.mesh_results.point_sets["bottom"]))
        lines.append(f"    'Gamma_Bottom': ('vertex {','.join(map(str, bottom_ids))}', 'vertex'),")
        lines.append(f"    'Gamma_Top': ('vertex {','.join(map(str, top_ids))}', 'vertex'),")
        return lines

    def _solver_blocks(self) -> Tuple[str, str]:
        """Returns (precond_fn_def, ls_block) for the configured linear_solver."""
        if self.linear_solver == "direct":
            return "", "    'ls': ('ls.scipy_direct', {}),"

        precond_fn_def = '''
def setup_precond(mtx, context):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    d = mtx.diagonal().copy()
    d[d == 0] = 1.0
    Dinv = sp.diags(1.0 / d)
    scaled = (Dinv @ mtx).tocsc()
    ilu = spla.spilu(scaled, drop_tol=1e-3, fill_factor=2)
    def solve(b):
        return ilu.solve(Dinv @ b)
    return spla.LinearOperator(mtx.shape, solve)
'''
        ls_block = f"""    'ls': ('ls.scipy_iterative', {{
        'method': 'bicgstab',
        'i_max': {self.linear_solver_i_max!r},
        'eps_r': {self.linear_solver_eps_r!r},
        'setup_precond': setup_precond,
    }}),"""
        return precond_fn_def, ls_block

    def _newton_ts_options_blocks(self, ls_block: str) -> str:
        """
        SfePy `solvers`/`options` dict body (literal source), shared by both the
        pressure and heat input files: the linear solver block passed in (`ls_block`,
        from _solver_blocks), the Newton nonlinear solver (`report_status=True` so
        _run_sfepy_input_file's _check_convergence can parse a real pass/fail signal),
        and the `ts.simple` time-stepping config using self.t0/t1/num_steps.
        """
        return f"""
solvers = {{
{ls_block}
    'newton': ('nls.newton', {{
        'i_max': 5,
        'eps_a': 1e-6,
        'is_linear': False,
        # report_status defaults to False in SfePy -- without it, the only
        # printed line per solve is the per-iteration residual (no
        # definitive "did this actually converge" signal). With it,
        # SfePy also prints a final "cond: N, iter: ..., err0: ..., err: ..."
        # line per solve (N: 0 = converged, 1 = max iterations reached
        # without converging, 2 = linesearch gave up) -- that's what
        # _run_sfepy_input_file()'s convergence check parses.
        'report_status': True,
    }}),
    'ts': ('ts.simple', {{
        't0': t0,
        't1': t1,
        'n_step': num_steps,
        'verbose': 1,
    }}),
}}

output_dir = os.environ.get("SFEpy_OUTPUT_DIR", "output")
os.makedirs(output_dir, exist_ok=True)

options = {{
    'nls': 'newton',
    'ls': 'ls',
    'ts': 'ts',
    'save_times': 'all',
    'output_dir': output_dir,
}}
"""

    # ------------------------------------------------------------------
    # Stage 1: pressure (Darcy flow)
    # ------------------------------------------------------------------

    def build_pressure_input_file(self, output_path: str) -> str:
        """Write the SfePy input file for the steady-state Darcy pressure solve."""
        region_lines = self._region_lines()
        material_lines = []
        flow_terms = []

        for mat_id, lith_id in self.mat_id_to_lithology.items():
            props = self.rock_properties[self._lithology_name(lith_id)]
            perm_over_mu = props.permeability / self.fluid.mu
            # dw_diffusion requires its material coefficient as a full D x D
            # tensor (isotropic permeability -> scaled identity matrix).
            perm_over_mu_tensor = [[perm_over_mu if i == j else 0.0 for j in range(3)] for i in range(3)]
            material_lines.append(f"    'm{mat_id}': ({{'perm_over_mu': {perm_over_mu_tensor!r}}},),")
            flow_terms.append(f"dw_diffusion.i.Omega{mat_id}(m{mat_id}.perm_over_mu, q, p)")

        if self.fault_group_id is not None:
            perm_over_mu = self.fault_zone_properties.permeability / self.fluid.mu
            perm_over_mu_tensor = [[perm_over_mu if i == j else 0.0 for j in range(3)] for i in range(3)]
            material_lines.append(f"    'm_fault': ({{'perm_over_mu': {perm_over_mu_tensor!r}}},),")
            flow_terms.append("dw_diffusion.i.Omega_fault(m_fault.perm_over_mu, q, p)")

        flow_eq = "\n  + ".join(flow_terms) + "\n  = 0"
        precond_fn_def, ls_block = self._solver_blocks()

        content = f'''"""
Auto-generated SfePy problem: steady-state Darcy pressure (stage 1 of 2).
Generated by HydrothermalProblemBuilder -- do not edit by hand.
"""
import os
import numpy as np

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")

regions = {{
{chr(10).join(region_lines)}
}}

fields = {{
    'pressure': ('real', 1, 'Omega', 1),
}}

t0 = {self.t0!r}
t1 = {self.t1!r}
# Pressure has no time-derivative term (steady-state Darcy flow), so it is
# always solved in a single step regardless of the heat stage's num_steps --
# run_simulation_sfepy()'s `(time,) = pressure_sim.nodes_by_time.keys()` depends on this.
num_steps = 1
dt = (t1 - t0) / num_steps

variables = {{
    'p': ('unknown field', 'pressure', 1),
    'q': ('test field', 'pressure', 'p'),
}}
{precond_fn_def}
materials = {{
{chr(10).join(material_lines)}
}}

ebcs = {{
    'p_bottom': ('Gamma_Bottom', {{'p.0': {self.p_bottom!r}}}),
    'p_top': ('Gamma_Top', {{'p.0': {self.p_top!r}}}),
}}

integrals = {{
    'i': 1,
}}

equations = {{
    'flow': """{flow_eq}""",
}}
{self._newton_ts_options_blocks(ls_block)}'''
        with open(output_path, "w") as f:
            f.write(content)
        return output_path

    # ------------------------------------------------------------------
    # Velocity hand-off
    # ------------------------------------------------------------------

    def compute_darcy_velocity(self, pressure_sim: SimulationResults, time: float) -> Dict[int, np.ndarray]:
        """
        Reconstruct the solved pressure field as a PyVista grid and compute
        the per-cell Darcy velocity v = -(permeability/mu) * grad(p), split
        by mesh block (mat_id) so each region gets its own array. If a fault
        zone is active, cells within it are pulled out of their host
        block's array into a separate `self.fault_group_id`-keyed array --
        matching exactly how _run_sfepy_input_file() reassigns those same
        cells' mat_id when building the mesh (same per-block boolean-mask
        split, concatenated in the same block order, so the resulting
        arrays line up with SfePy's own per-region cell enumeration, which
        preserves each cell's relative/global order when filtering by group).

        Relies on mesh_results.elements' block order surviving through
        SfePy's mesh conversion and back out through the VTK it writes (the
        same assumption run_sfepy's own mat_id tagging already depends on).
        """
        nodes = pressure_sim.nodes_by_time[time]
        cells = pressure_sim.cells_by_time[time]
        celltypes = pressure_sim.celltypes_by_time[time]
        grid = pv.UnstructuredGrid(cells, celltypes, nodes)
        grid.point_data["p"] = pressure_sim.node_data_by_time[time]["p"]
        # p is point (nodal) data; compute_derivative's output follows its
        # input's location (point in -> point out), and preference='cell'
        # does NOT change that -- so convert to cell data first (averaging
        # nodal values per cell) to get a genuinely per-cell gradient/
        # velocity out.
        grid = grid.point_data_to_cell_data()
        grid = grid.compute_derivative(scalars="p")
        grad_p = np.asarray(grid.cell_data["gradient"])

        fault_perm_over_mu = (
            self.fault_zone_properties.permeability / self.fluid.mu
            if self.fault_group_id is not None else None
        )

        velocities: Dict[int, np.ndarray] = {}
        fault_chunks = []
        offset = 0
        for mat_id, lith_id in self.mat_id_to_lithology.items():
            block = self.mesh_results.elements[mat_id]
            n = len(block.data)
            props = self.rock_properties[self._lithology_name(lith_id)]
            perm_over_mu = props.permeability / self.fluid.mu
            block_velocity = -perm_over_mu * grad_p[offset:offset + n]

            if self.fault_zone_cell_mask is not None:
                block_fault_mask = self.fault_zone_cell_mask[offset:offset + n]
                velocities[mat_id] = block_velocity[~block_fault_mask]
                fault_chunks.append(-fault_perm_over_mu * grad_p[offset:offset + n][block_fault_mask])
            else:
                velocities[mat_id] = block_velocity
            offset += n

        if self.fault_group_id is not None:
            velocities[self.fault_group_id] = (
                np.concatenate(fault_chunks) if fault_chunks else np.empty((0, 3))
            )
        return velocities

    # ------------------------------------------------------------------
    # Stage 2: temperature (advective-conductive heat transport)
    # ------------------------------------------------------------------

    def build_heat_input_file(self, output_path: str, velocity_npz_path: Optional[str] = None) -> str:
        """Write the SfePy input file for the temperature solve. If
        self.include_flow, uses the precomputed Darcy velocity stored at
        velocity_npz_path (required in that case); otherwise builds a pure
        conduction problem (no advection term, no stage 1 dependency)."""
        region_lines = self._region_lines()
        material_lines = []
        heat_terms = []
        velocity_fn_defs = []
        velocity_fn_names = []

        # (region_name, group_id_for_material/velocity_names, k_eff, rho_c_eff)
        # -- one per real lithology (mat_id_to_lithology's keys, not every
        # raw mesh_results.elements index -- see that method's docstring for
        # why), plus the fault zone group (if active) using
        # fault_zone_properties directly instead of a lithology lookup.
        # group_id for the fault zone is self.fault_group_id, matching the
        # numeric keys compute_darcy_velocity()/run_simulation_sfepy()'s
        # npz-saving loop already use for it -- everything downstream
        # (material/velocity function naming) stays a single, uniform loop
        # either way.
        groups: List[Tuple[str, int, float, float]] = []
        for mat_id, lith_id in self.mat_id_to_lithology.items():
            props = self.rock_properties[self._lithology_name(lith_id)]
            groups.append((
                f"Omega{mat_id}", mat_id,
                props.effective_conductivity(self.fluid), props.effective_heat_capacity(self.fluid),
            ))
        if self.fault_group_id is not None:
            props = self.fault_zone_properties
            groups.append((
                "Omega_fault", self.fault_group_id,
                props.effective_conductivity(self.fluid), props.effective_heat_capacity(self.fluid),
            ))

        for region_name, gid, k_eff, rho_c_eff in groups:
            material_lines.append(
                f"    'm{gid}': ({{'k_eff': {k_eff!r}, 'rho_c_eff': {rho_c_eff!r}}},),"
            )
            heat_eq_terms = [
                f"dw_volume_dot.i.{region_name}(m{gid}.rho_c_eff, s, dT/dt)",
                f"  + dw_laplace.i.{region_name}(m{gid}.k_eff, s, T)",
            ]

            if self.include_flow:
                material_lines.append(f"    'darcy{gid}': 'get_darcy_velocity_{gid}',")
                heat_eq_terms.append(f"  + dw_advect_div_free.i.{region_name}(darcy{gid}.v, s, T)")

                fn_name = f"get_darcy_velocity_{gid}"
                velocity_fn_names.append(fn_name)
                velocity_fn_defs.append(
                    f"def {fn_name}(ts, coors, mode=None, problem=None, **kwargs):\n"
                    f"    if mode != 'qp':\n"
                    f"        return\n"
                    f"    v_per_cell = _velocity_data['v{gid}']  # (n_cells, 3), precomputed from stage 1\n"
                    f"    n_qp = coors.shape[0] // v_per_cell.shape[0]\n"
                    f"    v = np.repeat(v_per_cell, n_qp, axis=0)\n"
                    f"    v = v.reshape(v.shape[0], v.shape[1], 1)\n"
                    f"    return {{'v': v}}\n"
                )
            heat_terms.append("\n".join(heat_eq_terms))

        heat_eq = "\n  + ".join(heat_terms) + "\n  = 0"
        functions_lines = ",\n".join(f"    '{name}': ({name},)" for name in velocity_fn_names)
        precond_fn_def, ls_block = self._solver_blocks()

        if self.include_flow:
            if velocity_npz_path is None:
                raise ValueError("velocity_npz_path is required when include_flow=True")
            velocity_data_line = f"_velocity_data = np.load({velocity_npz_path!r})"
            stage_docline = "Uses the Darcy velocity computed from stage 1's pressure solution."
        else:
            velocity_data_line = ""
            stage_docline = "Pure conduction (include_flow=False): no advection term, no stage 1 dependency."

        content = f'''"""
Auto-generated SfePy problem: heat transport.
{stage_docline}
Generated by HydrothermalProblemBuilder -- do not edit by hand.
"""
import os
import numpy as np

filename_mesh = os.environ.get("TEMP_MESH_FILE", "filename.mesh")

regions = {{
{chr(10).join(region_lines)}
}}

fields = {{
    'temperature': ('real', 1, 'Omega', 1),
}}

t0 = {self.t0!r}
t1 = {self.t1!r}
num_steps = {self.num_steps!r}
dt = (t1 - t0) / num_steps

variables = {{
    'T': ('unknown field', 'temperature', 0, 1),
    's': ('test field', 'temperature', 'T'),
}}

{velocity_data_line}

{chr(10).join(velocity_fn_defs)}
{precond_fn_def}
def ic_temp(coor, ic=None):
    # Exact linear interpolation between the enforced T_top/T_bottom EBCs
    # over the mesh's own actual z-extent (NOT a fixed geothermal-gradient
    # rate over the nominal domain height) -- mesh nodes are cell-centered,
    # inset from the nominal domain edges by half a cell on each side, so
    # using a fixed rate over the nominal height leaves a resolution-
    # dependent mismatch at the boundary that then diffuses through the
    # domain over time -- pure conduction (include_flow=False) should be
    # exactly static since this IC already is the steady-state solution.
    z_min, z_max = float(np.min(coor[:, 2])), float(np.max(coor[:, 2]))
    T_top, T_bottom = {self.t_top!r}, {self.t_bottom!r}
    return T_top + (T_bottom - T_top) * (z_max - coor[:, 2]) / (z_max - z_min)

functions = {{
    'ic_temp': (ic_temp,),
{functions_lines}
}}

materials = {{
{chr(10).join(material_lines)}
}}

ics = {{
    'ic': ('Omega', {{'T.0': 'ic_temp'}}),
}}

ebcs = {{
    'T_bottom': ('Gamma_Bottom', {{'T.0': {self.t_bottom!r}}}),
    'T_top': ('Gamma_Top', {{'T.0': {self.t_top!r}}}),
}}

integrals = {{
    'i': 1,
}}

equations = {{
    'heat': """{heat_eq}""",
}}
{self._newton_ts_options_blocks(ls_block)}'''
        with open(output_path, "w") as f:
            f.write(content)
        return output_path


@wbgeo_type(name='Custom SfePy Problem', color='#e07a5f', identifier='CustomSfepyBuilder')
@dataclass
class CustomSfepyBuilder:
    """
    Wraps a user-supplied, already-complete SfePy input file (with its own
    internal staging/sequencing logic -- e.g. a hand-written GOLEM-style
    coupled problem) instead of auto-generating one like
    HydrothermalProblemBuilder does. Runs through the same
    run_simulation_sfepy() and returns the same SimulationResults type.

    input_file_contents is stored as text, not a file path: @wbgeo_type
    instances are serialized/reconstructed between separate Workbench
    component executions, and a temp file path would not survive that
    round trip. The content is only written to disk transiently at run
    time, matching how HydrothermalProblemBuilder's own generated input
    files work.

    Validated at construction against mesh_results (known mesh type,
    known lithology mapping, and a best-effort static check that every
    'cells of group N' the file references actually exists on the mesh --
    see check_custom_sfepy_regions). This validation is deliberately not
    exhaustive: it cannot catch wrong material properties or malformed
    equations, only structural mismatches with the mesh. sfepy-run's own
    region resolution remains the final authority at run time.

    Unlike HydrothermalProblemBuilder, no pressure/heat staging, no
    velocity handoff -- the file is assumed to already contain everything
    it needs and gets run in a single sfepy-run invocation (see
    run_simulation_sfepy in sfepy_hydrothermal_run.py). On an unstructured
    mesh, the same mesh_type_code == "unstr" pre-filter
    HydrothermalProblemBuilder relies on (_run_sfepy_input_file, dropping
    non-volume blocks before Exodus export) applies here too -- so a
    custom file cannot reference a non-volume (fault-surface/boundary)
    block by group number on an unstructured mesh, the same limitation
    HydrothermalProblemBuilder already has. mesh_results with any
    well/source (dim < 2) block is rejected outright at construction (see
    check_mesh_has_no_engineering_objects) -- not supported yet.

    fault_zone_n_voxels: optional opt-in for a "damage zone" cell
    selection, the same mechanism HydrothermalProblemBuilder uses (see
    compute_fault_zone_cell_mask) -- a band of cells straddling each
    active fault's trace gets reassigned its own group id
    (self.fault_group_id), referenceable from the custom file as
    'cells of group {fault_group_id}' (e.g. as its own 'Omega_fault'
    region). Unlike HydrothermalProblemBuilder, no material/equation is
    auto-generated for this group -- the custom file's author decides
    what to do with it; this only provides the cell selection. Requires
    geomodel_result (raises if not given). None (default) leaves the
    mesh's real lithology groups as the only ones available, exactly like
    today.
    """
    input_file_contents: str
    mesh_results: MeshResults
    geomodel_result: Optional[StructuralModelResults] = None
    fault_zone_n_voxels: Optional[int] = None

    def __post_init__(self):
        check_mesh_has_known_type(self.mesh_results)
        check_mesh_has_no_engineering_objects(self.mesh_results)
        check_mesh_has_lithology_mapping(self.mesh_results)
        self.mesh_type = self.mesh_results.mesh_type
        self.mat_id_to_lithology = map_mat_id_to_lithology(self.mesh_results)

        self.fault_zone_cell_mask: Optional[np.ndarray] = None
        self.fault_group_id: Optional[int] = None
        if self.fault_zone_n_voxels is not None:
            if self.geomodel_result is None:
                raise ValueError(
                    "fault_zone_n_voxels was given but geomodel_result is None -- "
                    "computing the fault-zone cell mask needs the structural model."
                )
            self.fault_zone_cell_mask = compute_fault_zone_cell_mask(
                self.mesh_results, self.geomodel_result, self.fault_zone_n_voxels,
            )
            if self.fault_zone_cell_mask is not None:
                self.fault_group_id = len(self.mesh_results.elements)

        check_custom_sfepy_regions(
            self.input_file_contents, self.mesh_results,
            extra_valid_groups={self.fault_group_id} if self.fault_group_id is not None else set(),
        )
        if self.geomodel_result is not None:
            n_referenced = len(extract_referenced_groups(self.input_file_contents))
            n_lithologies = len(enumerate_rock_units(self.geomodel_result))
            if n_referenced < n_lithologies:
                logger.warning(
                    "File references %d group(s) via 'cells of group N' but the "
                    "structural model defines %d lithologies -- expected if a "
                    "lithology is split across disconnected mesh blocks (e.g. a "
                    "fault), otherwise the file may be missing coverage.",
                    n_referenced, n_lithologies,
                )
