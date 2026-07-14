import json
import math

import numpy as np
import pytest
from pydantic_core import to_jsonable_python

from core.object_components import MeshResults, MeshType
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    HydrothermalProblemBuilder,
    RockUnitProperties,
    FluidProperties,
    LINEAR_SOLVERS,
    check_mesh_has_known_type,
    check_mesh_has_lithology_mapping,
    enumerate_rock_units,
)


# -----------------------------------------------------------------------------
# RockUnitProperties / FluidProperties -- pure math
# -----------------------------------------------------------------------------

def test_rock_unit_properties_effective_conductivity_blends_solid_and_fluid():
    fluid = FluidProperties(k_fluid=0.6)
    rock = RockUnitProperties(name="rock1", porosity=0.2, k_solid=2.5)
    assert rock.effective_conductivity(fluid) == pytest.approx(0.2 * 0.6 + 0.8 * 2.5)


def test_rock_unit_properties_zero_porosity_reduces_to_solid_values():
    fluid = FluidProperties(k_fluid=0.6, rho_c_fluid=4.186e6)
    rock = RockUnitProperties(name="dry_rock", porosity=0.0, k_solid=3.1, rho_c_solid=2.1e6)
    assert rock.effective_conductivity(fluid) == pytest.approx(3.1)
    assert rock.effective_heat_capacity(fluid) == pytest.approx(2.1e6)


# -----------------------------------------------------------------------------
# enumerate_rock_units
# -----------------------------------------------------------------------------

def test_enumerate_rock_units_includes_basement_and_real_lithologies(model1_structural_result):
    units = enumerate_rock_units(model1_structural_result)
    names_by_id = dict(units)
    assert names_by_id[0] == "basement"
    assert set(names_by_id.values()) == {"basement", "rock1", "rock2"}
    # ids must be a contiguous 0..N-1 range
    assert sorted(names_by_id.keys()) == list(range(len(units)))


# -----------------------------------------------------------------------------
# check_mesh_has_known_type / check_mesh_has_lithology_mapping (pre-checks)
# -----------------------------------------------------------------------------

def test_check_mesh_has_known_type_passes_for_real_mesh(model1_implicit_mesh):
    check_mesh_has_known_type(model1_implicit_mesh)  # must not raise


def test_check_mesh_has_known_type_raises_for_none(model1_implicit_mesh):
    bad_mesh = model1_implicit_mesh.model_copy(update={"mesh_type": None})
    with pytest.raises(ValueError, match="mesh_type is None"):
        check_mesh_has_known_type(bad_mesh)


def test_check_mesh_has_lithology_mapping_passes_for_real_mesh(model1_implicit_mesh):
    check_mesh_has_lithology_mapping(model1_implicit_mesh)  # must not raise


def test_check_mesh_has_lithology_mapping_raises_when_missing(model1_implicit_mesh):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})
    with pytest.raises(ValueError, match="no lithology mapping"):
        check_mesh_has_lithology_mapping(bad_mesh)


def test_map_mat_id_to_lithology_drops_non_volume_sentinel_blocks(model1_implicit_mesh, model1_structural_result):
    # -1 is meshing's sentinel for a non-volume block (fault surface, well,
    # source, boundary/"extended" surface) -- must be dropped, not raise
    # and not show up as a "lithology".
    block_ids = list(model1_implicit_mesh.cell_data["block_id"])
    block_ids.append(np.full(5, -1, dtype=np.int64))
    extra_block = model1_implicit_mesh.elements[0].__class__(
        model1_implicit_mesh.elements[0].type,
        model1_implicit_mesh.elements[0].data[:5],
    )
    mesh_with_extra_block = model1_implicit_mesh.model_copy(update={
        "elements": list(model1_implicit_mesh.elements) + [extra_block],
        "cell_data": {"block_id": block_ids},
    })
    builder = HydrothermalProblemBuilder(
        mesh_results=mesh_with_extra_block, geomodel_result=model1_structural_result,
    )
    assert -1 not in builder.mat_id_to_lithology.values()
    assert len(model1_implicit_mesh.elements) not in builder.mat_id_to_lithology or True
    # the appended sentinel block's mat_id (last index) must not be a key
    assert (len(mesh_with_extra_block.elements) - 1) not in builder.mat_id_to_lithology


# -----------------------------------------------------------------------------
# HydrothermalProblemBuilder construction
# -----------------------------------------------------------------------------

def test_builder_construction_basic(model1_implicit_mesh, model1_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    assert builder.mesh_type == MeshType.IMPLICIT
    assert set(builder.rock_properties.keys()) == {"basement", "rock1", "rock2"}
    assert set(builder.mat_id_to_lithology.values()) == {0, 1, 2}
    assert builder.fault_zone_cell_mask is None
    assert builder.fault_group_id is None
    assert builder.t1 is not None and builder.t1 > 0
    assert builder.fluid == FluidProperties()


def test_builder_fills_defaults_for_unspecified_rock_units(model1_implicit_mesh, model1_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        rock_properties={"rock1": RockUnitProperties(name="rock1", porosity=0.3)},
    )
    assert builder.rock_properties["rock1"].porosity == pytest.approx(0.3)
    # basement/rock2 not specified -> filled with RockUnitProperties(name=...) defaults
    assert builder.rock_properties["basement"].porosity == pytest.approx(RockUnitProperties(name="basement").porosity)


def test_builder_invalid_linear_solver_raises(model1_implicit_mesh, model1_structural_result):
    with pytest.raises(ValueError, match="linear_solver"):
        HydrothermalProblemBuilder(
            mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
            linear_solver="not_a_real_solver",
        )


def test_builder_accepts_both_linear_solvers(model1_implicit_mesh, model1_structural_result):
    for solver in LINEAR_SOLVERS:
        builder = HydrothermalProblemBuilder(
            mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
            linear_solver=solver,
        )
        assert builder.linear_solver == solver


def test_builder_rejects_mesh_with_no_type(model1_implicit_mesh, model1_structural_result):
    bad_mesh = model1_implicit_mesh.model_copy(update={"mesh_type": None})
    with pytest.raises(ValueError, match="mesh_type is None"):
        HydrothermalProblemBuilder(mesh_results=bad_mesh, geomodel_result=model1_structural_result)


def test_builder_rejects_mesh_with_no_lithology_mapping(model1_implicit_mesh, model1_structural_result):
    bad_mesh = model1_implicit_mesh.model_copy(update={"cell_data": None})
    with pytest.raises(ValueError, match="no lithology mapping"):
        HydrothermalProblemBuilder(mesh_results=bad_mesh, geomodel_result=model1_structural_result)


def test_builder_survives_json_roundtrip(model1_implicit_mesh, model1_structural_result):
    """
    Mimics exactly what the real Workbench backend does to every @wbgeo_type
    value between component executions (see examples/pydantic_nodesapi_simulation.py's
    copy_fn): serialize via pydantic_core.to_jsonable_python + json.dumps,
    then reconstruct via Cls(**json.loads(...)). Derived/private attributes
    (rock_units, mat_id_to_lithology, ...) are computed in __post_init__, not
    stored as dataclass fields, so this also verifies they survive being
    dropped from the JSON and recomputed on the other side.
    """
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=5e9,
    )
    as_json = json.dumps(to_jsonable_python(builder))
    rebuilt = HydrothermalProblemBuilder(**json.loads(as_json))

    assert rebuilt.mesh_type == builder.mesh_type
    assert rebuilt.rock_units == builder.rock_units
    assert rebuilt.mat_id_to_lithology == builder.mat_id_to_lithology
    assert rebuilt.rock_properties.keys() == builder.rock_properties.keys()
    assert rebuilt.t1 == pytest.approx(builder.t1)


def test_default_diffusion_timescale_is_positive_and_finite(model1_implicit_mesh, model1_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    assert math.isfinite(builder.t1)
    assert builder.t1 > 0


def test_explicit_t1_is_not_overridden(model1_implicit_mesh, model1_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result, t1=1234.5,
    )
    assert builder.t1 == pytest.approx(1234.5)


# -----------------------------------------------------------------------------
# compute_cell_materials
# -----------------------------------------------------------------------------

def test_compute_cell_materials_without_fault_zone_only_uses_lithology_names(
    model1_implicit_mesh, model1_structural_result
):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    materials = builder.compute_cell_materials()
    total_cells = sum(len(mat_id_block.data) for mat_id_block in
                      [model1_implicit_mesh.elements[i] for i in builder.mat_id_to_lithology])
    assert len(materials) == total_cells
    assert set(np.unique(materials).tolist()) <= {"basement", "rock1", "rock2"}


# -----------------------------------------------------------------------------
# Generated SfePy input files -- text is syntactically valid + references
# the right regions, without actually invoking sfepy-run.
# -----------------------------------------------------------------------------

def test_build_pressure_input_file_is_valid_python_and_references_all_lithologies(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
    )
    out_path = str(tmp_path / "pressure.py")
    builder.build_pressure_input_file(out_path)
    content = open(out_path).read()
    compile(content, out_path, "exec")  # raises SyntaxError if malformed

    for mat_id in builder.mat_id_to_lithology:
        assert f"Omega{mat_id}" in content
    assert "Gamma_Top" in content
    assert "Gamma_Bottom" in content
    assert "Omega_fault" not in content  # no fault zone active


def test_build_heat_input_file_pure_conduction_is_valid_python(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        include_flow=False,
    )
    out_path = str(tmp_path / "heat.py")
    builder.build_heat_input_file(out_path)
    content = open(out_path).read()
    compile(content, out_path, "exec")
    assert "dw_advect_div_free" not in content  # include_flow=False -> no advection term


def test_build_pressure_input_file_direct_solver_uses_scipy_direct(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        linear_solver="direct",
    )
    out_path = str(tmp_path / "pressure.py")
    builder.build_pressure_input_file(out_path)
    content = open(out_path).read()
    compile(content, out_path, "exec")
    assert "ls.scipy_direct" in content
    assert "ls.scipy_iterative" not in content


def test_build_heat_input_file_with_flow_requires_velocity_path(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        include_flow=True,
    )
    with pytest.raises(ValueError, match="velocity_npz_path"):
        builder.build_heat_input_file(str(tmp_path / "heat.py"))

    out_path = str(tmp_path / "heat.py")
    builder.build_heat_input_file(out_path, velocity_npz_path="dummy_velocity.npz")
    content = open(out_path).read()
    compile(content, out_path, "exec")
    assert "dw_advect_div_free" in content
    assert "dummy_velocity.npz" in content


# -----------------------------------------------------------------------------
# Fault zone -- detection + fault_activity gating (model2, real fault)
# -----------------------------------------------------------------------------

_FAULT_ZONE_PROPS = RockUnitProperties(
    name="fault_zone", porosity=0.02, permeability=1e-19, k_solid=0.5, rho_c_solid=2.3e6
)


def test_fault_zone_detected_on_faulted_model(model2_implicit_mesh, model2_faulted_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh, geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1,
    )
    assert builder.fault_zone_cell_mask is not None
    assert builder.fault_zone_cell_mask.any()
    assert builder.fault_group_id == len(model2_implicit_mesh.elements)


def test_fault_zone_none_on_fault_free_model(model1_implicit_mesh, model1_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh, geomodel_result=model1_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS,
    )
    assert builder.fault_zone_cell_mask is None
    assert builder.fault_group_id is None


def test_fault_zone_respects_restricted_fault_activity(
    model2_faulted_structural_result, model2_implicit_mesh,
    model2_faulted_structural_result_restricted_activity, model2_implicit_mesh_restricted_activity,
):
    """
    A fault restricted (via set_fault_activity_by_group, called before
    compute_structural_model) to only affect its older stratigraphic group
    must exclude the younger group's cells from the fault zone -- this was a
    real bug in an earlier version (raw get_domain_mask() ignored
    fault_activity entirely). Cross-checks the two known-good cell counts at
    this exact fixture resolution (30x15x15): 764 fully-active, 596 restricted.
    """
    full_builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh, geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1,
    )
    full_count = int(full_builder.fault_zone_cell_mask.sum())
    assert full_count == 764

    restricted_builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh_restricted_activity,
        geomodel_result=model2_faulted_structural_result_restricted_activity,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1,
    )
    restricted_count = int(restricted_builder.fault_zone_cell_mask.sum())
    assert restricted_count == 596
    assert restricted_count < full_count


def test_compute_cell_materials_labels_fault_zone_cells(model2_implicit_mesh, model2_faulted_structural_result):
    builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh, geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1,
    )
    materials = builder.compute_cell_materials()
    assert "fault_zone" in set(np.unique(materials).tolist())


def test_fault_zone_region_lines_include_omega_fault(model2_implicit_mesh, model2_faulted_structural_result, tmp_path):
    builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh, geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1, include_flow=False,
    )
    out_path = str(tmp_path / "heat.py")
    builder.build_heat_input_file(out_path)
    content = open(out_path).read()
    compile(content, out_path, "exec")
    assert "Omega_fault" in content
    assert f"cells of group {builder.fault_group_id}" in content


def test_fault_zone_pressure_input_file_includes_fault_material(
    model2_implicit_mesh, model2_faulted_structural_result, tmp_path
):
    """include_flow=True + a fault zone -> build_pressure_input_file must add
    a separate 'm_fault'/Omega_fault diffusion term, not just the heat file."""
    builder = HydrothermalProblemBuilder(
        mesh_results=model2_implicit_mesh, geomodel_result=model2_faulted_structural_result,
        fault_zone_properties=_FAULT_ZONE_PROPS, fault_zone_n_voxels=1, include_flow=True,
    )
    out_path = str(tmp_path / "pressure.py")
    builder.build_pressure_input_file(out_path)
    content = open(out_path).read()
    compile(content, out_path, "exec")
    assert "m_fault" in content
    assert "dw_diffusion.i.Omega_fault(m_fault.perm_over_mu, q, p)" in content
