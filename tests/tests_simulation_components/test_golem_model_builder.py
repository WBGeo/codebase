import numpy as np
import pytest
from meshio import CellBlock

import core.simulation_components.simulation_packages.golem.golem_model_builder as golem_builder
from core.object_components import MeshType
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    SfepyHydrothermalProblemBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    GolemCustomBuilder,
    GolemFaultProperties,
    GolemHydrothermalProblemBuilder,
    GolemSimpleReservoirProblemBuilder,
)
from core.simulation_components.simulation_packages.golem.golem_run import (
    validate_golem_input,
)


def _boundary_triangle():
    return CellBlock("triangle", np.array([[0, 1, 2]], dtype=int))


def _synthetic_hydrothermal_unstructured_mesh(base_mesh, *, with_fault=True):
    """Synthetic mesh valid for GolemHydrothermalProblemBuilder.

    Hydrothermal GOLEM accepts only exported 3D rock blocks and 2D fault
    blocks. In particular it must not contain the 1D well block that belongs
    to the simple-reservoir benchmark layout. The final six triangle blocks
    represent WBGeo's exterior boundaries and are excluded from materials.
    """
    nodes = np.asarray(base_mesh.nodes)
    if len(nodes) < 4:
        pytest.skip("base mesh fixture does not contain enough nodes")

    material_blocks = [
        CellBlock("tetra", np.array([[0, 1, 2, 3]], dtype=int)),
    ]
    block_ids = [np.array([0], dtype=int)]

    if with_fault:
        material_blocks.append(
            CellBlock("triangle", np.array([[0, 1, 2]], dtype=int))
        )
        block_ids.append(np.array([-1], dtype=int))

    boundaries = [_boundary_triangle() for _ in range(6)]
    block_ids.extend([np.array([-1], dtype=int) for _ in range(6)])

    return base_mesh.model_copy(
        update={
            "mesh_type": MeshType.UNSTRUCTURED,
            "elements": material_blocks + boundaries,
            "cell_data": {"block_id": block_ids},
        }
    )


def _synthetic_unstructured_mesh(base_mesh, *, include_geological_fault=False):
    """Small synthetic material layout used without invoking Gmsh.

    Ordering matches GolemSimpleReservoirProblemBuilder's documented contract:
    3D volume(s), optional geological fault(s), two synthetic fractures,
    one well, then six exterior-boundary blocks.
    """
    nodes = np.asarray(base_mesh.nodes)
    if len(nodes) < 4:
        pytest.skip("base mesh fixture does not contain enough nodes")

    material_blocks = [
        CellBlock("tetra", np.array([[0, 1, 2, 3]], dtype=int)),
    ]
    block_ids = [np.array([0], dtype=int)]

    if include_geological_fault:
        material_blocks.append(
            CellBlock("triangle", np.array([[0, 1, 2]], dtype=int))
        )
        block_ids.append(np.array([-1], dtype=int))

    # Two internally-created fracture planes.
    material_blocks.extend([
        CellBlock("triangle", np.array([[0, 1, 2]], dtype=int)),
        CellBlock("triangle", np.array([[0, 2, 3]], dtype=int)),
        CellBlock("line", np.array([[0, 1]], dtype=int)),
    ])
    block_ids.extend([
        np.array([-1], dtype=int),
        np.array([-1], dtype=int),
        np.array([-1], dtype=int),
    ])

    boundaries = [_boundary_triangle() for _ in range(6)]
    block_ids.extend([np.array([-1], dtype=int) for _ in range(6)])

    return base_mesh.model_copy(
        update={
            "mesh_type": MeshType.UNSTRUCTURED,
            "elements": material_blocks + boundaries,
            "cell_data": {"block_id": block_ids},
        }
    )


# -----------------------------------------------------------------------------
# GolemCustomBuilder -- text handling only, no GOLEM/Docker
# -----------------------------------------------------------------------------

def test_custom_golem_builder_rejects_missing_mesh_placeholder(model1_implicit_mesh):
    with pytest.raises(ValueError, match="GOLEM_MESH_FILE"):
        GolemCustomBuilder(
            input_file_contents="[Mesh]\n[]\n[Outputs]\n[]\n",
            mesh_results=model1_implicit_mesh,
        )


def test_custom_golem_builder_rejects_unknown_mesh_type(model1_implicit_mesh):
    mesh = model1_implicit_mesh.model_copy(update={"mesh_type": None})
    content = (
        "[Mesh]\n"
        "  type = FileMeshGenerator\n"
        "  file = ${GOLEM_MESH_FILE}\n"
        "[]\n"
        "[Outputs]\n[]\n"
    )
    with pytest.raises(ValueError, match="implicit, structured, or unstructured"):
        GolemCustomBuilder(input_file_contents=content, mesh_results=mesh)


def test_custom_golem_builder_updates_existing_residual_flag(
    model1_implicit_mesh, tmp_path
):
    content = (
        "[Mesh]\n"
        "  type = FileMeshGenerator\n"
        "  file = ${GOLEM_MESH_FILE}\n"
        "[]\n"
        "[Outputs]\n"
        "  print_linear_residuals = true\n"
        "[]\n"
    )
    builder = GolemCustomBuilder(
        input_file_contents=content,
        mesh_results=model1_implicit_mesh,
        print_linear_residuals=False,
    )
    path = tmp_path / "custom.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "print_linear_residuals = false" in generated
    assert generated.count("print_linear_residuals") == 1


def test_custom_golem_builder_injects_residual_flag_into_outputs(
    model1_implicit_mesh, tmp_path
):
    content = (
        "[Mesh]\n"
        "  type = FileMeshGenerator\n"
        "  file = ${GOLEM_MESH_FILE}\n"
        "[]\n"
        "[Outputs]\n"
        "[]\n"
    )
    builder = GolemCustomBuilder(
        input_file_contents=content,
        mesh_results=model1_implicit_mesh,
        print_linear_residuals=True,
    )
    path = tmp_path / "custom.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "[Outputs]\n  print_linear_residuals = true" in generated


# -----------------------------------------------------------------------------
# GolemHydrothermalProblemBuilder -- generated MOOSE text, no solve
# -----------------------------------------------------------------------------

def test_hydrothermal_builder_generates_valid_input(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        end_time=246.0,
        num_steps=2,
    )
    path = tmp_path / "hydro.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    # Step size is derived, so end_time, dt and num_steps can't disagree.
    assert builder.dt == pytest.approx(123.0)
    assert "dt = 123.0" in generated
    assert "end_time = 246.0" in generated
    assert "num_steps = 2" in generated

    # Important top-level structure and mesh placeholder.
    for block in (
        "[Mesh]",
        "[Variables]",
        "[Kernels]",
        "[BCs]",
        "[Materials]",
        "[Executioner]",
        "[Outputs]",
    ):
        assert block in generated
    assert "file = ${GOLEM_MESH_FILE}" in generated

    # Current default IC behaviour: boundary-consistent parsed functions.
    assert "[initial_pressure]" in generated
    assert "[initial_temperature]" in generated
    assert "type = FunctionIC" in generated

    # Implicit meshes use the same solver strategy as explicit ones.
    assert "petsc_options = '-snes_ksp_ew'" in generated
    assert "petsc_options_value = 'preonly hypre boomeramg'" in generated
    assert "fgmres asm ilu 1 1.0e-04 500" in generated
    assert "1.0e-12 20" not in generated

    # Test the validator on a real builder product.
    validate_golem_input(
        generated, model1_implicit_mesh, model1_implicit_mesh.mesh_type
    )


def test_hydrothermal_builder_time_defaults_match_sfepy(
    model1_implicit_mesh, model1_structural_result
):
    """Same time-control names and defaults as the SfePy hydrothermal builder:
    end_time=None resolves to the model's diffusion timescale."""
    golem = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
    )
    sfepy = SfepyHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        linear_solver="iterative",
    )
    assert golem.start_time == sfepy.start_time == 0.0
    assert golem.num_steps == sfepy.num_steps == 3
    assert golem.end_time == pytest.approx(sfepy.end_time)
    assert golem.dt == pytest.approx(golem.end_time / 3)
    assert golem.print_linear_residuals is sfepy.print_linear_residuals is False


def test_hydrothermal_builder_rejects_end_time_before_start_time(
    model1_implicit_mesh, model1_structural_result
):
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        GolemHydrothermalProblemBuilder(
            mesh_results=model1_implicit_mesh,
            geomodel_result=model1_structural_result,
            start_time=10.0,
            end_time=5.0,
        )


def test_hydrothermal_builder_uses_constant_initial_conditions_when_requested(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=model1_implicit_mesh,
        geomodel_result=model1_structural_result,
        initial_pressure=12.5,
        initial_temperature=34.5,
    )
    path = tmp_path / "constant_ic.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "initial_condition = 12.5" in generated
    assert "initial_condition = 34.5" in generated
    assert "[initial_pressure]" not in generated
    assert "[initial_temperature]" not in generated


def test_hydrothermal_builder_structured_mesh_uses_robust_explicit_solver_settings(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    structured_mesh = model1_implicit_mesh.model_copy(
        update={"mesh_type": MeshType.STRUCTURED}
    )
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=structured_mesh,
        geomodel_result=model1_structural_result,
    )
    path = tmp_path / "structured.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "petsc_options = '-snes_ksp_ew'" in generated
    assert "petsc_options_value = 'preonly hypre boomeramg'" in generated
    assert "fgmres asm ilu 1 1.0e-04 500" in generated


@pytest.mark.parametrize("mesh_type", [MeshType.IMPLICIT, MeshType.STRUCTURED])
def test_hydrothermal_builder_executioner_has_absolute_tolerance_and_dtmin(
    model1_implicit_mesh, model1_structural_result, tmp_path, mesh_type
):
    """Without nl_abs_tol a step starting near steady state can't meet MOOSE's
    purely relative criterion and dt is halved indefinitely (seen on the
    unstructured model 1 mesh with the default end_time); dtmin bounds that."""
    mesh = model1_implicit_mesh.model_copy(update={"mesh_type": mesh_type})
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=mesh,
        geomodel_result=model1_structural_result,
        end_time=3000.0,
        num_steps=3,
    )
    path = tmp_path / "executioner.i"
    builder.build_input_file(str(path))
    generated = path.read_text()
    executioner = generated.split("[Executioner]", 1)[1].split("[]", 1)[0]

    assert "nl_abs_tol = 1e-08" in executioner
    assert "dt = 1000.0" in executioner
    assert "dtmin = 1.0" in executioner


def test_hydrothermal_builder_fault_blocks_require_fault_properties(
    model1_implicit_mesh, model1_structural_result
):
    mesh = _synthetic_hydrothermal_unstructured_mesh(
        model1_implicit_mesh, with_fault=True
    )

    # For GolemHydrothermalProblemBuilder the two synthetic triangles above are
    # simply exported 2D fault blocks too, so every exported 2D block needs a
    # fault-properties entry/template.
    with pytest.raises(ValueError, match="fault_properties was not supplied"):
        GolemHydrothermalProblemBuilder(
            mesh_results=mesh,
            geomodel_result=model1_structural_result,
        )


def test_hydrothermal_builder_fault_input_uses_frac_and_supg(
    model1_implicit_mesh, model1_structural_result, tmp_path
):
    mesh = _synthetic_hydrothermal_unstructured_mesh(
        model1_implicit_mesh, with_fault=True
    )
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=mesh,
        geomodel_result=model1_structural_result,
        fault_properties=GolemFaultProperties(name="fault", aperture=0.02),
    )
    path = tmp_path / "faulted.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "material_type = FRAC" in generated
    assert "fracture_aperture_uo" in generated
    assert "type = GolemMaterialTHSUPG" in generated
    assert "type = GolemKernelTimeTSUPG" in generated
    assert "type = GolemKernelTHSUPG" in generated


# -----------------------------------------------------------------------------
# GolemSimpleReservoirProblemBuilder -- mock Gmsh/mesh creation completely
# -----------------------------------------------------------------------------

def test_simple_reservoir_builder_scales_classifies_and_generates_input_without_meshing(
    monkeypatch, model1_implicit_mesh, model1_structural_result, tmp_path
):
    fake_mesh = _synthetic_unstructured_mesh(
        model1_implicit_mesh, include_geological_fault=True
    )
    received = {}

    def fake_create_unstructured_mesh_data(**kwargs):
        received.update(kwargs)
        return fake_mesh

    monkeypatch.setattr(
        golem_builder,
        "create_unstructured_mesh_data",
        fake_create_unstructured_mesh_data,
    )

    builder = GolemSimpleReservoirProblemBuilder(
        geomodel_result=model1_structural_result,
        mesh_size=25.0,
        curve_mesh_size=4.0,
        time_sequence=[10.0, 20.0, 50.0],
    )

    assert builder.mesh_results is fake_mesh
    assert received["mesh_size"] == 25.0
    assert received["curve_mesh_size"] == 4.0
    assert received["extra_planes"] == builder.fracture_planes
    assert received["wells"] == builder.wells

    # Layout: block 1 = volume, block 2 = geological fault,
    # blocks 3/4 = two synthetic fractures, block 5 = well.
    assert builder.volume_blocks == [1]
    assert builder.geological_fault_blocks == [2]
    assert builder.fracture_blocks == [3, 4]
    assert builder.well_blocks == [5]
    assert builder.reservoir_property_blocks == [1, 2]
    assert builder.end_time == 50.0
    assert builder.source_end_time == 50.0

    path = tmp_path / "reservoir.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

    assert "block = '1'" in generated
    assert "[geological_faults]" in generated
    assert "block = '2'" in generated
    assert "block = '3 4'" in generated
    assert "block = '5'" in generated
    assert "time_sequence = '10 20 50'" in generated
    assert "file = ${GOLEM_MESH_FILE}" in generated

    validate_golem_input(generated, fake_mesh, MeshType.UNSTRUCTURED)

    mesh_recipe_path = tmp_path / "model1_generated_reservoir_mesh.py"
    returned_path = builder.build_mesh_generation_file(
        str(mesh_recipe_path),
        mesh_variable_name="mesh_model1_simple_reservoir",
    )
    assert returned_path == str(mesh_recipe_path)

    mesh_recipe = mesh_recipe_path.read_text()
    assert "reservoir_extent =" in mesh_recipe
    assert "reservoir_fracture_planes = [" in mesh_recipe
    assert "reservoir_wells = [" in mesh_recipe
    assert "reservoir_refinement = Refinement(" in mesh_recipe
    assert "mesh_model1_simple_reservoir = create_unstructured_mesh_data(" in mesh_recipe
    assert "geomodel_result=structural_model_result" in mesh_recipe
    assert "mesh_size=25.0" in mesh_recipe
    assert "curve_mesh_size=4.0" in mesh_recipe

    # The generated recipe itself must be valid Python.
    compile(mesh_recipe, str(mesh_recipe_path), "exec")


@pytest.mark.parametrize(
    "time_sequence, message",
    [
        ([], "at least one"),
        ([0.0, 1.0], "greater than start_time"),
        ([2.0, 1.0], "strictly increasing"),
    ],
)
def test_simple_reservoir_time_control_validation(
    monkeypatch,
    model1_implicit_mesh,
    model1_structural_result,
    time_sequence,
    message,
):
    fake_mesh = _synthetic_unstructured_mesh(model1_implicit_mesh)

    monkeypatch.setattr(
        golem_builder,
        "create_unstructured_mesh_data",
        lambda **kwargs: fake_mesh,
    )

    with pytest.raises(ValueError, match=message):
        GolemSimpleReservoirProblemBuilder(
            geomodel_result=model1_structural_result,
            time_sequence=time_sequence,
        )


def test_simple_reservoir_rejects_nonpositive_physical_input(
    model1_structural_result,
):
    # Validation happens before any mesh generation, so this test cannot invoke Gmsh.
    with pytest.raises(ValueError, match="rock_permeability"):
        GolemSimpleReservoirProblemBuilder(
            geomodel_result=model1_structural_result,
            rock_permeability=0.0,
        )

# -----------------------------------------------------------------------------
# GolemCustomBuilder -- hierarchical mesh placeholders
# -----------------------------------------------------------------------------

def _custom_golem_input(mesh_placeholder):
    return (
        "[Mesh]\n"
        "  type = FileMeshGenerator\n"
        f"  file = {mesh_placeholder}\n"
        "[]\n"
        "[Variables]\n  [pore_pressure]\n  []\n[]\n"
        "[Kernels]\n  [k]\n    variable = pore_pressure\n  []\n[]\n"
        "[Executioner]\n  type = Transient\n[]\n"
        "[Outputs]\n[]\n"
        "[BCs]\n  [dummy]\n    variable = pore_pressure\n    boundary = top\n    value = 0\n  []\n[]\n"
        "[Materials]\n  [m]\n    block = '1 2 3'\n  []\n[]\n"
    )


def test_custom_golem_builder_accepts_hierarchical_mesh1(model1_implicit_mesh):
    builder = GolemCustomBuilder(
        input_file_contents=_custom_golem_input("${GOLEM_MESH1_FILE}"),
        mesh1=model1_implicit_mesh,
    )
    assert builder.referenced_meshes() == {
        "${GOLEM_MESH1_FILE}": model1_implicit_mesh
    }


def test_custom_golem_builder_accepts_hierarchical_mesh2(model1_implicit_mesh):
    builder = GolemCustomBuilder(
        input_file_contents=_custom_golem_input("${GOLEM_MESH2_FILE}"),
        mesh2=model1_implicit_mesh,
    )
    assert builder.referenced_meshes() == {
        "${GOLEM_MESH2_FILE}": model1_implicit_mesh
    }


def test_custom_golem_builder_rejects_unconnected_hierarchical_mesh(
    model1_implicit_mesh,
):
    with pytest.raises(ValueError, match=r"connected mesh.*GOLEM_MESH2_FILE"):
        GolemCustomBuilder(
            input_file_contents=_custom_golem_input("${GOLEM_MESH2_FILE}"),
            mesh1=model1_implicit_mesh,
        )


def test_custom_golem_builder_rejects_multiple_mesh_sources(model1_implicit_mesh):
    content = _custom_golem_input("${GOLEM_MESH1_FILE}")
    with pytest.raises(ValueError, match="exactly one of mesh_results, mesh1, mesh2, or mesh_file"):
        GolemCustomBuilder(
            input_file_contents=content,
            mesh1=model1_implicit_mesh,
            mesh2=model1_implicit_mesh,
        )


def test_custom_golem_builder_accepts_external_mesh_file(tmp_path):
    mesh_path = tmp_path / "existing_mesh.exo"
    mesh_path.write_bytes(b"existing-exodus")
    builder = GolemCustomBuilder(
        input_file_contents=_custom_golem_input("${GOLEM_MESH_FILE}"),
        mesh_file=str(mesh_path),
    )
    assert builder.referenced_meshes() == {}
    placeholder, source, staged_name = builder.referenced_mesh_file()
    assert placeholder == "${GOLEM_MESH_FILE}"
    assert source == mesh_path.resolve()
    assert staged_name == "existing_mesh.exo"


def test_custom_golem_builder_accepts_external_mesh_file_name(tmp_path):
    mesh_path = tmp_path / "existing_mesh.exo"
    mesh_path.write_bytes(b"existing-exodus")
    builder = GolemCustomBuilder(
        input_file_contents=_custom_golem_input("${GOLEM_MESH_FILE}"),
        mesh_file=str(mesh_path),
        mesh_file_name="reservoir.exo",
    )
    assert builder.referenced_mesh_file()[2] == "reservoir.exo"


def test_custom_golem_builder_rejects_missing_external_mesh_file(tmp_path):
    with pytest.raises(ValueError, match="mesh_file does not exist"):
        GolemCustomBuilder(
            input_file_contents=_custom_golem_input("${GOLEM_MESH_FILE}"),
            mesh_file=str(tmp_path / "missing.exo"),
        )


def test_custom_golem_builder_rejects_external_mesh_with_hierarchy_placeholder(tmp_path):
    mesh_path = tmp_path / "existing_mesh.exo"
    mesh_path.write_bytes(b"existing-exodus")
    with pytest.raises(ValueError, match="without a connected mesh"):
        GolemCustomBuilder(
            input_file_contents=_custom_golem_input("${GOLEM_MESH1_FILE}"),
            mesh_file=str(mesh_path),
        )


# G.3: pressure/temperature IC gradients must follow actual mesh node Z range.
@pytest.mark.parametrize("mesh_type", [MeshType.IMPLICIT, MeshType.STRUCTURED])
def test_hydrothermal_gradient_uses_actual_mesh_nodes(
    mesh_type, model1_implicit_mesh, model1_structural_result, tmp_path
):
    nodes = np.asarray(model1_implicit_mesh.nodes, dtype=float).copy()
    # Make the mesh range deliberately differ from the structural grid extent.
    nodes[:, 2] = 100.0 + (nodes[:, 2] - nodes[:, 2].min()) * 0.5
    mesh = model1_implicit_mesh.model_copy(update={"mesh_type": mesh_type, "nodes": nodes})
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=mesh, geomodel_result=model1_structural_result,
        p_bottom=1.0e6, p_top=0.0, t_bottom=60.0, t_top=10.0,
    )
    assert builder.z_bottom == pytest.approx(nodes[:, 2].min())
    assert builder.z_top == pytest.approx(nodes[:, 2].max())
    assert builder.initial_temperature_gradient == pytest.approx(
        (builder.t_top - builder.t_bottom) / np.ptp(nodes[:, 2])
    )
    path = tmp_path / "gradient.i"
    builder.build_input_file(str(path))
    content = path.read_text()
    assert "[initial_pressure]" in content
    assert "[initial_temperature]" in content


def test_hydrothermal_unstructured_gradient_keeps_structural_extent(
    model1_implicit_mesh, model1_structural_result
):
    mesh = _synthetic_hydrothermal_unstructured_mesh(
        model1_implicit_mesh, with_fault=False
    )
    builder = GolemHydrothermalProblemBuilder(
        mesh_results=mesh, geomodel_result=model1_structural_result
    )
    extent = model1_structural_result.structural_frame.grid.extent
    assert builder.z_bottom == pytest.approx(extent[4])
    assert builder.z_top == pytest.approx(extent[5])
