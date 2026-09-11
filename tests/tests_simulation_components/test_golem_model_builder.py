import numpy as np
import pytest
from meshio import CellBlock

import core.simulation_components.simulation_packages.golem.golem_model_builder as golem_builder
from core.object_components import MeshType
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
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
# CustomGolemBuilder -- text handling only, no GOLEM/Docker
# -----------------------------------------------------------------------------

def test_custom_golem_builder_rejects_missing_mesh_placeholder(model1_implicit_mesh):
    with pytest.raises(ValueError, match="GOLEM_MESH_FILE"):
        CustomGolemBuilder(
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
        CustomGolemBuilder(input_file_contents=content, mesh_results=mesh)


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
    builder = CustomGolemBuilder(
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
    builder = CustomGolemBuilder(
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
        num_steps=2,
        dt=123.0,
    )
    path = tmp_path / "hydro.i"
    builder.build_input_file(str(path))
    generated = path.read_text()

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

    # Current implicit-mesh solver strategy.
    assert "fgmres asm ilu 1 1.0e-12 20" in generated
    assert "preonly hypre boomeramg 1.0e-12 20" in generated

    # Test the validator on a real builder product.
    validate_golem_input(
        generated, model1_implicit_mesh, model1_implicit_mesh.mesh_type
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
    assert "preonly hypre boomeramg 1.0e-04 500" in generated
    assert "fgmres asm ilu 1 1.0e-04 500" in generated


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
