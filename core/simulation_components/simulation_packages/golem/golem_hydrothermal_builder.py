"""Build a GOLEM/MOOSE input file from WBGeo mesh and structural results.

Unlike the SfePy implementation, faults remain any lower-dimensional material
blocks already present in a WBGeo mesh. No neighbouring volume cells are
relabelled and no fracture-zone distance search is performed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic.dataclasses import dataclass

from core.object_components import MeshResults, MeshType, StructuralModelResults
from core.simulation_components.simulation_packages.sfepy.sfepy_hydrothermal_builder import (
    FluidProperties,
    RockUnitProperties,
)
from py_api_wbgeo.nodesapi import wbgeo_type

# Backward-compatible names. Both solvers now use the same concrete models.
GolemFluidProperties = FluidProperties
GolemRockProperties = RockUnitProperties


class GolemFaultProperties(GolemRockProperties):
    """Properties placed directly on a 2D fault element block."""
    name: str = "fault"
    porosity: float = 1.0
    permeability: float = 1.0e-10
    aperture: float = 0.05                   # GOLEM scaling_factor_initial, m


def enumerate_rock_units(model: StructuralModelResults) -> List[Tuple[int, str]]:
    units: List[Tuple[int, str]] = [(0, "basement")]
    for group in reversed(model.structural_frame.structural_groups):
        for element in reversed(group.structural_elements):
            if element.id is not None:
                units.append((int(element.id), element.name))
    return sorted(units, key=lambda item: item[0])


def _block_lithology_ids(mesh: MeshResults) -> Dict[int, int]:
    if not mesh.cell_data or "block_id" not in mesh.cell_data:
        raise ValueError("mesh_results.cell_data['block_id'] is required")
    result: Dict[int, int] = {}
    for index, values in enumerate(mesh.cell_data["block_id"]):
        unique = np.unique(values)
        if len(unique) != 1:
            raise ValueError(f"Mesh block {index} contains multiple block_id values: {unique}")
        result[index] = int(unique[0])
    return result


@wbgeo_type(name="Custom GOLEM Problem", color="#e07a5f",
            identifier="CustomGolemBuilder")
@dataclass
class CustomGolemBuilder:
    """Run caller-supplied GOLEM input text against a WBGeo mesh.

    This is the GOLEM counterpart of ``CustomSfepyBuilder``. The custom input
    must use ``${GOLEM_MESH_FILE}`` in its ``FileMeshGenerator``; the runner
    replaces that token with the exported Exodus path inside Docker.
    """

    input_file_contents: str
    mesh_results: MeshResults
    geomodel_result: Optional[StructuralModelResults] = None
    print_linear_residuals: bool = True

    def __post_init__(self) -> None:
        if not self.input_file_contents.strip():
            raise ValueError("input_file_contents must not be empty")
        if "${GOLEM_MESH_FILE}" not in self.input_file_contents:
            raise ValueError(
                "Custom GOLEM input must contain ${GOLEM_MESH_FILE} in its "
                "FileMeshGenerator block."
            )
        supported_mesh_types = {
            MeshType.IMPLICIT,
            MeshType.STRUCTURED,
            MeshType.UNSTRUCTURED,
        }
        if self.mesh_results.mesh_type not in supported_mesh_types:
            raise ValueError(
                "Custom GOLEM input requires an implicit, structured, or "
                f"unstructured WBGeo mesh; got {self.mesh_results.mesh_type!r}."
            )

    def build_input_file(self, output_path: str) -> str:
        content = self.input_file_contents
        residual_value = str(self.print_linear_residuals).lower()
        residual_pattern = re.compile(
            r"(^\s*print_linear_residuals\s*=\s*)\S+", re.MULTILINE
        )
        if residual_pattern.search(content):
            content = residual_pattern.sub(
                rf"\g<1>{residual_value}", content, count=1
            )
        else:
            outputs_pattern = re.compile(r"^\[Outputs\]\s*$", re.MULTILINE)
            if not outputs_pattern.search(content):
                raise ValueError("Custom GOLEM input must contain an [Outputs] block")
            content = outputs_pattern.sub(
                f"[Outputs]\n  print_linear_residuals = {residual_value}",
                content,
                count=1,
            )

        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        return output_path


@wbgeo_type(name="GOLEM Hydrothermal Problem", color="#e07a5f",
            identifier="GolemHydrothermalProblemBuilder")
@dataclass
class GolemHydrothermalProblemBuilder:
    """Generate one coupled thermal-hydraulic GOLEM input file.

    The Exodus block numbering is assumed to follow ``mesh_results.elements``.
    Implicit, structured and unstructured WBGeo meshes are accepted.
    Three-dimensional blocks are rock units. Any exported two-dimensional
    material blocks are faults and receive ``material_type = frac``. For an
    unstructured mesh, WBGeo's six trailing exterior-boundary blocks are not
    materials and are excluded before detecting faults.
    """

    mesh_results: MeshResults
    geomodel_result: StructuralModelResults
    rock_properties: Optional[Dict[str, GolemRockProperties]] = None
    fault_properties: Optional[Dict[str, GolemFaultProperties]] = None
    fluid: Optional[GolemFluidProperties] = None
    p_top: float = 0.0
    p_bottom: float = 1.0e6
    t_top: float = 10.0
    t_bottom: float = 60.0
    # None generates a boundary-consistent linear pressure profile. A number
    # retains support for a spatially constant pressure initial condition.
    initial_pressure: Optional[float] = None
    # None generates the same linear bottom-to-top temperature profile as
    # HydrothermalProblemBuilder. A number retains support for a constant IC.
    initial_temperature: Optional[float] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    dt: float = 86400.0
    num_steps: int = 365
    output_interval: int = 1
    print_linear_residuals: bool = True

    def __post_init__(self) -> None:
        supported_mesh_types = {
            MeshType.IMPLICIT,
            MeshType.STRUCTURED,
            MeshType.UNSTRUCTURED,
        }
        if self.mesh_results.mesh_type not in supported_mesh_types:
            raise ValueError(
                "GOLEM requires mesh_results.mesh_type to be IMPLICIT, "
                "STRUCTURED or UNSTRUCTURED; got "
                f"{self.mesh_results.mesh_type!r}."
            )
        if self.dt <= 0 or self.num_steps < 1:
            raise ValueError("dt must be > 0 and num_steps must be >= 1")
        if self.end_time is None:
            self.end_time = self.start_time + self.dt * self.num_steps
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")

        extent = self.geomodel_result.structural_frame.grid.extent
        self.z_bottom = float(extent[4])
        self.z_top = float(extent[5])
        if self.z_top <= self.z_bottom:
            raise ValueError("The structural-model z extent must have positive height")
        self.initial_temperature_gradient = (
            self.t_top - self.t_bottom
        ) / (self.z_top - self.z_bottom)

        self.fluid = self.fluid or GolemFluidProperties()
        self.rock_units = enumerate_rock_units(self.geomodel_result)
        self.rock_properties = dict(self.rock_properties or {})
        for _, name in self.rock_units:
            self.rock_properties.setdefault(name, GolemRockProperties(name=name))
        self.block_lithology = _block_lithology_ids(self.mesh_results)

        # Only unstructured WBGeo meshes append six 2D exterior-boundary
        # blocks. Fault blocks precede them and remain exported subdomains.
        # Implicit and structured meshes use all their element blocks.
        if self.mesh_results.mesh_type == MeshType.UNSTRUCTURED:
            self.material_block_count = len(self.mesh_results.elements) - 6
        else:
            self.material_block_count = len(self.mesh_results.elements)
        if self.material_block_count <= 0:
            raise ValueError("The mesh has no exportable material blocks")

        self.volume_blocks = [
            i for i, block in enumerate(self.mesh_results.elements[:self.material_block_count])
            if block.dim == 3
        ]
        self.fault_blocks = [
            i for i, block in enumerate(self.mesh_results.elements[:self.material_block_count])
            if block.dim == 2
        ]
        unsupported = [
            i for i, block in enumerate(self.mesh_results.elements[:self.material_block_count])
            if block.dim not in (2, 3)
        ]
        if unsupported:
            raise ValueError(f"Unsupported GOLEM material blocks: {unsupported}")

        supplied_fault_properties = dict(self.fault_properties or {})
        if self.fault_blocks and not supplied_fault_properties:
            raise ValueError(
                f"The mesh contains {len(self.fault_blocks)} exported 2D fault "
                "block(s), but fault_properties was not supplied. Pass one "
                "GolemFaultProperties entry per 2D fault block. Entries are "
                "assigned in dictionary insertion order."
            )
        if self.fault_blocks and len(supplied_fault_properties) != len(self.fault_blocks):
            raise ValueError(
                f"The mesh contains {len(self.fault_blocks)} exported 2D fault "
                f"block(s), but {len(supplied_fault_properties)} fault_properties "
                "entrie(s) were supplied. Pass exactly one entry per fault block."
            )
        self.fault_properties = supplied_fault_properties

    def _rock_name(self, lithology_id: int) -> str:
        for unit_id, name in self.rock_units:
            if unit_id == lithology_id:
                return name
        raise ValueError(f"Lithology id {lithology_id} is absent from the structural model")

    def _fault_property(self, ordinal: int) -> GolemFaultProperties:
        names = list(self.fault_properties)
        return self.fault_properties[names[ordinal]]

    @staticmethod
    def _material(name: str, block_id: int, props: GolemRockProperties,
                  fluid: GolemFluidProperties, fracture: bool = False) -> str:
        fracture_lines = ""
        if fracture:
            aperture = getattr(props, "aperture")
            fracture_lines = f"\n    material_type = frac\n    scaling_factor_initial = {aperture!r}"
        return f"""  [./block_{block_id}]
    type = GolemMaterialTH
    block = {block_id}{fracture_lines}
    porosity_initial = {props.porosity!r}
    porosity_uo = porosity_{block_id}
    permeability_initial = {props.permeability!r}
    permeability_uo = permeability_{block_id}
    fluid_viscosity_initial = {fluid.viscosity!r}
    fluid_viscosity_uo = fluid_viscosity_{block_id}
    fluid_density_initial = {fluid.density!r}
    fluid_density_uo = fluid_density_{block_id}
    fluid_thermal_conductivity_uo = fluid_thermal_conductivity_{block_id}
    fluid_heat_capacity_uo = fluid_heat_capacity_{block_id}
    solid_thermal_conductivity_uo = solid_thermal_conductivity_{block_id}
    solid_heat_capacity_uo = solid_heat_capacity_{block_id}
    solid_density_uo = solid_density_{block_id}
  [../]"""

    @staticmethod
    def _property_user_objects(block_id: int, props: GolemRockProperties,
                               fluid: GolemFluidProperties) -> str:
        """GOLEM constant-property objects required by the target Docker image."""
        return f"""  [./porosity_{block_id}]
    type = GolemPorosityConstant
    execute_on = 'INITIAL'
  [../]
  [./permeability_{block_id}]
    type = GolemPermeabilityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_density_{block_id}]
    type = GolemFluidDensityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_viscosity_{block_id}]
    type = GolemFluidViscosityConstant
    execute_on = 'INITIAL'
  [../]
  [./fluid_thermal_conductivity_{block_id}]
    type = GolemFluidThermalConductivityConstant
    value = {fluid.thermal_conductivity!r}
    execute_on = 'INITIAL'
  [../]
  [./fluid_heat_capacity_{block_id}]
    type = GolemFluidHeatCapacityConstant
    value = {fluid.heat_capacity!r}
    execute_on = 'INITIAL'
  [../]
  [./solid_thermal_conductivity_{block_id}]
    type = GolemSolidThermalConductivityConstant
    value = {props.thermal_conductivity!r}
    execute_on = 'INITIAL'
  [../]
  [./solid_heat_capacity_{block_id}]
    type = GolemSolidHeatCapacityConstant
    value = {props.heat_capacity!r}
    execute_on = 'INITIAL'
  [../]
  [./solid_density_{block_id}]
    type = GolemSolidDensityConstant
    value = {props.density!r}
    execute_on = 'INITIAL'
  [../]"""

    def build_input_file(self, output_path: str) -> str:
        # Both implicit and explicit structured meshes are hexahedral, but
        # their assembled systems can have different ordering, partitioning,
        # and conditioning. Preserve the settings that already work for the
        # implicit mesh and use a more robust solve for explicit meshes.
        if self.mesh_results.mesh_type == MeshType.IMPLICIT:
            coupled_flags_line = ""
            coupled_options = (
                "-ksp_type -ksp_rtol -ksp_max_it "
                "-snes_type -snes_linesearch_type "
                "-snes_atol -snes_rtol -snes_max_it"
            )
            coupled_solver = (
                "fgmres 1.0e-12 20 newtonls basic "
                "1.0e-04 1.0e-12 25"
            )
            pressure_options = (
                "-ksp_type -pc_type -sub_pc_type "
                "-sub_pc_factor_levels -ksp_rtol -ksp_max_it"
            )
            pressure_solver = "fgmres asm ilu 1 1.0e-12 20"
            temperature_options = (
                "-ksp_type -pc_type -pc_hypre_type "
                "-ksp_rtol -ksp_max_it"
            )
            temperature_solver = "preonly hypre boomeramg 1.0e-12 20"
        else:
            # GOLEM's reference HTM strategy, reduced to the H-T system:
            # Eisenstat-Walker chooses an appropriate inner KSP tolerance for
            # each Newton iteration, pressure uses AMG, and temperature uses
            # overlapping ASM/ILU. This avoids both early oversolving and a
            # fixed, excessively strict inner tolerance on explicit meshes.
            coupled_flags_line = "      petsc_options = '-snes_ksp_ew'\n"
            coupled_options = (
                "-ksp_type -ksp_rtol -ksp_max_it "
                "-snes_type -snes_linesearch_type "
                "-snes_atol -snes_stol -snes_max_it"
            )
            coupled_solver = (
                "fgmres 1.0e-12 50 newtonls basic "
                "1.0e-04 0 100"
            )
            pressure_options = (
                "-ksp_type -pc_type -pc_hypre_type "
                "-ksp_rtol -ksp_max_it"
            )
            pressure_solver = "preonly hypre boomeramg 1.0e-04 500"
            temperature_options = (
                "-ksp_type -pc_type -sub_pc_type "
                "-sub_pc_factor_levels -ksp_rtol -ksp_max_it"
            )
            temperature_solver = "fgmres asm ilu 1 1.0e-04 500"

        # Use semantic node-set names for every mesh type instead of assuming
        # that numeric Exodus side-set IDs have the same ordering in all WBGeo
        # exporters. The verified explicit-structured file contains named
        # front/back/bottom/top/left/right node sets, as do the working meshes.
        boundary_generator = """  [create_sideset]
    type = SideSetsFromNodeSetsGenerator
    input = input_mesh
  []"""

        materials = []
        property_user_objects = []
        for block_index in self.volume_blocks:
            # mesh_results.elements is zero-based, but the Exodus writer and
            # therefore GOLEM expose element block IDs starting at 1.
            mesh_block_id = block_index + 1
            lith_id = self.block_lithology[block_index]
            props = self.rock_properties[self._rock_name(lith_id)]
            materials.append(self._material(props.name, mesh_block_id, props, self.fluid))
            property_user_objects.append(
                self._property_user_objects(mesh_block_id, props, self.fluid)
            )
        for ordinal, block_index in enumerate(self.fault_blocks):
            mesh_block_id = block_index + 1
            props = self._fault_property(ordinal)
            materials.append(
                self._material(props.name, mesh_block_id, props, self.fluid, fracture=True)
            )
            property_user_objects.append(
                self._property_user_objects(mesh_block_id, props, self.fluid)
            )

        function_definitions = []

        if self.initial_pressure is None:
            pressure_ic = """    [InitialCondition]
      type = FunctionIC
      function = initial_pressure
    []"""
            pressure_gradient = (
                self.p_top - self.p_bottom
            ) / (self.z_top - self.z_bottom)
            function_definitions.append(f"""  [initial_pressure]
    type = ParsedFunction
    expression = 'gradient*(z-z_bottom)+pressure_bottom'
    symbol_names = 'gradient z_bottom pressure_bottom'
    symbol_values = '{pressure_gradient!r} {self.z_bottom!r} {self.p_bottom!r}'
  []""")
        else:
            pressure_ic = f"    initial_condition = {self.initial_pressure!r}"

        if self.initial_temperature is None:
            temperature_ic = """    [InitialCondition]
      type = FunctionIC
      function = initial_temperature
    []"""
            function_definitions.append(f"""  [initial_temperature]
    type = ParsedFunction
    expression = 'gradient*(z-z_bottom)+temperature_bottom'
    symbol_names = 'gradient z_bottom temperature_bottom'
    symbol_values = '{self.initial_temperature_gradient!r} {self.z_bottom!r} {self.t_bottom!r}'
  []""")
        else:
            temperature_ic = f"    initial_condition = {self.initial_temperature!r}"

        if function_definitions:
            functions_block = "\n[Functions]\n" + "\n".join(function_definitions) + "\n[]\n"
        else:
            functions_block = ""

        content = f"""# Auto-generated by GolemHydrothermalProblemBuilder.
# Faults are the mesh's actual 2D blocks; no volume-cell damage zone is used.
[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${{GOLEM_MESH_FILE}}
  []
{boundary_generator}
[]

[GlobalParams]
  pore_pressure = pore_pressure
  temperature = temperature
  scaling_uo = scaling
[]

[Variables]
  [./pore_pressure]
    order = FIRST
    family = LAGRANGE
{pressure_ic}
  [../]
  [./temperature]
    order = FIRST
    family = LAGRANGE
{temperature_ic}
  [../]
[]

[Kernels]
  [./HKernel]
    type = GolemKernelH
    variable = pore_pressure
  [../]
  [./temp_time]
    type = GolemKernelTimeT
    variable = temperature
  [../]
  [./THKernel]
    type = GolemKernelTH
    variable = temperature
    is_conservative = true
  [../]
[]

[AuxVariables]
  [./vx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vz]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./darcyx]
    type = GolemDarcyVelocity
    variable = vx
    component = 0
  [../]
  [./darcyy]
    type = GolemDarcyVelocity
    variable = vy
    component = 1
  [../]
  [./darcyz]
    type = GolemDarcyVelocity
    variable = vz
    component = 2
  [../]
[]

{functions_block}

[BCs]
  [pressure_bottom]
    type = DirichletBC
    variable = pore_pressure
    boundary = bottom
    value = {self.p_bottom!r}
  []
  [pressure_top]
    type = DirichletBC
    variable = pore_pressure
    boundary = top
    value = {self.p_top!r}
  []
  [temperature_bottom]
    type = DirichletBC
    variable = temperature
    boundary = bottom
    value = {self.t_bottom!r}
  []
  [temperature_top]
    type = DirichletBC
    variable = temperature
    boundary = top
    value = {self.t_top!r}
  []
[]

[Materials]
{chr(10).join(materials)}
[]

[UserObjects]
  [./scaling]
    type = GolemScaling
    execute_on = 'INITIAL'
  [../]
{chr(10).join(property_user_objects)}
[]

[Preconditioning]
  active = 'fieldsplit'
  [./fieldsplit]
    type = FSP
    topsplit = pT
    [./pT]
      splitting = 'p T'
      splitting_type = multiplicative
{coupled_flags_line}      petsc_options_iname = '{coupled_options}'
      petsc_options_value = '{coupled_solver}'
    [../]
    [./p]
      vars = 'pore_pressure'
      petsc_options_iname = '{pressure_options}'
      petsc_options_value = '{pressure_solver}'
    [../]
    [./T]
      vars = 'temperature'
      petsc_options_iname = '{temperature_options}'
      petsc_options_value = '{temperature_solver}'
    [../]
  [../]
[]

[Executioner]
  type = Transient
  solve_type = Newton
  automatic_scaling = true
  start_time = {self.start_time!r}
  end_time = {self.end_time!r}
  dt = {self.dt!r}
  num_steps = {self.num_steps!r}
[]

[Outputs]
  time_step_interval = {self.output_interval!r}
  print_linear_residuals = {str(self.print_linear_residuals).lower()}
  perf_graph = true
  [out]
    type = Exodus
    file_base = 'output/wbgeo'
  []
[]
"""
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        return output_path
