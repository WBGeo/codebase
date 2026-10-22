"""Build GOLEM/MOOSE model input files from WBGeo mesh and structural results.

Unlike the SfePy implementation, faults remain any lower-dimensional material
blocks already present in a WBGeo mesh. No neighbouring volume cells are
relabelled and no fracture-zone distance search is performed.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic.dataclasses import dataclass

from core.object_components import MeshResults, MeshType, StructuralModelResults
from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement,
    LinearWellRefinement,
    ExtraPlaneRefinement,
)
from core.simulation_components.simulation_packages.sfepy.sfepy_model_builder import (
    FluidProperties,
    RockUnitProperties,
)
from py_api_wbgeo.nodesapi import wbgeo_type
from core.simulation_components.simulation_packages.golem.golem_input_parser import (
    locate_blocks, top_block,
)

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
    # Optional path of the caller's original .i file. When hierarchy sampling
    # is injected, the runner saves a readable augmented copy beside this file.
    input_file_path: Optional[str] = None
    # Optional directory containing auxiliary files referenced by the GOLEM input.
    # The runner stages this directory as `files/` beside wbgeo.i in Docker.
    # For mesh2 hierarchy-consumer runs, golem_run automatically prepares
    # hierarchy_<group>.csv for PropertyReadFile and resolves __N_<GROUP>__
    # (and optional __NPROP_<GROUP>__) placeholders in the staged input.
    auxiliary_files_dir: Optional[str] = None
    mesh_results: Optional[MeshResults] = None
    mesh1: Optional[MeshResults] = None
    mesh2: Optional[MeshResults] = None
    # Existing mesh file mode: use an already-created mesh file directly.
    mesh_file: Optional[str] = None
    mesh_file_name: Optional[str] = None
    geomodel_result: Optional[StructuralModelResults] = None
    print_linear_residuals: bool = True
    hierarchy_sample_nodes: Optional[Dict[str, Any]] = None
    sample_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not self.input_file_contents.strip():
            raise ValueError("input_file_contents must not be empty")

        if self.auxiliary_files_dir is not None:
            from pathlib import Path
            auxiliary_dir = Path(self.auxiliary_files_dir).expanduser()
            if not auxiliary_dir.is_dir():
                raise ValueError(
                    "auxiliary_files_dir does not exist or is not a directory: "
                    f"{auxiliary_dir}"
                )

        # A custom run has exactly one mesh source. mesh_file is the file-based
        # counterpart of mesh_results and therefore uses GOLEM_MESH_FILE.
        supplied = [
            self.mesh_results is not None,
            self.mesh1 is not None,
            self.mesh2 is not None,
            self.mesh_file is not None,
        ]
        if sum(supplied) != 1:
            raise ValueError(
                "CustomGolemBuilder requires exactly one of mesh_results, mesh1, "
                "mesh2, or mesh_file."
            )

        placeholders = {
            "${GOLEM_MESH_FILE}": self.mesh_results if self.mesh_results is not None else self.mesh_file,
            "${GOLEM_MESH1_FILE}": self.mesh1,
            "${GOLEM_MESH2_FILE}": self.mesh2,
        }
        referenced = [token for token in placeholders if token in self.input_file_contents]
        if not referenced:
            raise ValueError(
                "Custom GOLEM input must contain ${GOLEM_MESH_FILE}, "
                "${GOLEM_MESH1_FILE}, or ${GOLEM_MESH2_FILE} in a "
                "FileMeshGenerator block."
            )
        if len(referenced) != 1:
            raise ValueError("Custom GOLEM input must reference exactly one mesh placeholder.")

        missing = [token for token in referenced if placeholders[token] is None]
        if missing:
            raise ValueError(
                "Custom GOLEM input references mesh placeholder(s) without a "
                "connected mesh: " + ", ".join(missing)
            )

        if self.mesh_file is not None:
            from pathlib import Path
            mesh_path = Path(self.mesh_file).expanduser()
            if not mesh_path.is_file():
                raise ValueError(f"mesh_file does not exist or is not a file: {mesh_path}")
            if referenced != ["${GOLEM_MESH_FILE}"]:
                raise ValueError("mesh_file must be used with ${GOLEM_MESH_FILE}.")
            if self.mesh_file_name is not None:
                name = Path(self.mesh_file_name).name
                if not name or name != self.mesh_file_name:
                    raise ValueError("mesh_file_name must be a file name, not a path.")

        supported_mesh_types = {
            MeshType.IMPLICIT,
            MeshType.STRUCTURED,
            MeshType.UNSTRUCTURED,
        }
        for mesh in (self.mesh_results, self.mesh1, self.mesh2):
            if mesh is not None and mesh.mesh_type not in supported_mesh_types:
                raise ValueError(
                    "Custom GOLEM input requires implicit, structured, or "
                    f"unstructured WBGeo meshes; got {mesh.mesh_type!r}."
                )

    def referenced_meshes(self):
        """Return placeholder -> MeshResults for meshes used by the custom input."""
        candidates = {
            "${GOLEM_MESH_FILE}": self.mesh_results,
            "${GOLEM_MESH1_FILE}": self.mesh1,
            "${GOLEM_MESH2_FILE}": self.mesh2,
        }
        return {
            token: mesh for token, mesh in candidates.items()
            if mesh is not None and token in self.input_file_contents
        }

    def referenced_mesh_file(self):
        """Return (placeholder, path, staging name) for external-file mode."""
        if self.mesh_file is None or "${GOLEM_MESH_FILE}" not in self.input_file_contents:
            return None
        from pathlib import Path
        source = Path(self.mesh_file).expanduser().resolve()
        name = self.mesh_file_name or source.name
        return "${GOLEM_MESH_FILE}", source, name

    def _hierarchy_sampling_groups(self) -> Dict[str, np.ndarray]:
        """Return ordered mesh1-frame point groups for GOLEM sampling."""
        data = self.hierarchy_sample_nodes
        if not data:
            return {}
        groups: Dict[str, np.ndarray] = {}
        inside = data.get("inside_nodes", {}).get("big_frame")
        if inside is not None:
            groups["inside"] = np.asarray(inside, dtype=float)
        for face, values in data.get("big_frame", {}).items():
            groups[str(face)] = np.asarray(values, dtype=float)
        for name, arr in groups.items():
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(
                    f"Hierarchy sampling group {name!r} must have columns "
                    "[NodeID, X, Y, Z]."
                )
        return groups

    def _inject_hierarchy_samplers(self, content: str) -> str:
        groups = self._hierarchy_sampling_groups()
        if not groups:
            return content
        variables = self.sample_variables or ["pore_pressure", "temp"]
        children = []
        for name, arr in groups.items():
            if arr.shape[0] == 0:
                continue
            points = "\n".join(
                f"      {x:.17g} {y:.17g} {z:.17g}"
                for x, y, z in arr[:, 1:4]
            )
            children.append(
                f"  [hierarchy_{name}]\n"
                f"    type = PointValueSampler\n"
                f"    variable = '{' '.join(variables)}'\n"
                f"    points = '\n{points}\n    '\n"
                # Required by PointValueSampler in the GOLEM/MOOSE build.
                f"    sort_by = id\n"
                f"    execute_on = TIMESTEP_END\n"
                f"  []"
            )
        if not children:
            return content
        child_text = "\n".join(children)
        # Locate insertion points with the shared MOOSE block parser.
        lines, vector_block = top_block(content, "VectorPostprocessors")
        if vector_block is not None:
            lines.insert(vector_block.end, child_text + "\n")
            content = "".join(lines)
        else:
            block = "[VectorPostprocessors]\n" + child_text + "\n[]\n\n"
            lines, executioner = top_block(content, "Executioner")
            if executioner is not None:
                lines.insert(executioner.start, block)
                content = "".join(lines)
            else:
                content += "\n" + block

        lines, outputs_block = top_block(content, "Outputs")
        if outputs_block is None:
            content += "\n[Outputs]\n  csv = true\n[]\n"
        else:
            outputs_text = "".join(lines[outputs_block.start + 1:outputs_block.end])
            has_csv_shortcut = re.search(
                r"^\s*csv\s*=\s*(?:true|1)\s*(?:#.*)?$",
                outputs_text, re.MULTILINE | re.IGNORECASE,
            ) is not None
            has_csv_object = re.search(
                r"^\s*type\s*=\s*CSV\s*(?:#.*)?$",
                outputs_text, re.MULTILINE | re.IGNORECASE,
            ) is not None
            if not (has_csv_shortcut or has_csv_object):
                lines.insert(outputs_block.start + 1, "  csv = true\n")
                content = "".join(lines)
        return content

    def build_input_file(self, output_path: str) -> str:
        # Hierarchy samplers are injected by the GOLEM runner after it reads
        # the completed custom input file. This keeps input-file generation
        # unchanged and lets the runner decide from the referenced mesh level.
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
    fault_properties: Optional[
        Union[GolemFaultProperties, Dict[str, GolemFaultProperties]]
    ] = None
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
    end_time: Optional[float] = 5e11
    dt: float = 1.667e11
    num_steps: int = 3
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
        if self.mesh_results.mesh_type != MeshType.UNSTRUCTURED:
            # Cell-centred implicit/structured nodes can be inset from the
            # nominal structural-model extent. Anchor the IC to actual nodes.
            node_coordinates = getattr(self.mesh_results, "nodes", None)
            if node_coordinates is None:
                node_coordinates = getattr(self.mesh_results, "vertices", None)
            if node_coordinates is None:
                raise ValueError("Mesh node coordinates are required for GOLEM gradient initial conditions")
            coordinates = np.asarray(node_coordinates, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] < 3 or coordinates.shape[0] == 0:
                raise ValueError("GOLEM mesh nodes must be a nonempty N x 3 coordinate array")
            self.z_bottom = float(np.min(coordinates[:, 2]))
            self.z_top = float(np.max(coordinates[:, 2]))
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

        if isinstance(self.fault_properties, GolemFaultProperties):
            supplied_fault_properties = {
                self.fault_properties.name: self.fault_properties
            }
        else:
            supplied_fault_properties = dict(self.fault_properties or {})
        if self.fault_blocks and not supplied_fault_properties:
            raise ValueError(
                f"The mesh contains {len(self.fault_blocks)} exported 2D fault "
                "block(s), but fault_properties was not supplied. Pass one "
                "GolemFaultProperties entry per 2D fault block. Entries are "
                "assigned in dictionary insertion order."
            )
        # One property object can represent several element blocks belonging
        # to the same physical fault.
        if self.fault_blocks and len(supplied_fault_properties) == 1:
            template = next(iter(supplied_fault_properties.values()))
            supplied_fault_properties = {
                f"{template.name}_{index + 1}": template
                for index in range(len(self.fault_blocks))
            }
        if self.fault_blocks and len(supplied_fault_properties) != len(self.fault_blocks):
            raise ValueError(
                f"The mesh contains {len(self.fault_blocks)} exported 2D fault "
                f"block(s), but {len(supplied_fault_properties)} fault_properties "
                "entrie(s) were supplied. Pass exactly one entry per fault block."
            )
        for fault_name, props in supplied_fault_properties.items():
            if props.aperture <= 0:
                raise ValueError(
                    f"Fault aperture for {fault_name!r} must be greater than zero"
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
                  fluid: GolemFluidProperties, fracture: bool = False,
                  use_supg: bool = False) -> str:
        fracture_lines = ""
        if fracture:
            fracture_lines = (
                "\n    material_type = FRAC"
                f"\n    fracture_aperture_uo = fracture_aperture_{block_id}"
            )
        material_type = "GolemMaterialTHSUPG" if use_supg else "GolemMaterialTH"
        return f"""  [./block_{block_id}]
    type = {material_type}
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
                               fluid: GolemFluidProperties,
                               fracture: bool = False) -> str:
        """GOLEM constant-property objects required by the target Docker image."""
        aperture_object = ""
        if fracture:
            aperture_object = f"""
  [./fracture_aperture_{block_id}]
    type = GolemFractureApertureConstant
    value = {getattr(props, 'aperture')!r}
    execute_on = 'INITIAL'
  [../]"""
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
  [../]{aperture_object}"""

    def build_input_file(self, output_path: str) -> str:
        # Keep the proven no-fault formulation byte-for-byte equivalent. The
        # fracture formulation is enabled only when actual 2D element blocks
        # are present in the exported mesh.
        has_faults = bool(self.fault_blocks)
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
            materials.append(self._material(
                props.name, mesh_block_id, props, self.fluid,
                use_supg=has_faults,
            ))
            property_user_objects.append( self._property_user_objects(mesh_block_id, props, self.fluid) )
        for ordinal, block_index in enumerate(self.fault_blocks):
            mesh_block_id = block_index + 1
            props = self._fault_property(ordinal)
            materials.append(
                self._material(
                    props.name, mesh_block_id, props, self.fluid,
                    fracture=True, use_supg=True,
                )
            )
            property_user_objects.append(
                self._property_user_objects( mesh_block_id, props, self.fluid, fracture=True )
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

        supg_global = "  supg_uo = supg\n" if has_faults else ""
        temperature_time_kernel = (
            "GolemKernelTimeTSUPG" if has_faults else "GolemKernelTimeT"
        )
        if has_faults:
            temperature_diffusion_kernel = """  [./TKernel]
    type = GolemKernelT
    variable = temperature
  [../]
"""
            temperature_advection_kernel = """  [./THKernel]
    type = GolemKernelTHSUPG
    variable = temperature
  [../]"""
            supg_user_object = """  [./supg]
    type = GolemSUPG
    execute_on = 'INITIAL'
  [../]
"""
        else:
            temperature_diffusion_kernel = ""
            temperature_advection_kernel = """  [./THKernel]
    type = GolemKernelTH
    variable = temperature
    is_conservative = true
  [../]"""
            supg_user_object = ""

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
{supg_global.rstrip()}
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
    type = {temperature_time_kernel}
    variable = temperature
  [../]
{temperature_diffusion_kernel}{temperature_advection_kernel}
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
{supg_user_object}{chr(10).join(property_user_objects)}
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
  [exo]
    type = Exodus
    file_base = 'output/wbgeo'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [csv]
    type = CSV
  []
[]
"""
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        return output_path
@wbgeo_type(name="GOLEM Simple Reservoir Problem", color="#e07a5f",
            identifier="GolemSimpleReservoirProblemBuilder")
@dataclass
class GolemSimpleReservoirProblemBuilder:
    """Create the Model-1 reservoir mesh AND its GOLEM input automatically.

    The reservoir mesh fills the complete structural-model extent. The two
    synthetic fracture planes keep their original Model-1 separation and are
    centred laterally so their distances to the left/right model boundaries are
    equal. The well keeps its original shape and offsets relative to the second
    fracture plane. Injection/extraction points remain at the centres of the
    first/second fracture planes, respectively.

    All 3D material blocks receive one common reservoir material. Existing 2D
    geological faults receive the same intrinsic rock properties, while the
    two internally-created 2D planes remain synthetic fractures. Exported 1D
    material blocks are wells.
    WBGeo's six trailing exterior-boundary blocks are excluded from materials.
    """

    # ------------------------------------------------------------------
    # Required model input
    # ------------------------------------------------------------------
    geomodel_result: StructuralModelResults

    # ------------------------------------------------------------------
    # Unstructured-mesh inputs
    # These are passed to create_unstructured_mesh_data().
    # ------------------------------------------------------------------
    mesh_size: float = 30.0
    curve_mesh_size: float = 5.0
    mapping_litho: str = "automatic_centers"
    gmsh_flag: bool = False
    merge_file: Optional[str] = None
    z_threshold: Optional[float] = None

    # Any additional keyword accepted by create_unstructured_mesh_data() can
    # be supplied here without changing this builder.
    unstructured_mesh_options: Optional[Dict[str, Any]] = None

    # Optional actual-coordinate overrides. When None, the reservoir uses the
    # full structural-model extent and the internal Model-1 geometry is centred
    # inside it while preserving its original absolute dimensions/distances.
    reservoir_extent_override: Optional[
        Tuple[float, float, float, float, float, float]
    ] = None
    fracture_planes_override: Optional[List[Tuple[float, ...]]] = None
    wells_override: Optional[List[Tuple[float, ...]]] = None

    # ------------------------------------------------------------------
    # Refinement inputs
    # ------------------------------------------------------------------
    well_refine_size_min: float = 5.0
    well_refine_size_max: float = 30.0
    well_refine_dist_min: float = 10.0
    well_refine_dist_max: float = 60.0

    fracture_refine_size_min: float = 5.0
    fracture_refine_size_max: float = 30.0
    fracture_refine_dist_min: float = 10.0
    fracture_refine_dist_max: float = 60.0

    # If True, the six refinement lengths above are interpreted in the
    # reference 0..1000 model and scaled with the smallest model-axis scale.
    # Mesh_size and curve_mesh_size remain direct unstructured-mesh inputs.
    scale_refinement_with_model: bool = True

    # ------------------------------------------------------------------
    # Reservoir / fracture / well material inputs
    # One common material is intentionally applied to ALL 3D blocks.
    # ------------------------------------------------------------------
    rock_porosity: float = 0.1
    rock_permeability: float = 1.0e-15
    rock_density: float = 2.6e3
    rock_thermal_conductivity: float = 3.0
    rock_heat_capacity: float = 950.0

    fracture_porosity: float = 1.0
    fracture_aperture: float = 1.0e-4

    # Geological faults already present in geomodel_result are NOT given a
    # separate fault material. They use exactly the same intrinsic rock
    # properties as the 3D reservoir. GOLEM still needs an aperture because
    # an exported geological fault is a lower-dimensional 2D FRAC block.
    geological_fault_aperture: float = 1.0

    well_porosity: float = 1.0
    well_permeability: float = 3.12e-4
    well_scaling_factor: float = 0.05

    fluid_density: float = 1.0e3
    fluid_viscosity: float = 1.0e-3
    fluid_heat_capacity: float = 4.2e3
    fluid_thermal_conductivity: float = 0.65
    fluid_modulus: float = 2.5e9

    # ------------------------------------------------------------------
    # Initial and boundary conditions
    # ------------------------------------------------------------------
    initial_pressure: float = 1.0e7
    initial_temperature: float = 150.0
    boundary_pressure: float = 1.0e7
    boundary_temperature: float = 150.0
    injection_temperature: float = 70.0

    # None -> use the automatically scaled reference Model-1 points.
    injection_point: Optional[Tuple[float, float, float]] = None
    extraction_point: Optional[Tuple[float, float, float]] = None

    injection_rate: float = 3.0
    extraction_rate: float = 3.0
    source_start_time: float = 0.0
    source_end_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Transient solve inputs
    # 19 solve times + INITIAL gives the 20 Exodus states used by Model 1.
    # ------------------------------------------------------------------
    start_time: float = 0.0
    end_time: Optional[float] = None
    time_sequence: Optional[List[float]] = None
    automatic_scaling: bool = True
    compute_scaling_once: bool = False

    # Low-level solver controls from the working simple-reservoir input.
    coupled_petsc_options_iname: str = (
        "-ksp_type -ksp_rtol -ksp_max_it -snes_type "
        "-snes_linesearch_type -snes_atol -snes_rtol "
        "-snes_stol -snes_max_it"
    )
    coupled_petsc_options_value: str = (
        "fgmres 1e-12 100 newtonls basic 1e-10 1e-4 0 1000"
    )
    pressure_petsc_options_iname: str = (
        "-ksp_type -pc_type -pc_hypre_type"
    )
    pressure_petsc_options_value: str = "preonly hypre boomeramg"
    temperature_petsc_options_iname: str = (
        "-ksp_type -pc_type -sub_pc_type -sub_pc_factor_levels "
        "-ksp_rtol -ksp_max_it"
    )
    temperature_petsc_options_value: str = "fgmres asm ilu 1 1e-4 500"

    print_linear_residuals: bool = True
    print_nonlinear_converged_reason: bool = True
    perf_graph: bool = True

    # ------------------------------------------------------------------
    # Reference Model-1 template geometry. These are intentionally internal:
    # users normally provide model extent + meshing/solver parameters only.
    # ------------------------------------------------------------------
    REFERENCE_MODEL_EXTENT: ClassVar[Tuple[float, ...]] = (
        0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0
    )
    # The generated reservoir now uses the full structural-model domain.
    REFERENCE_RESERVOIR_EXTENT: ClassVar[Tuple[float, ...]] = (
        0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0
    )
    REFERENCE_FRACTURE_PLANES: ClassVar[Tuple[Tuple[float, ...], ...]] = (
        (
            400.0, 450.0, 450.0,
            400.0, 550.0, 450.0,
            400.0, 550.0, 550.0,
            400.0, 450.0, 550.0,
        ),
        (
            600.0, 450.0, 450.0,
            600.0, 550.0, 450.0,
            600.0, 550.0, 550.0,
            600.0, 450.0, 550.0,
        ),
    )
    REFERENCE_WELLS: ClassVar[Tuple[Tuple[float, ...], ...]] = (
        (
            575.0, 500.0, 490.0,
            587.5, 500.0, 495.0,
            600.0, 500.0, 500.0,
            612.5, 500.0, 505.0,
            625.0, 500.0, 510.0,
        ),
    )
    REFERENCE_INJECTION_POINT: ClassVar[Tuple[float, float, float]] = (
        400.0, 500.0, 500.0
    )
    REFERENCE_EXTRACTION_POINT: ClassVar[Tuple[float, float, float]] = (
        600.0, 500.0, 500.0
    )

    def __post_init__(self) -> None:
        self._validate_inputs()
        self._prepare_scaled_geometry()
        self._build_reservoir_mesh()
        self._classify_material_blocks()
        self._prepare_time_controls()

    # ------------------------------------------------------------------
    # Geometry scaling
    # ------------------------------------------------------------------
    @staticmethod
    def _extent_tuple(extent) -> Tuple[float, float, float, float, float, float]:
        values = tuple(float(value) for value in extent)
        if len(values) != 6:
            raise ValueError("Model extent must contain 6 values: xmin,xmax,ymin,ymax,zmin,zmax")
        return values

    def _axis_scales(self) -> Tuple[float, float, float]:
        ref = self.REFERENCE_MODEL_EXTENT
        dst = self.model_extent
        result = []
        for axis in range(3):
            r0, r1 = ref[2 * axis], ref[2 * axis + 1]
            d0, d1 = dst[2 * axis], dst[2 * axis + 1]
            if r1 <= r0 or d1 <= d0:
                raise ValueError("Reference and structural-model extents must have positive size")
            result.append((d1 - d0) / (r1 - r0))
        return tuple(result)

    def _scale_point(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        ref = self.REFERENCE_MODEL_EXTENT
        dst = self.model_extent
        out = []
        for axis, value in enumerate(point):
            r0, r1 = ref[2 * axis], ref[2 * axis + 1]
            d0, d1 = dst[2 * axis], dst[2 * axis + 1]
            fraction = (float(value) - r0) / (r1 - r0)
            out.append(d0 + fraction * (d1 - d0))
        return tuple(out)

    def _scale_extent(
        self, extent: Tuple[float, ...]
    ) -> Tuple[float, float, float, float, float, float]:
        lo = self._scale_point((extent[0], extent[2], extent[4]))
        hi = self._scale_point((extent[1], extent[3], extent[5]))
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    @staticmethod
    def _translate_flat_xyz(
        coordinates: Tuple[float, ...],
        *,
        reference_origin: Tuple[float, float, float],
        destination_origin: Tuple[float, float, float],
    ) -> Tuple[float, ...]:
        """Translate flat x,y,z coordinates without changing their shape/size."""
        if len(coordinates) % 3:
            raise ValueError("Plane/well coordinate sequences must contain x,y,z triples")
        translated = []
        for i in range(0, len(coordinates), 3):
            translated.extend(
                (
                    destination_origin[0] + coordinates[i] - reference_origin[0],
                    destination_origin[1] + coordinates[i + 1] - reference_origin[1],
                    destination_origin[2] + coordinates[i + 2] - reference_origin[2],
                )
            )
        return tuple(float(v) for v in translated)

    def _default_centered_geometry(self):
        """Return planes, well and source points for the full model domain.

        The original Model-1 fracture spacing is preserved as an absolute
        distance. The plane pair is centred in x, giving equal left/right
        margins. Their 100 x 100 geometry remains centred in y and z. The well
        is translated with fracture 2, preserving all original offsets.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.model_extent
        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)
        z_center = 0.5 * (zmin + zmax)

        reference_plane_1_x = float(self.REFERENCE_FRACTURE_PLANES[0][0])
        reference_plane_2_x = float(self.REFERENCE_FRACTURE_PLANES[1][0])
        plane_spacing = reference_plane_2_x - reference_plane_1_x
        if plane_spacing <= 0:
            raise ValueError("Reference fracture-plane spacing must be positive")
        if (xmax - xmin) < plane_spacing:
            raise ValueError(
                "Structural-model x extent is smaller than the required "
                f"simple-reservoir fracture spacing ({plane_spacing})."
            )

        # Both planes have the same lateral distance to the corresponding
        # x-boundary while keeping their original absolute separation.
        plane_1_x = x_center - 0.5 * plane_spacing
        plane_2_x = x_center + 0.5 * plane_spacing

        reference_center = self.REFERENCE_INJECTION_POINT
        plane_1 = self._translate_flat_xyz(
            self.REFERENCE_FRACTURE_PLANES[0],
            reference_origin=reference_center,
            destination_origin=(plane_1_x, y_center, z_center),
        )
        plane_2 = self._translate_flat_xyz(
            self.REFERENCE_FRACTURE_PLANES[1],
            reference_origin=self.REFERENCE_EXTRACTION_POINT,
            destination_origin=(plane_2_x, y_center, z_center),
        )

        wells = [
            self._translate_flat_xyz(
                well,
                reference_origin=self.REFERENCE_EXTRACTION_POINT,
                destination_origin=(plane_2_x, y_center, z_center),
            )
            for well in self.REFERENCE_WELLS
        ]

        injection = (plane_1_x, y_center, z_center)
        extraction = (plane_2_x, y_center, z_center)
        return [plane_1, plane_2], wells, injection, extraction

    def _prepare_scaled_geometry(self) -> None:
        self.model_extent = self._extent_tuple(
            self.geomodel_result.structural_frame.grid.extent
        )
        self.axis_scales = self._axis_scales()
        self.length_scale = min(self.axis_scales)

        # By default, Simple Reservoir now occupies the complete structural
        # domain. An explicit override remains an actual-coordinate override.
        self.reservoir_extent = (
            self._extent_tuple(self.reservoir_extent_override)
            if self.reservoir_extent_override is not None
            else self.model_extent
        )

        (
            default_fracture_planes,
            default_wells,
            default_injection,
            default_extraction,
        ) = self._default_centered_geometry()

        self.fracture_planes = (
            [tuple(float(v) for v in plane) for plane in self.fracture_planes_override]
            if self.fracture_planes_override is not None
            else default_fracture_planes
        )

        self.wells = (
            [tuple(float(v) for v in well) for well in self.wells_override]
            if self.wells_override is not None
            else default_wells
        )

        if self.injection_point is None:
            self.injection_point = default_injection
        else:
            self.injection_point = tuple(float(v) for v in self.injection_point)

        if self.extraction_point is None:
            self.extraction_point = default_extraction
        else:
            self.extraction_point = tuple(float(v) for v in self.extraction_point)

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------
    def _scaled_refinement_value(self, value: float) -> float:
        if self.scale_refinement_with_model:
            return float(value) * self.length_scale
        return float(value)

    def _build_reservoir_mesh(self) -> None:
        refinement = Refinement(
            wells=LinearWellRefinement(
                SizeMin=self._scaled_refinement_value(self.well_refine_size_min),
                SizeMax=self._scaled_refinement_value(self.well_refine_size_max),
                DistMin=self._scaled_refinement_value(self.well_refine_dist_min),
                DistMax=self._scaled_refinement_value(self.well_refine_dist_max),
            ),
            extraplanes=ExtraPlaneRefinement(
                SizeMin=self._scaled_refinement_value(self.fracture_refine_size_min),
                SizeMax=self._scaled_refinement_value(self.fracture_refine_size_max),
                DistMin=self._scaled_refinement_value(self.fracture_refine_dist_min),
                DistMax=self._scaled_refinement_value(self.fracture_refine_dist_max),
            ),
        )
        self.refinement = refinement

        mesh_kwargs: Dict[str, Any] = {
            "geomodel_result": self.geomodel_result,
            "extent": self.reservoir_extent,
            "wells": self.wells,
            "extra_planes": self.fracture_planes,
            "mesh_size": self.mesh_size,
            "curve_mesh_size": self.curve_mesh_size,
            "mapping_litho": self.mapping_litho,
            "refinement": refinement,
            "gmsh_flag": self.gmsh_flag,
        }
        if self.merge_file is not None:
            mesh_kwargs["merge_file"] = self.merge_file
        if self.z_threshold is not None:
            mesh_kwargs["z_threshold"] = self.z_threshold
        mesh_kwargs.update(dict(self.unstructured_mesh_options or {}))

        # Give a clear message if this WBGeo version does not expose one of the
        # requested mesh options instead of failing with a cryptic TypeError.
        signature = inspect.signature(create_unstructured_mesh_data)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            unsupported = [
                key for key in mesh_kwargs if key not in signature.parameters
            ]
            if unsupported:
                raise ValueError(
                    "This WBGeo create_unstructured_mesh_data() version does "
                    f"not support these requested option(s): {unsupported}. "
                    f"Supported arguments are: {list(signature.parameters)}"
                )

        self.mesh_results = create_unstructured_mesh_data(**mesh_kwargs)

    def build_mesh_generation_file(
        self,
        output_path: str,
        *,
        mesh_variable_name: str = "mesh_simple_reservoir",
        geomodel_variable_name: str = "structural_model_result",
    ) -> str:
        """Write a readable Python recipe that reproduces this reservoir mesh.

        The file records the resolved geometry used by the builder, the
        effective refinement values, and the mesh options passed to
        ``create_unstructured_mesh_data``. The generated recipe expects the
        structural-model result to exist under ``geomodel_variable_name``.
        """
        identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if not identifier.fullmatch(mesh_variable_name):
            raise ValueError(
                "mesh_variable_name must be a valid Python identifier; got "
                f"{mesh_variable_name!r}"
            )
        if not identifier.fullmatch(geomodel_variable_name):
            raise ValueError(
                "geomodel_variable_name must be a valid Python identifier; got "
                f"{geomodel_variable_name!r}"
            )

        def _format_flat_xyz(values: Tuple[float, ...]) -> str:
            rows = []
            for index in range(0, len(values), 3):
                xyz = values[index:index + 3]
                rows.append(
                    "        " + ", ".join(repr(float(value)) for value in xyz) + ","
                )
            return "\n".join(rows)

        extent_text = ", ".join(
            repr(float(value)) for value in self.reservoir_extent
        )
        planes_text = "\n".join(
            "    (\n" + _format_flat_xyz(tuple(plane)) + "\n    ),"
            for plane in self.fracture_planes
        )
        wells_text = "\n".join(
            "    (\n" + _format_flat_xyz(tuple(well)) + "\n    ),"
            for well in self.wells
        )

        well_size_min = self._scaled_refinement_value(self.well_refine_size_min)
        well_size_max = self._scaled_refinement_value(self.well_refine_size_max)
        well_dist_min = self._scaled_refinement_value(self.well_refine_dist_min)
        well_dist_max = self._scaled_refinement_value(self.well_refine_dist_max)
        frac_size_min = self._scaled_refinement_value(self.fracture_refine_size_min)
        frac_size_max = self._scaled_refinement_value(self.fracture_refine_size_max)
        frac_dist_min = self._scaled_refinement_value(self.fracture_refine_dist_min)
        frac_dist_max = self._scaled_refinement_value(self.fracture_refine_dist_max)

        mesh_kwargs: Dict[str, Any] = {
            "mesh_size": self.mesh_size,
            "curve_mesh_size": self.curve_mesh_size,
            "mapping_litho": self.mapping_litho,
            "gmsh_flag": self.gmsh_flag,
        }
        if self.merge_file is not None:
            mesh_kwargs["merge_file"] = self.merge_file
        if self.z_threshold is not None:
            mesh_kwargs["z_threshold"] = self.z_threshold
        mesh_kwargs.update(dict(self.unstructured_mesh_options or {}))

        option_lines = []
        option_dict: Dict[str, Any] = {}
        for key, value in mesh_kwargs.items():
            if identifier.fullmatch(str(key)):
                option_lines.append(f"    {key}={value!r},")
            else:
                option_dict[key] = value

        kwargs_setup = ""
        kwargs_expansion = ""
        if option_dict:
            kwargs_setup = (
                "\nreservoir_extra_mesh_options = " + repr(option_dict) + "\n"
            )
            kwargs_expansion = "\n    **reservoir_extra_mesh_options,"

        option_text = "\n".join(option_lines)

        recipe = f"""# Generated by GolemSimpleReservoirProblemBuilder.
# This records the exact reservoir mesh recipe used by the builder.
# It expects `{geomodel_variable_name}` to already contain StructuralModelResults.

from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement,
    LinearWellRefinement,
    ExtraPlaneRefinement,
)

reservoir_extent = ({extent_text})

# Synthetic fracture planes resolved by the builder.
reservoir_fracture_planes = [
{planes_text}
]

# Each well is one flat x, y, z sequence.
reservoir_wells = [
{wells_text}
]

reservoir_refinement = Refinement(
    wells=LinearWellRefinement(
        SizeMin={well_size_min!r},
        SizeMax={well_size_max!r},
        DistMin={well_dist_min!r},
        DistMax={well_dist_max!r},
    ),
    extraplanes=ExtraPlaneRefinement(
        SizeMin={frac_size_min!r},
        SizeMax={frac_size_max!r},
        DistMin={frac_dist_min!r},
        DistMax={frac_dist_max!r},
    ),
)
{kwargs_setup}
{mesh_variable_name} = create_unstructured_mesh_data(
    geomodel_result={geomodel_variable_name},
    extent=reservoir_extent,
    wells=reservoir_wells,
    extra_planes=reservoir_fracture_planes,
    refinement=reservoir_refinement,
{option_text}{kwargs_expansion}
)
"""

        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(recipe)
        return output_path

    def _classify_material_blocks(self) -> None:
        """Classify material blocks independently of whether the model has faults.

        WBGeo unstructured meshes append six exterior-boundary blocks. Before
        those blocks the material ordering is assumed to be:

          * structural-model 3D volume blocks,
          * optional structural-model 2D geological-fault block(s),
          * the internally-added extra-plane 2D blocks,
          * the internally-added 1D well block(s).

        The number of synthetic fracture blocks is known exactly from
        ``self.fracture_planes``. Therefore the LAST N 2D material blocks are
        the synthetic reservoir fractures and any earlier 2D blocks are
        geological faults.

        Geological faults are assigned the SAME intrinsic rock properties as
        the 3D reservoir. They remain ``material_type = FRAC`` in GOLEM only
        because a 2D element needs lower-dimensional aperture scaling.
        """
        if self.mesh_results.mesh_type != MeshType.UNSTRUCTURED:
            raise ValueError(
                "The reservoir builder must create an UNSTRUCTURED mesh"
            )
        if len(self.mesh_results.elements) <= 6:
            raise ValueError(
                "The generated unstructured mesh has no material blocks before "
                "the six WBGeo exterior-boundary blocks"
            )

        material_blocks = self.mesh_results.elements[:-6]

        self.volume_blocks = [
            i + 1 for i, block in enumerate(material_blocks) if block.dim == 3
        ]
        two_d_blocks = [
            i + 1 for i, block in enumerate(material_blocks) if block.dim == 2
        ]
        self.well_blocks = [
            i + 1 for i, block in enumerate(material_blocks) if block.dim == 1
        ]

        unsupported = [
            (i + 1, block.dim)
            for i, block in enumerate(material_blocks)
            if block.dim not in (1, 2, 3)
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported GOLEM material blocks (block_id, dim): {unsupported}"
            )

        n_synthetic_fractures = len(self.fracture_planes)
        if len(two_d_blocks) < n_synthetic_fractures:
            raise ValueError(
                "The generated mesh contains fewer 2D material blocks than the "
                f"{n_synthetic_fractures} internally-created fracture plane(s). "
                f"Detected 2D blocks: {two_d_blocks}"
            )

        if n_synthetic_fractures:
            self.fracture_blocks = two_d_blocks[-n_synthetic_fractures:]
            self.geological_fault_blocks = two_d_blocks[:-n_synthetic_fractures]
        else:
            self.fracture_blocks = []
            self.geological_fault_blocks = two_d_blocks

        # All blocks that use reservoir intrinsic properties. 3D blocks are
        # standard domain materials; geological_fault_blocks use a separate
        # lower-dimensional GOLEM material definition but with these same
        # rock-property values.
        self.reservoir_property_blocks = (
            list(self.volume_blocks) + list(self.geological_fault_blocks)
        )

        if not self.volume_blocks:
            raise ValueError("Generated reservoir mesh has no 3D material block")
        if not self.fracture_blocks:
            raise ValueError(
                "Generated reservoir mesh has no internally-created 2D fracture block"
            )
        if not self.well_blocks:
            raise ValueError("Generated reservoir mesh has no 1D well block")

    # ------------------------------------------------------------------
    # Validation / time controls
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        positive = {
            "mesh_size": self.mesh_size,
            "curve_mesh_size": self.curve_mesh_size,
            "rock_porosity": self.rock_porosity,
            "rock_permeability": self.rock_permeability,
            "rock_density": self.rock_density,
            "rock_thermal_conductivity": self.rock_thermal_conductivity,
            "rock_heat_capacity": self.rock_heat_capacity,
            "fracture_porosity": self.fracture_porosity,
            "fracture_aperture": self.fracture_aperture,
            "geological_fault_aperture": self.geological_fault_aperture,
            "well_porosity": self.well_porosity,
            "well_permeability": self.well_permeability,
            "well_scaling_factor": self.well_scaling_factor,
            "fluid_density": self.fluid_density,
            "fluid_viscosity": self.fluid_viscosity,
            "fluid_heat_capacity": self.fluid_heat_capacity,
            "fluid_thermal_conductivity": self.fluid_thermal_conductivity,
            "fluid_modulus": self.fluid_modulus,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0]
        if invalid:
            raise ValueError(
                "These simple-reservoir inputs must be > 0: " + ", ".join(invalid)
            )

    def _prepare_time_controls(self) -> None:
        if self.time_sequence is None:
            self.time_sequence = [
                1.0,
                86400.0,
                604800.0,
                2592000.0,
                7776000.0,
                15552000.0,
                31536000.0,
                63072000.0,
                126144000.0,
                315360000.0,
                630720000.0,
                946080000.0,
                1261440000.0,
                1576800000.0,
                1892160000.0,
                2207520000.0,
                2522880000.0,
                2838240000.0,
                3153600014.0,
            ]
        else:
            self.time_sequence = [float(value) for value in self.time_sequence]

        if not self.time_sequence:
            raise ValueError("time_sequence must contain at least one solve time")
        if any(t <= self.start_time for t in self.time_sequence):
            raise ValueError("Every time_sequence value must be greater than start_time")
        if any(b <= a for a, b in zip(self.time_sequence, self.time_sequence[1:])):
            raise ValueError("time_sequence values must be strictly increasing")

        if self.end_time is None:
            self.end_time = float(self.time_sequence[-1])
        if self.end_time < self.time_sequence[-1]:
            raise ValueError("end_time cannot be earlier than the final time_sequence value")

        if self.source_end_time is None:
            self.source_end_time = self.end_time
        if self.source_end_time < self.source_start_time:
            raise ValueError("source_end_time must be >= source_start_time")

    # ------------------------------------------------------------------
    # GOLEM input generation
    # ------------------------------------------------------------------
    @staticmethod
    def _point_text(point: Tuple[float, float, float]) -> str:
        if len(point) != 3:
            raise ValueError("GOLEM source points must contain exactly x, y, z")
        return " ".join(f"{float(value):.16g}" for value in point)

    @staticmethod
    def _block_text(blocks: List[int]) -> str:
        return " ".join(str(value) for value in blocks)

    def build_input_file(self, output_path: str) -> str:
        volume_blocks = self._block_text(self.volume_blocks)
        geological_fault_blocks = self._block_text(self.geological_fault_blocks)
        fracture_blocks = self._block_text(self.fracture_blocks)
        well_blocks = self._block_text(self.well_blocks)

        geological_fault_material = ""
        if self.geological_fault_blocks:
            geological_fault_material = f"""  [geological_faults]
    type = GolemMaterialTHSUPG
    block = '{geological_fault_blocks}'
    material_type = FRAC
    fracture_aperture_uo = geological_fault_aperture_constant
    porosity_initial = {self.rock_porosity!r}
    porosity_uo = porosity_constant
    fluid_modulus = {self.fluid_modulus!r}
    permeability_type = isotropic
    permeability_initial = {self.rock_permeability!r}
    permeability_uo = permeability_constant
    solid_density_uo = solid_density_domain
    solid_thermal_conductivity_uo = lambda_domain
    solid_heat_capacity_uo = cp_domain
  []
"""

        injection_point = self._point_text(self.injection_point)
        extraction_point = self._point_text(self.extraction_point)
        time_sequence = " ".join(
            f"{float(value):.16g}" for value in self.time_sequence
        )

        content = f"""# Auto-generated by GolemSimpleReservoirProblemBuilder.
# The builder generated the unstructured WBGeo reservoir mesh internally.
# Structural model extent : {self.model_extent}
# Reservoir extent        : {self.reservoir_extent}
# 3D reservoir blocks             : {volume_blocks}
# 2D geological fault blocks      : {geological_fault_blocks or "none"}
# 2D synthetic fracture blocks    : {fracture_blocks}
# 1D well blocks                  : {well_blocks}

[Mesh]
  [input_mesh]
    type = FileMeshGenerator
    file = ${{GOLEM_MESH_FILE}}
  []
  [boundary_sidesets]
    type = SideSetsFromNodeSetsGenerator
    input = input_mesh
  []
  [injection_node]
    type = ExtraNodesetGenerator
    input = boundary_sidesets
    coord = '{injection_point}'
    use_closest_node = true
    new_boundary = injection_node
  []
[]

[Variables]
  [pore_pressure]
    order = FIRST
    family = LAGRANGE
    [InitialCondition]
      type = ConstantIC
      value = {self.initial_pressure!r}
    []
  []
  [temperature]
    order = FIRST
    family = LAGRANGE
    [InitialCondition]
      type = ConstantIC
      value = {self.initial_temperature!r}
    []
  []
[]

[GlobalParams]
  pore_pressure = pore_pressure
  temperature = temperature
  fluid_density_initial = {self.fluid_density!r}
  fluid_density_uo = fluid_density_constant
  fluid_viscosity_initial = {self.fluid_viscosity!r}
  fluid_viscosity_uo = fluid_viscosity_constant
  fluid_heat_capacity_uo = fluid_heat_capacity_constant
  fluid_thermal_conductivity_uo = fluid_thermal_conductivity_constant
  scaling_uo = scaling
  supg_uo = supg
[]

[Kernels]
  [Darcy]
    type = GolemKernelH
    variable = pore_pressure
  []
  [H_storage]
    type = GolemKernelTimeH
    variable = pore_pressure
  []
  [T_storage]
    type = GolemKernelTimeTSUPG
    variable = temperature
  []
  [Fourier_diffusion]
    type = GolemKernelT
    variable = temperature
  []
  [T_advection]
    type = GolemKernelTHSUPG
    variable = temperature
  []
[]

[BCs]
  [pressure_bc]
    type = DirichletBC
    variable = pore_pressure
    boundary = 'top bottom front back left right'
    value = {self.boundary_pressure!r}
  []
  [temperature_bc]
    type = DirichletBC
    variable = temperature
    boundary = 'top bottom front back left right'
    value = {self.boundary_temperature!r}
  []
  [injection_temperature]
    type = DirichletBC
    variable = temperature
    boundary = injection_node
    value = {self.injection_temperature!r}
  []
[]

[Materials]
  [uniform_rock]
    type = GolemMaterialTHSUPG
    block = '{volume_blocks}'
    porosity_initial = {self.rock_porosity!r}
    porosity_uo = porosity_constant
    fluid_modulus = {self.fluid_modulus!r}
    permeability_type = isotropic
    permeability_initial = {self.rock_permeability!r}
    permeability_uo = permeability_constant
    solid_density_uo = solid_density_domain
    solid_thermal_conductivity_uo = lambda_domain
    solid_heat_capacity_uo = cp_domain
  []
{geological_fault_material}  [fractures]
    type = GolemMaterialTHSUPG
    block = '{fracture_blocks}'
    material_type = FRAC
    fracture_aperture_uo = aperture_constant
    porosity_initial = {self.fracture_porosity!r}
    porosity_uo = porosity_constant
    fluid_modulus = {self.fluid_modulus!r}
    permeability_initial = 0
    permeability_uo = permeability_cubic_law
  []
  [well]
    type = GolemMaterialTHSUPG
    block = '{well_blocks}'
    material_type = well
    scaling_factor_initial = {self.well_scaling_factor!r}
    porosity_initial = {self.well_porosity!r}
    porosity_uo = porosity_constant
    permeability_initial = {self.well_permeability!r}
    permeability_uo = permeability_constant
    fluid_modulus = {self.fluid_modulus!r}
  []
[]

[AuxVariables]
  [frac_permeability]
    order = CONSTANT
    family = MONOMIAL
    block = '{fracture_blocks}'
  []
  [frac_aperture]
    order = CONSTANT
    family = MONOMIAL
    block = '{fracture_blocks}'
  []
[]

[DiracKernels]
  [injection]
    type = GolemDiracKernelTH
    variable = pore_pressure
    source_point = '{injection_point}'
    source_type = injection
    constant_rate = {self.injection_rate!r}
    start_time = {self.source_start_time!r}
    end_time = {self.source_end_time!r}
  []
  [extraction]
    type = GolemDiracKernelTH
    variable = pore_pressure
    source_point = '{extraction_point}'
    source_type = extraction
    constant_rate = {self.extraction_rate!r}
    start_time = {self.source_start_time!r}
    end_time = {self.source_end_time!r}
  []
[]

[AuxKernels]
  [frac_permeability_aux]
    type = MaterialStdVectorAux
    variable = frac_permeability
    property = permeability
    index = 0
    execute_on = TIMESTEP_END
  []
  [frac_aperture_aux]
    type = MaterialRealAux
    variable = frac_aperture
    property = scaling_factor
    block = '{fracture_blocks}'
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[UserObjects]
  [scaling]
    type = GolemScaling
    characteristic_stress = 1
    characteristic_length = 1
    characteristic_time = 1
    characteristic_temperature = 1
    execute_on = INITIAL
  []
  [supg]
    type = GolemSUPG
    execute_on = INITIAL
  []
  [fluid_density_constant]
    type = GolemFluidDensityConstant
    execute_on = INITIAL
  []
  [fluid_viscosity_constant]
    type = GolemFluidViscosityConstant
    execute_on = INITIAL
  []
  [porosity_constant]
    type = GolemPorosityConstant
    execute_on = INITIAL
  []
  [permeability_constant]
    type = GolemPermeabilityConstant
    execute_on = INITIAL
  []
  [permeability_cubic_law]
    type = GolemPermeabilityCubicLaw
    execute_on = INITIAL
  []
  [solid_density_domain]
    type = GolemSolidDensityConstant
    value = {self.rock_density!r}
    execute_on = INITIAL
  []
  [lambda_domain]
    type = GolemSolidThermalConductivityConstant
    value = {self.rock_thermal_conductivity!r}
    execute_on = INITIAL
  []
  [cp_domain]
    type = GolemSolidHeatCapacityConstant
    value = {self.rock_heat_capacity!r}
    execute_on = INITIAL
  []
  [aperture_constant]
    type = GolemFractureApertureConstant
    value = {self.fracture_aperture!r}
    execute_on = INITIAL
  []
  [geological_fault_aperture_constant]
    type = GolemFractureApertureConstant
    value = {self.geological_fault_aperture!r}
    execute_on = INITIAL
  []
  [fluid_heat_capacity_constant]
    type = GolemFluidHeatCapacityConstant
    value = {self.fluid_heat_capacity!r}
    execute_on = INITIAL
  []
  [fluid_thermal_conductivity_constant]
    type = GolemFluidThermalConductivityConstant
    value = {self.fluid_thermal_conductivity!r}
    execute_on = INITIAL
  []
[]

[Postprocessors]
  [pressure_injection]
    type = PointValue
    point = '{injection_point}'
    variable = pore_pressure
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [temperature_injection]
    type = PointValue
    point = '{injection_point}'
    variable = temperature
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [pressure_extraction]
    type = PointValue
    point = '{extraction_point}'
    variable = pore_pressure
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
  [temperature_extraction]
    type = PointValue
    point = '{extraction_point}'
    variable = temperature
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = csv
  []
[]

[Preconditioning]
  [fieldsplit]
    type = FSP
    full = true
    topsplit = HT
    [HT]
      splitting = 'H T'
      splitting_type = additive
      petsc_options_iname = '{self.coupled_petsc_options_iname}'
      petsc_options_value = '{self.coupled_petsc_options_value}'
    []
    [H]
      vars = pore_pressure
      petsc_options_iname = '{self.pressure_petsc_options_iname}'
      petsc_options_value = '{self.pressure_petsc_options_value}'
    []
    [T]
      vars = temperature
      petsc_options_iname = '{self.temperature_petsc_options_iname}'
      petsc_options_value = '{self.temperature_petsc_options_value}'
    []
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  start_time = {self.start_time!r}
  end_time = {self.end_time!r}
  automatic_scaling = {str(self.automatic_scaling).lower()}
  compute_scaling_once = {str(self.compute_scaling_once).lower()}
  [TimeStepper]
    type = TimeSequenceStepper
    time_sequence = '{time_sequence}'
  []
[]

[Outputs]
  print_linear_residuals = {str(self.print_linear_residuals).lower()}
  perf_graph = {str(self.perf_graph).lower()}
  print_nonlinear_converged_reason = {str(self.print_nonlinear_converged_reason).lower()}
  [exo]
    type = Exodus
    file_base = 'output/wbgeo'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [csv]
    type = CSV
  []
[]
"""
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        return output_path
