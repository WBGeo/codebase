import logging
import os
import subprocess
import uuid
import shutil

import typing
from py_api_wbgeo.nodesapi import wbgeo_component, wbgeo_type, wbgeo_inspector, \
  InspectorHelper, AnnotatedScriptType

from core.meshing_components.mesh_format.exodus.Exo_format import (
  export_mesh_results_to_exodus
)

from core.object_components import MeshResults

from core.meshing_components.mesh_format.exodus.Exo_format import MeshType

from core.object_components import SimulationResults
from core.simulation_components.output_format.exodus.unified_format_exodus import \
  load_exodus_results

from core.meshing_components.mesh_format.exodus.Exo_format import MeshType_A

logger = logging.getLogger(__name__)

import re

GolemInputFileType = typing.Annotated[
  str, AnnotatedScriptType(name='path', color='aqua', identifier='GolemInputFileType',
                           controlled='RemoteFile|endswith=.i')]


def validate_golem_input(content: str, mesh_test: MeshResults):
  required_blocks = [
    "Mesh",
    "Variables",
    "Kernels",
    "Executioner",
    "Outputs",
    "BCs",
    "Materials",
  ]

  missing = []

  for block in required_blocks:
    pattern = rf"^\[{block}\]"
    if not re.search(pattern, content, re.MULTILINE):
      missing.append(block)

  if missing:
    raise ValueError(
      "Missing required blocks: "
      + ", ".join(missing)
    )

  # --------------------------------------------------
  # Check Mesh block contains FileMeshGenerator
  # --------------------------------------------------

  mesh_pattern = re.compile(
    r"""
    \[Mesh\]
    .*?
    type\s*=\s*FileMeshGenerator
    .*?
    file\s*=\s*\$\{GOLEM_MESH_FILE\}
    """,
    re.DOTALL | re.VERBOSE,
  )

  if not mesh_pattern.search(content):
    raise ValueError(
      "Mesh block must contain "
      "type = FileMeshGenerator "
      "and file = ${GOLEM_MESH_FILE}"
    )

  # --------------------------------------------------
  # Check number of Materials blocks
  # --------------------------------------------------

  materials_match = re.search(
    r"\[Materials\](.*?)\n\[\]",
    content,
    re.DOTALL
  )

  if not materials_match:
    raise ValueError(
      "Materials block not found"
    )

  materials_content = materials_match.group(1)

  # Count material blocks
  # Example:
  # [block_1]
  # [block_2]
  # ...

  material_blocks = re.findall(
    r"\n\s*\[block_\d+\]",
    materials_content
  )

  n_material_blocks = len(material_blocks)

  # Number of blocks in MeshResults
  n_mesh_blocks = len(mesh_test.elements)

  n_mesh_blocks = len(mesh_test.elements) - 6

  if n_material_blocks != n_mesh_blocks:
    raise ValueError(
      f"Number of material blocks ({n_material_blocks}) "
      f"does not match number of mesh blocks ({n_mesh_blocks})"
    )

  print(
    f"Validation successful: "
    f"{n_mesh_blocks} mesh blocks and "
    f"{n_material_blocks} material blocks."
  )


@wbgeo_component(identifier='wbgeo:golem_simulation',
                 color="#e5d016",
                 border_color='#000000',
                 group='simulation_components',
                 title='GOLEM Simulation')
def example_golem(mesh_results: MeshResults,
                  mesh_type: MeshType_A = MeshType_A.UNSTRUCTURED,
                  input_file: GolemInputFileType = None,
                  nproc: int = 4
                  ) -> SimulationResults:
  # the directory we work in, shared as a docker volume
  workspace_dir = os.getenv('GOLEM_WORKSPACE', '.')
  workspace_dir = workspace_dir + "/" + str(uuid.uuid4())

  try:  # create mounted tempdir
    os.makedirs(workspace_dir)

    # write mesh-results into temp dir
    exo_buffer = export_mesh_results_to_exodus(
      mesh_results,
      type=mesh_type
    )
    exo_path = os.path.join(workspace_dir, "wbgeo_mesh.exo")

    with open(exo_path, "wb") as f:
      f.write(exo_buffer.read())

    # modify the model.i
    with open(input_file, "r") as f:
      model_content = f.read()

    validate_golem_input(
      model_content,
      mesh_results
    )

    # change exodus path
    model_content = model_content.replace(
      "${GOLEM_MESH_FILE}",
      exo_path
    )
    # write model_generated.i
    generated_input = os.path.join(
      workspace_dir,
      "model_generated.i"
    )
    with open(generated_input, "w") as f:
      f.write(model_content)

    # Execute golem via mpiexec
    command = [
      "mpiexec",
      "-n", str(nproc),
      "--allow-run-as-root",
      os.getenv('GOLEM_BINARY', '~/projects/golem/golem-opt'),
      "-i", generated_input,
    ]

    res = subprocess.run(command, cwd=workspace_dir, capture_output=True, text=True)

    if res.returncode != 0:
      import sys
      print(res.stdout)
      print(res.stderr, file=sys.stderr)
      raise ValueError("Failed to executegolem with ", command, workspace_dir, res.stdout,
                       res.stderr)

    output_file = os.path.join(workspace_dir, './output/wbgeo-.e')
    if not os.path.exists(output_file):
      raise ValueError("Simulation outputs not found at ", output_file)

    # return MeshResults as the output
    return load_exodus_results(output_file)

  finally:
    shutil.rmtree(workspace_dir, ignore_errors=True)
