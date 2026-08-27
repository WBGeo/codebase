import logging
import os
import tempfile
import subprocess

from core.meshing_components.mesh_format.exodus.Exo_format import (
    export_mesh_results_to_exodus
)

from core.object_components import MeshResults


logger = logging.getLogger(__name__)


import re

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


def run_golem_with_image(
    image_name,
    input_file,
    project_dir,
    example_dir,
    docker_dir,
    mesh_test: MeshResults,
    mesh_type="unstr",
    output_dir=None,
    nproc=4
):

    # =====================================================
    # PROJECT AND OUTPUT DIRECTORIES
    # =====================================================

    os.makedirs(project_dir, exist_ok=True)

    if output_dir is None:
        output_dir = os.path.join(
            project_dir,
            "golem_output"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    print("Host output directory:")
    print(output_dir)



    # =====================================================
    # CREATE TEMP GOLEM WORK DIRECTORY
    # INSIDE DOCKER MOUNT
    # =====================================================

    tmp_dir = tempfile.mkdtemp(
        prefix="golem_mesh_",
        dir=project_dir
    )


    print("Temporary GOLEM directory:")
    print(tmp_dir)



    # =====================================================
    # EXPORT WBGeo MESH TO EXODUS
    # =====================================================

    exo_buffer = export_mesh_results_to_exodus(
        mesh_test,
        type=mesh_type
    )


    exo_path = os.path.join(
        tmp_dir,
        "wbgeo_mesh.exo"
    )


    with open(exo_path, "wb") as f:
        f.write(
            exo_buffer.read()
        )


    print("Host Exodus mesh:")
    print(exo_path)



    # =====================================================
    # HOST PATH -> DOCKER PATH
    # =====================================================

    docker_mesh_path = exo_path.replace(
        project_dir,
        docker_dir
    )


    docker_output_dir = output_dir.replace(
        project_dir,
        docker_dir
    )


    print("Docker Exodus mesh:")
    print(docker_mesh_path)

    print("Docker output directory:")
    print(docker_output_dir)



    # =====================================================
    # CREATE GENERATED GOLEM INPUT FILE
    # =====================================================

    host_example_dir = example_dir.replace(
        docker_dir,
        project_dir
    )


    original_input = os.path.join(
        host_example_dir,
        input_file
    )


    print("Original GOLEM input:")
    print(original_input)

    with open(original_input, "r") as f:
        model_content = f.read()

    validate_golem_input(
        model_content,
        mesh_test
    )
    with open(original_input, "r") as f:
        model_content = f.read()



    # Replace MOOSE substitution variable

    model_content = model_content.replace(
        "${GOLEM_MESH_FILE}",
        docker_mesh_path
    )


    if "${GOLEM_MESH_FILE}" in model_content:
        raise RuntimeError(
            "ERROR: GOLEM_MESH_FILE replacement failed"
        )



    generated_input = os.path.join(
        tmp_dir,
        "model_generated.i"
    )


    with open(generated_input, "w") as f:
        f.write(model_content)



    docker_input = generated_input.replace(
        project_dir,
        docker_dir
    )


    print("Generated GOLEM input:")
    print(docker_input)



    # =====================================================
    # RUN GOLEM
    # =====================================================

    bash_command = f"""

echo "Starting GOLEM..."

cd {example_dir}


echo "Checking mesh:"
ls -lh {docker_mesh_path}


echo "Checking input:"
ls -lh {docker_input}


echo "Checking output directory:"
ls -lh {docker_output_dir}


mpiexec --allow-run-as-root \
-n {nproc} \
../golem-opt \
-i {docker_input}


echo "GOLEM finished"


echo "Output files:"
ls -lh {docker_output_dir}

"""


    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",

        # project mount
        "-v",
        f"{project_dir}:{docker_dir}",

        image_name,

        "-c",
        bash_command
    ]



    print("Running GOLEM...")

    process = subprocess.Popen(cmd)

    process.wait()



    print("\n===== GOLEM FINISHED =====")
    print("Return code:", process.returncode)



    return {
        "return_code": process.returncode,
        "mesh_file": exo_path,
        "input_file": generated_input,
        "output_dir": output_dir
    }
