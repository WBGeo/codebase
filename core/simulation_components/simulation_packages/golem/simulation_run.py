import logging
import os
import shutil
import tempfile
import subprocess
import re

from core.meshing_components.mesh_format.exodus.Exo_format import (
    export_mesh_results_to_exodus
)

from core.object_components import MeshResults
from core.meshing_components.mesh_format.exodus.Exo_format import MeshType

logger = logging.getLogger(__name__)


def validate_golem_input(
    content: str,
    mesh_test: MeshResults,
    mesh_type: MeshType
):

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

        if not re.search(
            pattern,
            content,
            re.MULTILINE
        ):
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

    material_blocks = re.findall(
        r"\n\s*\[block_\d+\]",
        materials_content
    )

    n_material_blocks = len(
        material_blocks
    )

    # Number of blocks in MeshResults

    if mesh_type == MeshType.UNSTRUCTURED:

        n_mesh_blocks = (
            len(mesh_test.elements) - 6
        )

    else:

        n_mesh_blocks = len(
            mesh_test.elements
        )

    if n_material_blocks != n_mesh_blocks:

        raise ValueError(
            f"Number of material blocks "
            f"({n_material_blocks}) "
            f"does not match number of mesh blocks "
            f"({n_mesh_blocks})"
        )

    # --------------------------------------------------
    # Helper:
    # return only DIRECT child blocks of a parent block
    # --------------------------------------------------

    def get_direct_child_blocks(
        text: str,
        parent_block: str
    ):

        lines = text.splitlines()

        inside_parent = False
        depth = 0
        children = []

        parent_start = f"[{parent_block}]"

        for line in lines:

            stripped = line.strip()

            # Find parent block
            if not inside_parent:

                if stripped == parent_start:
                    inside_parent = True
                    depth = 0

                continue

            # ------------------------------------------
            # Normal MOOSE closing syntax:
            #
            # []
            # ------------------------------------------

            if stripped == "[]":

                if depth == 0:
                    # Closing parent block
                    break

                depth -= 1
                continue

            # ------------------------------------------
            # Old MOOSE closing syntax:
            #
            # [../]
            # ------------------------------------------

            if stripped == "[../]":

                if depth > 0:
                    depth -= 1

                continue

            # ------------------------------------------
            # Opening subsection:
            #
            # [temperature]
            #
            # or
            #
            # [./temperature]
            # ------------------------------------------

            match = re.match(
                r"^\[(?:\./)?([A-Za-z_][A-Za-z0-9_]*)\]$",
                stripped
            )

            if match:

                block_name = match.group(1)

                # Only direct children of parent
                if depth == 0:
                    children.append(
                        block_name
                    )

                depth += 1

        return children

    # --------------------------------------------------
    # Helper:
    # extract full contents of a parent block,
    # including nested blocks
    # --------------------------------------------------

    def extract_parent_block(
        text: str,
        parent_block: str
    ):

        lines = text.splitlines()

        inside_parent = False
        depth = 0
        collected = []

        parent_start = f"[{parent_block}]"

        for line in lines:

            stripped = line.strip()

            # Find parent
            if not inside_parent:

                if stripped == parent_start:
                    inside_parent = True

                continue

            # ------------------------------------------
            # Normal close []
            # ------------------------------------------

            if stripped == "[]":

                if depth == 0:
                    break

                depth -= 1
                collected.append(line)
                continue

            # ------------------------------------------
            # Old-style close [../]
            # ------------------------------------------

            if stripped == "[../]":

                if depth > 0:
                    depth -= 1

                collected.append(line)
                continue

            # ------------------------------------------
            # Opening subsection
            # ------------------------------------------

            match = re.match(
                r"^\[(?:\./)?([A-Za-z_][A-Za-z0-9_]*)\]$",
                stripped
            )

            if match:
                depth += 1

            collected.append(line)

        return "\n".join(
            collected
        )

    # --------------------------------------------------
    # Check every variable has at least one kernel
    # --------------------------------------------------

    variable_names = get_direct_child_blocks(
        content,
        "Variables"
    )

    if not variable_names:

        raise ValueError(
            "No variables found inside "
            "the [Variables] block"
        )

    # --------------------------------------------------
    # Read entire Kernels block
    # --------------------------------------------------

    kernels_content = extract_parent_block(
        content,
        "Kernels"
    )

    if not kernels_content.strip():

        raise ValueError(
            "Kernels block is empty"
        )

    # --------------------------------------------------
    # Find all:
    #
    # variable = temperature
    #
    # variable = 'temperature'
    #
    # variable = "temperature"
    # --------------------------------------------------

    kernel_variables = re.findall(
        r"""
        ^\s*
        variable
        \s*=\s*
        ['"]?
        ([A-Za-z_][A-Za-z0-9_]*)
        ['"]?
        \s*$
        """,
        kernels_content,
        re.MULTILINE | re.VERBOSE
    )

    kernel_variables = set(
        kernel_variables
    )

    # --------------------------------------------------
    # Find variables without any kernel
    # --------------------------------------------------

    variables_without_kernel = [

        variable

        for variable in variable_names

        if variable not in kernel_variables
    ]

    if variables_without_kernel:

        raise ValueError(
            "The following variables do not have "
            "at least one kernel: "
            + ", ".join(
                variables_without_kernel
            )
        )

    # --------------------------------------------------
    # Validation successful
    # --------------------------------------------------

    print(
        f"Validation successful: "
        f"{n_mesh_blocks} mesh blocks and "
        f"{n_material_blocks} material blocks."
    )

    print(
        f"Variables found: "
        f"{', '.join(variable_names)}"
    )

    print(
        f"Variables used in Kernels: "
        f"{', '.join(sorted(kernel_variables))}"
    )

    print(
        "All variables have at least one kernel."
    )

def run_golem_with_image(
    image_name,
    input_file,
    project_dir,
    example_dir,
    docker_dir,
    mesh_test: MeshResults,
    mesh_type: MeshType = MeshType.UNSTRUCTURED,
    output_dir=None,
    nproc=4
):

    # =====================================================
    # 1. NORMALIZE PATHS
    # =====================================================

    project_dir = os.path.abspath(project_dir)
    example_dir = os.path.abspath(example_dir)

    # =====================================================
    # 2. CHECK ORIGINAL INPUT FILE FIRST
    # =====================================================

    source = os.path.join(
        example_dir,
        input_file
    )

    if not os.path.isfile(source):
        raise FileNotFoundError(
            f"GOLEM input file does not exist:\n{source}"
        )

    print("======================================")
    print("ORIGINAL GOLEM INPUT")
    print("======================================")
    print(source)

    # =====================================================
    # 3. CREATE:
    #
    #    project_dir/golem/example
    #
    # Example:
    #
    #    project_dir = /project/ff
    #
    # becomes:
    #
    #    /project/ff/golem/example
    #
    # =====================================================

    golem_dir = os.path.join(
        project_dir,
        "golem"
    )

    example_dest_dir = os.path.join(
        golem_dir,
        "example"
    )

    os.makedirs(
        example_dest_dir,
        exist_ok=True
    )

    # =====================================================
    # 4. COPY INPUT FILE FIRST
    #
    # FROM:
    #   example_dir/model.i
    #
    # TO:
    #   project_dir/golem/example/model.i
    # =====================================================

    destination = os.path.join(
        example_dest_dir,
        input_file
    )

    print("\n======================================")
    print("COPYING GOLEM INPUT")
    print("======================================")

    print("FROM:")
    print(source)

    print("\nTO:")
    print(destination)

    shutil.copy2(
        source,
        destination
    )

    if not os.path.isfile(destination):
        raise RuntimeError(
            "Failed to copy GOLEM input file.\n\n"
            f"Expected:\n{destination}"
        )

    print("\nInput file copied successfully.")

    # =====================================================
    # 5. NOW PREPARE REST OF PROJECT
    # =====================================================

    os.makedirs(
        project_dir,
        exist_ok=True
    )

    if output_dir is None:
        output_dir = os.path.join(
            project_dir,
            "golem_output"
        )

    output_dir = os.path.abspath(output_dir)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # =====================================================
    # 6. DOCKER PATHS
    # =====================================================

    docker_example_dest_dir = os.path.join(
        docker_dir,
        "golem",
        "example"
    )

    docker_original_input = os.path.join(
        docker_example_dest_dir,
        input_file
    )

    print("\n======================================")
    print("PATH MAPPING")
    print("======================================")

    print("Host:")
    print(destination)

    print("\nDocker:")
    print(docker_original_input)

    # =====================================================
    # 7. CREATE TEMPORARY MESH DIRECTORY
    # =====================================================

    tmp_dir = tempfile.mkdtemp(
        prefix="golem_mesh_",
        dir=project_dir
    )

    print("\nTemporary GOLEM directory:")
    print(tmp_dir)

    try:

        # =================================================
        # EXPORT MESH
        # =================================================

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

        # =================================================
        # HOST MESH -> DOCKER MESH PATH
        # =================================================

        relative_exo_path = os.path.relpath(
            exo_path,
            project_dir
        )

        docker_mesh_path = os.path.join(
            docker_dir,
            relative_exo_path
        )

        print("\nHost Exodus mesh:")
        print(exo_path)

        print("\nDocker Exodus mesh:")
        print(docker_mesh_path)

        # =================================================
        # READ THE COPIED INPUT
        #
        # IMPORTANT:
        # We read:
        #
        # project_dir/golem/example/input_file
        #
        # NOT the original example_dir/input_file
        # =================================================

        print("\nReading copied input:")
        print(destination)

        with open(destination, "r") as f:
            model_content = f.read()

        # =================================================
        # VALIDATE
        # =================================================

        validate_golem_input(
            model_content,
            mesh_test,
            mesh_type
        )

        # =================================================
        # REPLACE MESH PATH
        # =================================================

        model_content = model_content.replace(
            "${GOLEM_MESH_FILE}",
            docker_mesh_path
        )

        if "${GOLEM_MESH_FILE}" in model_content:
            raise RuntimeError(
                "ERROR: GOLEM_MESH_FILE replacement failed"
            )

        # =================================================
        # WRITE GENERATED INPUT
        # =================================================

        generated_input = os.path.join(
            tmp_dir,
            "model_generated.i"
        )

        with open(generated_input, "w") as f:
            f.write(model_content)

        relative_generated_input = os.path.relpath(
            generated_input,
            project_dir
        )

        docker_input = os.path.join(
            docker_dir,
            relative_generated_input
        )

        # =================================================
        # OUTPUT DIRECTORIES
        # =================================================

        host_golem_output_dir = os.path.join(
            example_dest_dir,
            "output"
        )

        docker_golem_output_dir = os.path.join(
            docker_example_dest_dir,
            "output"
        )

        # Remove output from previous simulation
        if os.path.isdir(host_golem_output_dir):
            print("\nRemoving old GOLEM output:")
            print(host_golem_output_dir)

            shutil.rmtree(
                host_golem_output_dir
            )

        # =================================================
        # GOLEM COMMAND
        # =================================================

        bash_command = f"""
set -e

echo "======================================"
echo "Starting GOLEM"
echo "======================================"

cd "{docker_example_dest_dir}"

echo ""
echo "Working directory:"
pwd

echo ""
echo "Copied original input:"
ls -lh "{docker_original_input}"

echo ""
echo "Mesh:"
ls -lh "{docker_mesh_path}"

echo ""
echo "Generated input:"
ls -lh "{docker_input}"

echo ""
echo "======================================"
echo "Starting MPI / GOLEM"
echo "======================================"
echo ""

mpiexec --allow-run-as-root \
    -n {nproc} \
    ../golem-opt \
    -i "{docker_input}"

echo ""
echo "======================================"
echo "GOLEM finished"
echo "======================================"

echo ""
echo "Output:"
ls -lah "{docker_golem_output_dir}"
"""

        # =================================================
        # DOCKER COMMAND
        # =================================================

        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "-v",
            f"{project_dir}:{docker_dir}",
            image_name,
            "-c",
            bash_command
        ]

        print("\n======================================")
        print("RUNNING DOCKER")
        print("======================================")

        print("\nHost mount:")
        print(project_dir)

        print("\nDocker mount:")
        print(docker_dir)

        print("\n======================================")
        print("LIVE GOLEM OUTPUT")
        print("======================================\n")

        # =================================================
        # LIVE GOLEM OUTPUT
        # =================================================

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []

        if process.stdout is None:
            raise RuntimeError(
                "Could not access GOLEM stdout."
            )

        for line in process.stdout:

            print(
                line,
                end="",
                flush=True
            )

            output_lines.append(line)

        process.wait()

        full_output = "".join(
            output_lines
        )

        print("\n======================================")
        print("GOLEM PROCESS FINISHED")
        print("======================================")

        print("Return code:", process.returncode)

        if process.returncode != 0:
            raise RuntimeError(
                "GOLEM simulation failed.\n\n"
                f"Return code: {process.returncode}\n\n"
                f"OUTPUT:\n{full_output}"
            )

        # =================================================
        # CHECK GOLEM OUTPUT
        # =================================================

        if not os.path.isdir(
            host_golem_output_dir
        ):
            raise RuntimeError(
                "GOLEM finished but output directory "
                "was not created.\n\n"
                f"Expected:\n"
                f"{host_golem_output_dir}"
            )

        # =================================================
        # COPY RESULTS TO FINAL OUTPUT DIRECTORY
        # =================================================

        print("\n======================================")
        print("COPYING OUTPUT")
        print("======================================")

        print("FROM:")
        print(host_golem_output_dir)

        print("\nTO:")
        print(output_dir)

        shutil.copytree(
            host_golem_output_dir,
            output_dir,
            dirs_exist_ok=True
        )

        # =================================================
        # FIND EXODUS RESULT
        # =================================================

        output_file = os.path.join(
            output_dir,
            "wbgeo-.e"
        )

        if not os.path.exists(output_file):

            exodus_files = []

            for root, dirs, files in os.walk(
                output_dir
            ):
                for filename in files:

                    if (
                        filename.endswith(".e")
                        or filename.endswith(".exo")
                    ):
                        exodus_files.append(
                            os.path.join(
                                root,
                                filename
                            )
                        )

            if not exodus_files:
                raise RuntimeError(
                    "Simulation completed, but no Exodus "
                    "result was found in:\n"
                    f"{output_dir}"
                )

            output_file = max(
                exodus_files,
                key=os.path.getmtime
            )

        print("\n======================================")
        print("SIMULATION SUCCESSFUL")
        print("======================================")

        print("Final Exodus result:")
        print(output_file)

        return {
            "return_code": process.returncode,
            "mesh_file": exo_path,
            "input_file": generated_input,
            "copied_original_input": destination,
            "output_file": output_file,
            "output_dir": output_dir
        }

    finally:

        # =================================================
        # CLEAN ONLY TEMPORARY MESH DIRECTORY
        # =================================================
        #
        # IMPORTANT:
        # This DOES NOT remove:
        #
        # project_dir/golem/example/input_file
        #
        # The copied original input stays there.
        # =================================================

        if os.path.exists(tmp_dir):

            print("\nRemoving temporary GOLEM directory:")
            print(tmp_dir)

            shutil.rmtree(
                tmp_dir,
                ignore_errors=True
            )

            print("Temporary directory removed.")
