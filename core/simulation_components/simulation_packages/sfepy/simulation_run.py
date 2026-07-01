import logging
import os
import pathlib
import typing
import tempfile
import subprocess
import numpy as np
import meshio

from enum import StrEnum

logger = logging.getLogger(__name__)
from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.object_components import MeshResults


# =========================================================
# INTERNAL ENUM (KEEP THIS — SAFE AND CLEAN)
# =========================================================
class MeshType(StrEnum):
    UNSTRUCTURED = "unstr"
    STRUCTURED = "str"
    IMPLICIT = "imp"


# =========================================================
# WBGeo SAFE TYPE (IMPORTANT FIX)
# =========================================================
MeshTypeInput = typing.Annotated[
    str,
    AnnotatedScriptType(
        name="MeshType",
        identifier="MeshType",
        controlled="Select|unstr|str|imp"
    )
]


# =========================================================
# INPUT TYPES
# =========================================================
SfepyInputFileType = typing.Annotated[
    str,
    AnnotatedScriptType(
        name='path',
        color='aqua',
        identifier='SfepyInputFileType',
        controlled='RemoteFile|endswith=.py'
    )
]

SfepyInputType = typing.Annotated[
    dict,
    AnnotatedScriptType(
        name="SfepyInput",
        identifier="SfepyInputType"
    )
]

SfepyOutputType = typing.Annotated[
    dict,
    AnnotatedScriptType(
        name="SfepyOutput",
        identifier="SfepyOutputType"
    )
]


# =========================================================
# INPUT COMPONENT
# =========================================================
@wbgeo_component(
    description='Input data for SfePy simulation',
    title='SfePy Input',
    color="#e5d016",
    border_color='#000000',
    group='simulation_components',
    identifier='wbgeo::sfepy_input_data',
    return_name='sfepy_input',
)
def sfepy_input_data(
    name: str = 'Hydrothermal simulation_components',
    input_file: SfepyInputFileType =
        'examples/synthetic_examples/model1/input_data/simulation_input_file/Hydro_thermal.py',
    output_dir: typing.Optional[str] = None
) -> SfepyInputType:

    datadir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
    full_input_path = os.path.join(datadir, input_file)

    return {
        "name": name,
        "input_file": full_input_path,
        "output_dir": output_dir
    }


# =========================================================
# SIMULATION COMPONENT (FIXED)
# =========================================================
@wbgeo_component(
    description='Simulation using Sfepy',
    title='Simulating with Sfepy',
    color="#e5d016",
    border_color='#000000',
    group='simulation_components',
    identifier='wbgeo::simulate_with_sfepy',
    return_name='Simulation',
)
def run_sfepy(
    sfepy_input_or_file: typing.Union[SfepyInputType, str],
    mesh_test: MeshResults,
    mesh_type: MeshTypeInput = "unstr",
    output_dir: typing.Optional[str] = None
) -> SfepyOutputType:

    # =====================================================
    # NORMALIZE MESH TYPE (STRING -> ENUM)
    # =====================================================
    try:
        mesh_type_enum = MeshType(mesh_type)
    except ValueError:
        raise ValueError(
            f"Invalid mesh_type='{mesh_type}'. "
            f"Valid options are: {[m.value for m in MeshType]}"
        )

    # =====================================================
    # INPUT HANDLING
    # =====================================================
    if isinstance(sfepy_input_or_file, dict):
        input_file = sfepy_input_or_file["input_file"]
        if output_dir is None:
            output_dir = sfepy_input_or_file.get("output_dir")
    else:
        input_file = sfepy_input_or_file

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================
    is_temp = False

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sfepy_output_")
        is_temp = True
        logger.info("Temp output: %s", output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("User output: %s", output_dir)

    os.environ["SFEpy_OUTPUT_DIR"] = output_dir

    # =====================================================
    # EXPORT MESH
    # =====================================================
    exo_buffer = export_mesh_results_to_exodus(
        mesh_test,
        type=mesh_type_enum
    )

    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp_exo:
        tmp_exo_path = tmp_exo.name
        tmp_exo.write(exo_buffer.getbuffer())

    mesh = meshio.read(tmp_exo_path, file_format="exodus")

    # add material IDs
    mat_ids = [
        np.full(len(cell_block.data), i, dtype=int)
        for i, cell_block in enumerate(mesh.cells)
    ]
    mesh.cell_data = {"mat_id": mat_ids}

    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as tmp_mesh:
        tmp_mesh_path = tmp_mesh.name

    mesh.write(tmp_mesh_path, file_format="medit")
    os.environ["TEMP_MESH_FILE"] = tmp_mesh_path

    # =====================================================
    # RUN SFEpy
    # =====================================================
    logger.info("Running SfePy...")
    process = subprocess.Popen(
        ["sfepy-run", input_file],
        env=os.environ,
        cwd=os.getcwd()
    )
    process.wait()
    logger.info("SfePy finished.")

    # =====================================================
    # CLEANUP
    # =====================================================
    for tmp_file in [tmp_exo_path, tmp_mesh_path]:
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    # =====================================================
    # RETURN
    # =====================================================
    return {
        "output_dir": output_dir,
        "is_temp": is_temp
    }


# =========================================================
# SAVE OUTPUT COMPONENT — placeholder, not yet implemented
# =========================================================
def save_outputs(sim_output: SfepyOutputType) -> SfepyOutputType:
    # TODO: implement output saving logic
    return sim_output
