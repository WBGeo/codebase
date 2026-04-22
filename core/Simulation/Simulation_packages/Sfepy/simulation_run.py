import os
import pathlib
import typing
import tempfile
import subprocess
import numpy as np
import meshio

from py_api_wbgeo.nodesapi import wbgeo_component, AnnotatedScriptType
from core.meshing_components.mesh_format.EXUDOS.Exo_format import export_mesh_results_to_exodus


# =====================================================
# TYPES (WBGeo SAFE)
# =====================================================
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

SfepyInputFileType = typing.Annotated[
    str,
    AnnotatedScriptType(
        name='path',
        color='aqua',
        identifier='SfepyInputFileType',
        controlled='RemoteFile|endswith=.py'
    )
]


# =====================================================
# INPUT COMPONENT
# =====================================================
@wbgeo_component(
    description='Input data for SfePy simulation',
    title='SfePy Input',
    color='#b0dfa9',
    border_color='#000000',
    group='Inputs',
    identifier='wbgeo::sfepy_input_data',
    return_name='sfepy_input',
)
def sfepy_input_data(
    name: str = 'Hydrothermal Simulation',
    input_file: SfepyInputFileType = 'examples/synthetic_examples/Model1/input_data/Simulation_input_file/Hydro_thermal.py',
    output_dir: typing.Optional[str] = None
) -> SfepyInputType:

    datadir = pathlib.Path(__file__).parent.parent.parent.resolve()
    full_input_path = os.path.join(datadir, input_file)

    return {
        "name": name,
        "input_file": full_input_path,
        "output_dir": output_dir
    }


# =====================================================
# SIMULATION COMPONENT
# =====================================================
@wbgeo_component(
    description='Simulation using Sfepy',
    title='Simulating with Sfepy',
    color="#74be9d",
    border_color='#000000',
    group='Meshing',
    identifier='Simulate_with_sfepy',
    return_name='Simulation',
)
def run_sfepy(
    sfepy_input_or_file: typing.Union[SfepyInputType, str],
    mesh_test: typing.Any,
    output_dir: typing.Optional[str] = None
) -> SfepyOutputType:

    # -------------------------------------------------
    # INPUT HANDLING
    # -------------------------------------------------
    if isinstance(sfepy_input_or_file, dict):
        input_file = sfepy_input_or_file["input_file"]
        if output_dir is None:
            output_dir = sfepy_input_or_file.get("output_dir")
    else:
        input_file = sfepy_input_or_file

    # -------------------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------------------
    is_temp = False

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sfepy_output_")
        is_temp = True
        print(f"[INFO] Temp output: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] User output: {output_dir}")

    os.environ["SFEpy_OUTPUT_DIR"] = output_dir

    # -------------------------------------------------
    # EXPORT MESH
    # -------------------------------------------------
    exo_buffer = export_mesh_results_to_exodus(mesh_test)

    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp_exo:
        tmp_exo_path = tmp_exo.name
        tmp_exo.write(exo_buffer.getbuffer())

    mesh = meshio.read(tmp_exo_path, file_format="exodus")

    mat_ids = [
        np.full(len(cell_block.data), i, dtype=int)
        for i, cell_block in enumerate(mesh.cells)
    ]
    mesh.cell_data = {"mat_id": mat_ids}

    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as tmp_mesh:
        tmp_mesh_path = tmp_mesh.name

    mesh.write(tmp_mesh_path, file_format="medit")
    os.environ["TEMP_MESH_FILE"] = tmp_mesh_path

    # -------------------------------------------------
    # RUN SFEpy
    # -------------------------------------------------
    print("[INFO] Running SfePy...")
    process = subprocess.Popen(
        ["sfepy-run", input_file],
        env=os.environ,
        cwd=os.getcwd()
    )
    process.wait()
    print("[INFO] SfePy finished.")

    # -------------------------------------------------
    # CLEAN TEMP FILES
    # -------------------------------------------------
    for tmp_file in [tmp_exo_path, tmp_mesh_path]:
        try:
            os.remove(tmp_file)
        except:
            pass

    # -------------------------------------------------
    # RETURN (WBGeo SAFE)
    # -------------------------------------------------
    return {
        "output_dir": output_dir,
        "is_temp": is_temp
    }


# =====================================================
# SAVE OUTPUT COMPONENT
# =====================================================
@wbgeo_component(
    title="Save outputs of simulation",
    description="Save outputs",
    group="Outputs",
    identifier="wbgeo::save_outputs",
    return_name="simulation_output",
)
def save_outputs(sim_output: SfepyOutputType) -> SfepyOutputType:
    return sim_output
