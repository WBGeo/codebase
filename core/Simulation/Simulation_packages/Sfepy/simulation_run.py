import os
import tempfile
import subprocess
import shutil
import numpy as np
import meshio
from core.meshing_components.mesh_format.EXUDOS.Exo_format import export_mesh_results_to_exodus

def run_sfepy(input_file, mesh_test, output_dir=None):
    """
    Run SfePy on a mesh.

    If output_dir is None:
        → create temporary output directory
        → remove it after execution

    If output_dir is provided:
        → keep it permanently
    """

    # -------------------------------------------------
    # Handle output directory
    # -------------------------------------------------
    auto_output = False

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sfepy_output_")
        auto_output = True
        print(f"[INFO] Using temporary output directory: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Using user-defined output directory: {output_dir}")

    os.environ["SFEpy_OUTPUT_DIR"] = output_dir

    # -------------------------------------------------
    # Export mesh to Exodus
    # -------------------------------------------------
    exo_buffer = export_mesh_results_to_exodus(mesh_test)

    with tempfile.NamedTemporaryFile(suffix=".exo", delete=False) as tmp_exo:
        tmp_exo_path = tmp_exo.name
        tmp_exo.write(exo_buffer.getbuffer())

    print(f"[INFO] Temporary Exodus file created at: {tmp_exo_path}")

    # -------------------------------------------------
    # Convert to SfePy mesh
    # -------------------------------------------------
    mesh = meshio.read(tmp_exo_path, file_format="exodus")

    mat_ids = [np.full(len(cell_block.data), i, dtype=int)
               for i, cell_block in enumerate(mesh.cells)]
    mesh.cell_data = {"mat_id": mat_ids}

    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as tmp_mesh:
        tmp_mesh_path = tmp_mesh.name

    mesh.write(tmp_mesh_path, file_format="medit")
    os.environ["TEMP_MESH_FILE"] = tmp_mesh_path

    print(f"[INFO] Temporary SfePy mesh created at: {tmp_mesh_path}")

    # -------------------------------------------------
    # Run SfePy
    # -------------------------------------------------
    print("[INFO] Start running SfePy...")
    process = subprocess.Popen(
        ["sfepy-run", input_file],
        env=os.environ,
        cwd=os.getcwd()
    )
    process.wait()
    print("[INFO] SfePy run completed.")

    # -------------------------------------------------
    # List output files
    # -------------------------------------------------
    print("[INFO] SfePy output files in:", output_dir)
    for root, _, files in os.walk(output_dir):
        for f in files:
            print(os.path.join(root, f))

    # -------------------------------------------------
    # Cleanup ONLY temporary helper files
    # -------------------------------------------------
    for tmp_file in [tmp_exo_path, tmp_mesh_path]:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception as e:
            print(f"[WARNING] Could not remove {tmp_file}: {e}")

    # -------------------------------------------------
    # Remove output_dir ONLY if it was auto-created
    for tmp_file in [tmp_exo_path, tmp_mesh_path]:
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
                print(f"[INFO] Removed temporary file: {tmp_file}")
        except Exception as e:
            print(f"[WARNING] Could not remove {tmp_file}: {e}")

    return output_dir
