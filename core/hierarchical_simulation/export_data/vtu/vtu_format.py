import os
import numpy as np

from core.object_components import MeshResults
from core.meshing_components.mesh_format.vtu.VTU_format import (
    export_mesh_results_to_vtu
)


def export_meshes_vtu(mesh1: MeshResults, mesh2: MeshResults, boundary_nodes: dict,
    mesh1_filename: str = "mesh1.vtu", mesh2_filename: str = "mesh2.vtu", boundary_folder: str = "boundary_nodes"
):
    """
    Export two hierarchical meshes to VTU format and save the
    matching boundary nodes of mesh1 separated by faces.

    Args:
        mesh1: Larger coarse mesh.
        mesh2: Smaller refined mesh.
        boundary_nodes:
            Dictionary containing face node arrays:
            {
                "top": Nx4,
                "bottom": Nx4,
                "left": Nx4,
                "right": Nx4,
                "front": Nx4,
                "back": Nx4
            }

            Each row: [NodeID, X, Y, Z]

        mesh1_filename: VTU filename for mesh1.
        mesh2_filename: VTU filename for mesh2.
        boundary_folder: Folder where boundary CSV files are saved.
    """

    os.makedirs(boundary_folder, exist_ok=True)


    # --------------------------------------------------
    # Export mesh1
    # --------------------------------------------------

    buf1 = export_mesh_results_to_vtu(mesh1)

    with open(mesh1_filename, "wb") as f:
        f.write(buf1.getbuffer())

    print(
        f"Mesh1 exported: {mesh1_filename} "
        f"({len(mesh1.nodes)} nodes, {len(mesh1.elements)} elements)"
    )


    # --------------------------------------------------
    # Export mesh2
    # --------------------------------------------------

    buf2 = export_mesh_results_to_vtu(mesh2)

    with open(mesh2_filename, "wb") as f:
        f.write(buf2.getbuffer())

    print(
        f"Mesh2 exported: {mesh2_filename} "
        f"({len(mesh2.nodes)} nodes, {len(mesh2.elements)} elements)"
    )


    # --------------------------------------------------
    # Save boundary nodes by face
    # --------------------------------------------------

    fmt = ['%d', '%.6f', '%.6f', '%.6f']
    header = "NodeID,X,Y,Z"

    for face, nodes in boundary_nodes.items():

        filename = os.path.join(
            boundary_folder,
            f"{face}_boundary_nodes.csv"
        )

        np.savetxt(filename, nodes, delimiter=",",
            header=header, comments="", fmt=fmt)

        print(
            f"{face.capitalize()} boundary nodes saved: "
            f"{filename} ({nodes.shape[0]} points)"
        )
