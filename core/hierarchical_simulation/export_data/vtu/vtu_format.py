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

    def _validate_face_array(nodes, label):
        arr = np.asarray(nodes)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                f"Boundary node array '{label}' must have shape (N, 4) "
                "with columns [NodeID, X, Y, Z]. "
                f"Got shape {arr.shape}."
            )
        return arr

    def _write_boundary_frame(frame_nodes, folder, frame_name=None):
        os.makedirs(folder, exist_ok=True)
        for face, nodes in frame_nodes.items():
            arr = _validate_face_array(
                nodes,
                f"{frame_name}/{face}" if frame_name else str(face),
            )
            filename = os.path.join(folder, f"{face}_boundary_nodes.csv")
            np.savetxt(
                filename, arr, delimiter=",",
                header=header, comments="", fmt=fmt
            )
            prefix = f"{frame_name} / " if frame_name else ""
            print(
                f"{prefix}{str(face).capitalize()} boundary nodes saved: "
                f"{filename} ({arr.shape[0]} points)"
            )

    # New hierarchical format:
    # {
    #     "big_frame":   {face: Nx4, ...},
    #     "small_frame": {face: Nx4, ...},
    # }
    # Both frames contain the same mesh1 node IDs, but coordinates are expressed
    # in the final big-mesh and small-mesh coordinate systems respectively.
    if (
        isinstance(boundary_nodes, dict)
        and "big_frame" in boundary_nodes
        and "small_frame" in boundary_nodes
    ):
        big_frame = boundary_nodes["big_frame"]
        small_frame = boundary_nodes["small_frame"]

        if not isinstance(big_frame, dict) or not isinstance(small_frame, dict):
            raise ValueError(
                "boundary_nodes['big_frame'] and boundary_nodes['small_frame'] "
                "must both be dictionaries of face arrays."
            )

        # Verify that corresponding faces carry the same mesh1 node IDs.
        common_faces = set(big_frame).intersection(small_frame)
        for face in common_faces:
            big_arr = _validate_face_array(big_frame[face], f"big_frame/{face}")
            small_arr = _validate_face_array(small_frame[face], f"small_frame/{face}")
            if big_arr.shape[0] != small_arr.shape[0] or not np.array_equal(
                big_arr[:, 0].astype(np.int64),
                small_arr[:, 0].astype(np.int64),
            ):
                raise ValueError(
                    f"Boundary node IDs differ between big_frame and "
                    f"small_frame for face '{face}'."
                )

        _write_boundary_frame(
            big_frame, os.path.join(boundary_folder, "big_frame"), "big_frame"
        )
        _write_boundary_frame(
            small_frame, os.path.join(boundary_folder, "small_frame"), "small_frame"
        )

    else:
        # Backward-compatible legacy format: {face: Nx4, ...}
        _write_boundary_frame(boundary_nodes, boundary_folder)
