import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def adjust_z_values(all_points_array: NDArray[np.floating], n_gx: int,n_gy: int, z_threshold: float = 0.2,
    tolerance: float = 0.5) -> NDArray[np.floating]:
    """
    Adjusts z-values of points in consecutive rows if they are within a tolerance distance.

    Args:
        - all_points_array (np.array): Array containing points for each surface and its intermediate points, where:
                                     - x = all_points_array[:, :n_gx * n_gy]
                                     - y = all_points_array[:, n_gx * n_gy: 2 * n_gx * n_gy]
                                     - z = all_points_array[:, 2 * n_gx * n_gy: 3 * n_gx * n_gy]
                                     - ids = all_points_array[:, -n_gx * n_gy:]
        - n_gx (int): Grid size in x direction.
        - n_gy (int): Grid size in y direction.
        - z_threshold (float): Value for moving a point in z direction.
        - tolerance (float): Tolerance for comparing location of two points.

    Returns:
        np.array: Adjusted array with updated z values.
    """

    z_values: NDArray[np.floating] = all_points_array[:, 2 * n_gx * n_gy: 3 * n_gx * n_gy]

    adjusted = False
    c=0

    while True:

        adjusted = False

        # Loop through consecutive rows
        for i in range(all_points_array.shape[0] - 1):
            # Compare the z-values of the ith row and (i+1)th row
            z_row_i: NDArray[np.floating] = z_values[i]
            z_row_i_plus_1: NDArray[np.floating] = z_values[i + 1]

            # Check if any corresponding points in the two rows have the same z-value (within tolerance)
            matching_indices: NDArray[np.bool_] = z_row_i_plus_1 - z_row_i <= tolerance

            if np.any(matching_indices):

                # Adjust the z-values in row i
                z_values[i][matching_indices] -= z_threshold
                adjusted = True  # Indicate that an adjustment was made
                # Break the loop to restart from the beginning
                break

        # If no adjustments were made in this pass, we can exit the loop
        if not adjusted:
            break

    # Update the original array with adjusted z-values
    all_points_array[:, 2 * n_gx * n_gy: 3 * n_gx * n_gy] = z_values

    return all_points_array


def create_hexahedral_elements_with_nodes(adjusted_array: NDArray[np.floating], n_gx: int, n_gy: int
    ) -> Tuple[NDArray[np.integer], NDArray[np.floating]]:
    """
    Creates hexahedral elements and renumbers the nodes for consistency.
    Saves both elements and nodes data.

    Args:
        - adjusted_array (np.array): Array of points with shape (n_layers, 3 * n_gx * n_gy + 1).
        - n_gx (int): Grid size in the x direction.
        - n_gy (int): Grid size in the y direction.

    Returns:
        tuples containing:
            - elements_array (np.array): Array of elements where each row contains:
                [element_id, node0, node1, node2, node3, node4, node5, node6, node7, surface_id].
            - nodes_array (np.array): Array of nodes where each row contains:
                [node_id, x, y, z, surface_id].
    """

    elements: list[list[int]] = []
    nodes: list[list[float]] = []
    element_id: int = 0

    num_layers = adjusted_array.shape[0]
    num_nodes_per_layer: int = n_gx * n_gy

    # Dictionary to map old node indices to new node numbers
    node_mapping: dict[int, int]  = {}
    new_node_id: int = 0

    # Iterate through layers (excluding the topmost layer)
    for layer in range(num_layers - 1):

        for j in range(n_gy - 1):

            for i in range(n_gx - 1):

                # Calculate the node indices for the current hexahedral element
                node_indices: list[int] = [
                    i + j * n_gx + layer * num_nodes_per_layer,
                    i + j * n_gx + 1 + layer * num_nodes_per_layer,
                    i + (j + 1) * n_gx + 1 + layer * num_nodes_per_layer,
                    i + (j + 1) * n_gx + layer * num_nodes_per_layer,
                    i + j * n_gx + (layer + 1) * num_nodes_per_layer,
                    i + j * n_gx + 1 + (layer + 1) * num_nodes_per_layer,
                    i + (j + 1) * n_gx + 1 + (layer + 1) * num_nodes_per_layer,
                    i + (j + 1) * n_gx + (layer + 1) * num_nodes_per_layer,
                ]
                new_nodes: list[int] = []
                # Map old node indices to new numbering and add to nodes array if not already added

                for old_index in node_indices:

                    if old_index not in node_mapping:
                        # Extract x, y, z coordinates and surface ID from adjusted_array
                        layer_id: int = old_index // num_nodes_per_layer
                        within_layer_index: int = old_index % num_nodes_per_layer
                        x: float = adjusted_array[layer_id, within_layer_index]
                        y: float = adjusted_array[layer_id, within_layer_index + num_nodes_per_layer]
                        z: float = adjusted_array[layer_id, within_layer_index + 2 * num_nodes_per_layer]
                        node_surface_id:float = adjusted_array[layer_id, within_layer_index + 3 * num_nodes_per_layer]
                        # Get always the surface_id of the nodes in the lowest layer
                        elem_surf: float = adjusted_array[layer_id - 1, within_layer_index + 3 * num_nodes_per_layer]
                        # Append node data and map old to new
                        nodes.append([new_node_id, x, y, z, node_surface_id])
                        node_mapping[old_index] = new_node_id
                        new_node_id += 1
                    new_nodes.append(node_mapping[old_index])

                # Calculate the mode of the surface IDs
                element_surface_id: float = elem_surf

                # Append the element information, including the mode of surface IDs
                elements.append([element_id] + new_nodes + [element_surface_id])
                element_id += 1

    # Convert the lists to numpy arrays for saving
    elements_array: NDArray[np.integer] = np.array(elements, dtype=int)
    nodes_array_full: NDArray[np.floating] = np.asarray(nodes, dtype=float)

    # Drop surface_id column → [node_id, x, y, z]
    nodes_array: NDArray[np.floating] = nodes_array_full[:, :-1]

    return elements_array, nodes_array


