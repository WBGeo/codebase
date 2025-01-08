import numpy as np

class Exos_inputs:
    def __init__(self, nodes_array, elements_array):
        """
        Initializes the Exos class.

        Args:
            nodes_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
            elements_array (np.ndarray): Array of elements with columns [element_id, node_id_1, node_id_2, node_id_3, node_id_4, surface_id].
        """
        self.nodes_array = nodes_array
        self.elements_array = elements_array
        self.bound = self.Bound(self)

    def num_nodes(self):
        """Returns the total number of nodes."""
        return self.nodes_array.shape[0]

    def num_elements(self):
        """Returns the total number of elements."""
        return self.elements_array.shape[0]

    def coord(self):
        """Returns the x, y, z coordinates of all nodes."""
        return self.nodes_array[:, 1:4]

    def surface_id(self):
        """Returns the surface IDs of all nodes."""
        return self.nodes_array[:, -1]

    class Bound:
        def __init__(self, exos_instance):
            """
            Initializes the Bound class.

            Args:
                exos_instance (Exos): Instance of the parent Exos class.
            """
            self.exos_instance = exos_instance
            self.nodes_array = exos_instance.nodes_array
            self.elements_array = exos_instance.elements_array
            self.x_coords = self.nodes_array[:, 1]
            self.y_coords = self.nodes_array[:, 2]
            self.z_coords = self.nodes_array[:, 3]
            self.surface_ids = self.nodes_array[:, -1]

        # Front
        def front(self):
            """Returns node IDs of front nodes (x = min(x)), excluding nodes with z_min, z_max, y_min, and y_max."""
            min_x = np.min(self.x_coords)
            max_x = np.max(self.x_coords)
            max_z = np.max(self.z_coords)
            min_z = np.min(self.z_coords)
            min_y = np.min(self.y_coords)

            front_nodes = self.nodes_array[self.y_coords == min_y]
            front_nodes = front_nodes[~np.isin(front_nodes[:, 3], [min_z, max_z])]  # Exclude nodes with z_min and z_max
            front_nodes = front_nodes[~np.isin(front_nodes[:, 1], [min_x, max_x])]  # Exclude nodes with y_min and y_max

            return front_nodes[:, 0].astype(int)  # Return node IDs

        def front_coords(self):
            """Returns coordinates of front nodes."""
            return self._get_boundary_coords(self.front())

        def front_surface_id(self):
            """Returns surface IDs of front nodes."""
            return self._get_boundary_surface_id(self.front())

        # Back
        def back(self):
            """Returns node IDs of back nodes (x = max(x)), excluding nodes with z_min, z_max, y_min, and y_max."""
            max_x = np.max(self.x_coords)
            min_x = np.min(self.x_coords)
            max_z = np.max(self.z_coords)
            min_z = np.min(self.z_coords)
            max_y = np.max(self.y_coords)

            back_nodes = self.nodes_array[self.y_coords == max_y]
            back_nodes = back_nodes[~np.isin(back_nodes[:, 3], [min_z, max_z])]  # Exclude nodes with z_min and z_max
            back_nodes = back_nodes[~np.isin(back_nodes[:, 1], [min_x, max_x])]  # Exclude nodes with y_min and y_max

            return back_nodes[:, 0].astype(int)  # Return node IDs

        def back_coords(self):
            """Returns coordinates of back nodes."""
            return self._get_boundary_coords(self.back())

        def back_surface_id(self):
            """Returns surface IDs of back nodes."""
            return self._get_boundary_surface_id(self.back())

        # Bottom
        def bottom(self):
            """Returns node IDs of bottom nodes (z = min(z)), no exclusions."""
            min_z = np.min(self.z_coords)
            bottom_nodes = self.nodes_array[self.z_coords == min_z]
            return bottom_nodes[:, 0].astype(int)  # Return node IDs

        def bottom_coords(self):
            """Returns coordinates of bottom nodes."""
            return self._get_boundary_coords(self.bottom())

        def bottom_surface_id(self):
            """Returns surface IDs of bottom nodes."""
            return self._get_boundary_surface_id(self.bottom())

        # Top
        def top(self):
            """Returns node IDs of top nodes (z = max(z)), no exclusions."""
            max_z = np.max(self.z_coords)
            top_nodes = self.nodes_array[self.z_coords == max_z]
            return top_nodes[:, 0].astype(int)  # Return node IDs

        def top_coords(self):
            """Returns coordinates of top nodes."""
            return self._get_boundary_coords(self.top())

        def top_surface_id(self):
            """Returns surface IDs of top nodes."""
            return self._get_boundary_surface_id(self.top())

        # Right
        def right(self):
            """Returns node IDs of right nodes (y = min(y)), excluding nodes with z_min and z_max."""
            max_x = np.max(self.x_coords)
            max_z = np.max(self.z_coords)
            min_z = np.min(self.z_coords)

            right_nodes = self.nodes_array[self.x_coords == max_x]
            right_nodes = right_nodes[~np.isin(right_nodes[:, 3], [min_z, max_z])]  # Exclude nodes with z_min and z_max

            return right_nodes[:, 0].astype(int)  # Return node IDs

        def right_coords(self):
            """Returns coordinates of right nodes."""
            return self._get_boundary_coords(self.right())

        def right_surface_id(self):
            """Returns surface IDs of right nodes."""
            return self._get_boundary_surface_id(self.right())

        # Left
        def left(self):
            """Returns node IDs of left nodes (y = max(y)), excluding nodes with z_min and z_max."""
            min_x = np.min(self.x_coords)
            min_z = np.min(self.z_coords)
            max_z = np.max(self.z_coords)

            left_nodes = self.nodes_array[self.x_coords == min_x]
            left_nodes = left_nodes[~np.isin(left_nodes[:, 3], [min_z, max_z])]  # Exclude nodes with z_min and z_max

            return left_nodes[:, 0].astype(int)  # Return node IDs

        def left_coords(self):
            """Returns coordinates of left nodes."""
            return self._get_boundary_coords(self.left())

        def left_surface_id(self):
            """Returns surface IDs of left nodes."""
            return self._get_boundary_surface_id(self.left())

        # Helper methods
        def _get_boundary_coords(self, node_ids):
            """Helper method to get the coordinates of given node_ids."""
            return self.nodes_array[np.isin(self.nodes_array[:, 0], node_ids), 1:4]

        def _get_boundary_surface_id(self, node_ids):
            """Helper method to get the surface IDs of given node_ids."""
            return self.nodes_array[np.isin(self.nodes_array[:, 0], node_ids), -1]

    # Element Methods
    def element_id(self):
        """Returns the element IDs of all elements."""
        return self.elements_array[:, 0]

    def element_surface_id(self):
        """Returns the surface IDs of all elements."""
        return self.elements_array[:, -1]

    def element_nodes(self):
        """Returns the node IDs for each element."""
        return self.elements_array[:, 1:9]  # Assuming 4 nodes per element (quadrilateral)

    def el_front(self):
        """Returns element IDs for front-facing elements (y = min(y))"""
        min_y = np.min(self.nodes_array[:, 2])
        front_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.min(nodes[:, 2]) == min_y:
                front_elements.append(element[0])

        return np.array(front_elements)

    def el_back(self):
        """Returns element IDs for back-facing elements (y = max(y))"""
        max_y = np.max(self.nodes_array[:, 2])
        back_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.max(nodes[:, 2]) == max_y:
                back_elements.append(element[0])

        return np.array(back_elements)

    def el_bottom(self):
        """Returns element IDs for bottom-facing elements (z = min(z))"""
        min_z = np.min(self.nodes_array[:, 3])
        bottom_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.min(nodes[:, 3]) == min_z:
                bottom_elements.append(element[0])

        return np.array(bottom_elements)

    def el_top(self):
        """Returns element IDs for top-facing elements (z = max(z))"""
        max_z = np.max(self.nodes_array[:, 3])
        top_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.max(nodes[:, 3]) == max_z:
                top_elements.append(element[0])

        return np.array(top_elements)

    def el_right(self):
        """Returns element IDs for right-facing elements (x = max(x))"""
        max_x = np.max(self.nodes_array[:, 1])
        right_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.min(nodes[:, 1]) == max_x:
                right_elements.append(element[0])

        return np.array(right_elements)

    def el_left(self):
        """Returns element IDs for left-facing elements (x = min(x))"""
        min_x = np.min(self.nodes_array[:, 1])
        left_elements = []

        for element in self.elements_array:
            nodes = self.nodes_array[element[1:9].astype(int)]
            if np.max(nodes[:, 1]) == min_x:
                left_elements.append(element[0])

        return np.array(left_elements)

    # Helper Methods
    def element_coords(self, element_ids):
        """Returns the coordinates of the nodes of given elements."""
        element_coords = []
        for element_id in element_ids:
            element = self.elements_array[element_id]
            coords = self.nodes_array[element[1:9].astype(int), 1:8]
            element_coords.append(coords)
        return np.array(element_coords)

    def element_surface_id(self, element_ids):
        """Returns the surface ID for the given element IDs."""
        return self.elements_array[element_ids, -1]

    def get_elements_by_surface_id(self, surface_id):
        """
        Gets the elements corresponding to a specific surface ID, excluding the surface ID column.

        Args:
            surface_id (int): The surface ID to filter elements for.

        Returns:
            np.ndarray: Array of elements with the specified surface ID, excluding the surface ID column.
        """
        # Filter elements based on the surface ID
        filtered_elements = self.elements_array[self.elements_array[:, -1] == surface_id]
        # Exclude the last column (surface ID)
        return filtered_elements[:, 1:-1]
