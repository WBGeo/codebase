import numpy as np
from core.meshing_components.geometry.Nodes import Nodes

class Elements(Nodes):
    def __init__(self, element_array, node_array):
        """
        Initializes the Elements class.

        Args:
            element_array (np.ndarray): Array of elements with columns [element_id, node_id_1, ..., node_id_n, surface_id].
            node_array (np.ndarray): Array of nodes with columns [node_id, x, y, z, surface_id].
        """
        super().__init__(node_array)  # Initializes the Nodes (and indirectly Points) class.
        self.element_array = element_array
        self.element_ids = element_array[:, 0].astype(int)
        self.element_node_ids = element_array[:, 1:-1].astype(int)  # Assuming elements are between the first and last columns
        self.surface_ids = element_array[:, -1].astype(int)

    def get_element_ids(self):
        """Returns the element IDs."""
        return self.element_ids

    def get_element_node_ids(self):
        """Returns the node IDs for each element."""
        return self.element_node_ids

    def total_elements(self):
        """Returns the total number of elements."""
        return len(self.element_ids)

    def elements_on_boundaries(self):
        """Returns a dictionary categorizing elements on each boundary."""
        elm_boundaries = {
            'front': self.el_front(),
            'back': self.el_back(),
            'bottom': self.el_bottom(),
            'top': self.el_top(),
            'left': self.el_left(),
            'right': self.el_right(),
        }
        return elm_boundaries

    def el_front(self):
        """Returns element IDs for front-facing elements (y = min(y))."""
        min_y = np.min(self.coordinates[:, 1])
        front_elements = []
        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.min(nodes[:, 1]) == min_y:
                front_elements.append(element[0].astype(int))
        return np.array(front_elements)

    def el_back(self):
        """Returns element IDs for back-facing elements (y = max(y))"""
        max_y = np.max(self.coordinates[:, 1])
        back_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.max(nodes[:, 1]) == max_y:
                back_elements.append(element[0].astype(int))

        return np.array(back_elements)

    def el_bottom(self):
        """Returns element IDs for bottom-facing elements (z = min(z))"""
        min_z = np.min(self.coordinates[:, 2])
        bottom_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.min(nodes[:, 2]) == min_z:
                bottom_elements.append(element[0].astype(int))

        return np.array(bottom_elements)

    def el_top(self):
        """Returns element IDs for top-facing elements (z = max(z))"""
        max_z = np.max(self.coordinates[:, 2])
        top_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.max(nodes[:, 2]) == max_z:
                top_elements.append(element[0].astype(int))

        return np.array(top_elements)

    def el_right(self):
        """Returns element IDs for right-facing elements (x = max(x))"""
        max_x = np.max(self.coordinates[:, 0])
        right_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.max(nodes[:, 0]) == max_x:
                right_elements.append(element[0].astype(int))

        return np.array(right_elements)

    def el_left(self):
        """Returns element IDs for left-facing elements (x = min(x))"""
        min_x = np.min(self.coordinates[:, 0])
        left_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:9].astype(int)]
            if np.min(nodes[:, 0]) == min_x:
                left_elements.append(element[0].astype(int))

        return np.array(left_elements)

    def element_by_surface_id(self):
        """Returns a dictionary categorizing elements by their surface ids."""
        elements_by_surface_id = {}
        unique_surface_ids = np.unique(self.surface_ids)
        for surface_id in unique_surface_ids:
            elements_n = self.get_elements_by_surface_id(surface_id)
            elements_by_surface_id[int(surface_id)] = np.array(elements_n)
        return elements_by_surface_id

    def get_elements_by_surface_id(self, surface_id):
        """Returns the elements associated with a particular surface ID."""
        return self.element_node_ids[self.surface_ids == surface_id]

