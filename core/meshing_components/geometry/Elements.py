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

    def __init__(self, element_array, node_array):

        super().__init__(node_array)

        self.element_array = np.asarray(element_array)

        self.element_ids = self.element_array[:, 0].astype(int)
        self.element_node_ids = self.element_array[:, 1:-1].astype(int)
        self.surface_ids = self.element_array[:, -1].astype(int)

    #################
    # basic getters
    ################
    def get_element_ids(self):
        return self.element_ids

    def get_element_node_ids(self):
        return self.element_node_ids

    def total_elements(self):
        return len(self.element_ids)

    ##########################
    # boundary helpers (FIXED)
    #########################
    def el_front(self):

        min_y = np.min(self.coordinates[:, 1])

        front_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.min(nodes[:, 1]) == min_y:
                front_elements.append(int(element[0]))

        return np.array(front_elements)

    def el_back(self):

        max_y = np.max(self.coordinates[:, 1])

        back_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.max(nodes[:, 1]) == max_y:
                back_elements.append(int(element[0]))

        return np.array(back_elements)


    def el_bottom(self):

        min_z = np.min(self.coordinates[:, 2])
        bottom_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.min(nodes[:, 2]) == min_z:
                bottom_elements.append(int(element[0]))

        return np.array(bottom_elements)


    def el_top(self):

        max_z = np.max(self.coordinates[:, 2])
        top_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.max(nodes[:, 2]) == max_z:
                top_elements.append(int(element[0]))

        return np.array(top_elements)


    def el_right(self):

        max_x = np.max(self.coordinates[:, 0])
        right_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.max(nodes[:, 0]) == max_x:
                right_elements.append(int(element[0]))

        return np.array(right_elements)


    def el_left(self):

        min_x = np.min(self.coordinates[:, 0])
        left_elements = []

        for element in self.element_array:
            nodes = self.coordinates[element[1:-1].astype(int)]

            if np.min(nodes[:, 0]) == min_x:
                left_elements.append(int(element[0]))

        return np.array(left_elements)


    ############
    # grouping
    ############
    def element_by_surface_id(self):

        elements_by_surface_id = {}

        for sid in np.unique(self.surface_ids):
            elements_by_surface_id[int(sid)] = self.get_elements_by_surface_id(sid)

        return elements_by_surface_id


    def get_elements_by_surface_id(self, surface_id):
        return self.element_node_ids[self.surface_ids == surface_id]
