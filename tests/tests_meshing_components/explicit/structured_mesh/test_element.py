import unittest
import numpy as np

from core.meshing_components.geometry.Elements import Elements


class TestElements(unittest.TestCase):

    def setUp(self):

        # -------------------------
        # NODE ARRAY:
        # [node_id, x, y, z, surface_id]
        # -------------------------
        self.node_array = np.array([
            [0, 0.0, 0.0, 0.0, 1],
            [1, 1.0, 0.0, 0.0, 1],
            [2, 1.0, 1.0, 0.0, 1],
            [3, 0.0, 1.0, 0.0, 1],
            [4, 0.0, 0.0, 1.0, 1],
            [5, 1.0, 0.0, 1.0, 1],
            [6, 1.0, 1.0, 1.0, 1],
            [7, 0.0, 1.0, 1.0, 1],
        ])

        # -------------------------
        # ELEMENT ARRAY:
        # [element_id, node_ids..., surface_id]
        # simple cube split example
        # -------------------------
        self.element_array = np.array([
            [10, 0, 1, 2, 3, 1],
            [11, 4, 5, 6, 7, 1],
        ])

        self.el = Elements(self.element_array, self.node_array)

    # ------------------------------------------------
    def test_element_ids(self):
        ids = self.el.get_element_ids()
        self.assertEqual(len(ids), 2)
        self.assertTrue(np.array_equal(ids, np.array([10, 11])))

    # ------------------------------------------------
    def test_element_node_ids_shape(self):
        nodes = self.el.get_element_node_ids()
        self.assertEqual(nodes.shape[0], 2)

    # ------------------------------------------------
    def test_total_elements(self):
        self.assertEqual(self.el.total_elements(), 2)

    # ------------------------------------------------
    def test_boundary_methods_return_type(self):

        self.assertIsInstance(self.el.el_front(), np.ndarray)
        self.assertIsInstance(self.el.el_back(), np.ndarray)
        self.assertIsInstance(self.el.el_left(), np.ndarray)
        self.assertIsInstance(self.el.el_right(), np.ndarray)
        self.assertIsInstance(self.el.el_top(), np.ndarray)
        self.assertIsInstance(self.el.el_bottom(), np.ndarray)

    # ------------------------------------------------
    def test_front_back_separation(self):

        front = self.el.el_front()
        back = self.el.el_back()

        self.assertIsInstance(front, np.ndarray)
        self.assertIsInstance(back, np.ndarray)

    # ------------------------------------------------
    def test_left_right_separation(self):

        left = self.el.el_left()
        right = self.el.el_right()

        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)

    # ------------------------------------------------
    def test_top_bottom_separation(self):

        top = self.el.el_top()
        bottom = self.el.el_bottom()

        self.assertIsInstance(top, np.ndarray)
        self.assertIsInstance(bottom, np.ndarray)

    # ------------------------------------------------
    def test_element_by_surface_id(self):

        res = self.el.element_by_surface_id()

        self.assertIsInstance(res, dict)
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    unittest.main()
