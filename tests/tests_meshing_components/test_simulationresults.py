import unittest
import numpy as np
from core.object_components import SimulationResults

class TestSimulationResults(unittest.TestCase):

    def setUp(self):
        # Create synthetic timestep data
        self.t1 = 0.0
        self.t2 = 1.0

        self.nodes_t1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        self.nodes_t2 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        self.cells_t1 = np.array([[0, 1]])
        self.cells_t2 = np.array([[0, 1]])

        self.celltypes_t1 = np.array([3])
        self.celltypes_t2 = np.array([3])

        self.node_data_t1 = {"disp": np.array([[0.0, 0.0, 0.0],
                                              [0.1, 0.0, 0.0]])}

        self.cell_data_t1 = {"stress": np.array([1.0])}

    # Initialization
    def test_init_empty(self):
        sim = SimulationResults()

        self.assertEqual(len(sim.nodes_by_time), 0)
        self.assertEqual(len(sim.cells_by_time), 0)

    # Add and retrieve timestep data
    def test_add_timestep_data(self):
        sim = SimulationResults()

        sim.nodes_by_time[self.t1] = self.nodes_t1
        sim.cells_by_time[self.t1] = self.cells_t1
        sim.celltypes_by_time[self.t1] = self.celltypes_t1

        self.assertIn(self.t1, sim.nodes_by_time)
        self.assertTrue(np.array_equal(sim.nodes_by_time[self.t1], self.nodes_t1))

    # Multiple timesteps
    def test_multiple_timesteps(self):
        sim = SimulationResults()

        sim.nodes_by_time[self.t1] = self.nodes_t1
        sim.nodes_by_time[self.t2] = self.nodes_t2

        self.assertEqual(len(sim.nodes_by_time), 2)
        self.assertTrue(np.array_equal(sim.nodes_by_time[self.t2], self.nodes_t2))

    # Node data
    def test_node_data(self):
        sim = SimulationResults()

        sim.node_data_by_time[self.t1] = self.node_data_t1

        self.assertIn("disp", sim.node_data_by_time[self.t1])
        self.assertEqual(sim.node_data_by_time[self.t1]["disp"].shape, (2, 3))

    # Cell data
    def test_cell_data(self):
        sim = SimulationResults()

        sim.cell_data_by_time[self.t1] = self.cell_data_t1

        self.assertIn("stress", sim.cell_data_by_time[self.t1])
        self.assertEqual(len(sim.cell_data_by_time[self.t1]["stress"]), 1)

    # Consistency check (nodes vs cells)
    def test_data_consistency(self):
        sim = SimulationResults()

        sim.nodes_by_time[self.t1] = self.nodes_t1
        sim.cells_by_time[self.t1] = self.cells_t1

        nodes = sim.nodes_by_time[self.t1]
        cells = sim.cells_by_time[self.t1]

        # Ensure indices in cells are valid
        self.assertTrue(np.all(cells < len(nodes)))

    # Missing timestep handling
    def test_missing_timestep(self):
        sim = SimulationResults()

        with self.assertRaises(KeyError):
            _ = sim.nodes_by_time[self.t1]

#######################################
if __name__ == "__main__":
    unittest.main()
