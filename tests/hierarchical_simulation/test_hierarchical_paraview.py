
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.hierarchical_simulation.visualization import hierarchical_paraview_visualization as pvmod


class TestParaViewHelpers(unittest.TestCase):

    def test_validate_exodus_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.exo"
            path.write_bytes(b"x")
            self.assertEqual(pvmod._validate_exodus_file(path), str(path.resolve()))

    def test_validate_missing_exodus_file(self):
        with self.assertRaises(FileNotFoundError):
            pvmod._validate_exodus_file("/definitely/not/a/mesh.exo")

    def test_explicit_paraview_executable(self):
        with tempfile.TemporaryDirectory() as td:
            executable = Path(td) / "paraview"
            executable.write_text("")
            self.assertEqual(
                pvmod._find_paraview_executable(executable),
                str(executable.resolve()),
            )

    @patch.object(pvmod.subprocess, "Popen")
    @patch.object(pvmod, "_find_paraview_executable", return_value="/pv/paraview")
    def test_open_two_hierarchical_files(self, find_exe, popen):
        process = Mock()
        popen.return_value = process

        with tempfile.TemporaryDirectory() as td:
            m1 = Path(td) / "mesh1.exo"
            m2 = Path(td) / "mesh2.exo"
            m1.write_bytes(b"1")
            m2.write_bytes(b"2")

            result = pvmod.open_hierarchical_exodus_in_paraview(m1, m2)

            self.assertIs(result, process)
            popen.assert_called_once_with(
                ["/pv/paraview", str(m1.resolve()), str(m2.resolve())]
            )
            process.wait.assert_not_called()

    @patch.object(pvmod.subprocess, "Popen")
    @patch.object(pvmod, "_find_paraview_executable", return_value="/pv/paraview")
    def test_wait_true_waits_for_paraview(self, find_exe, popen):
        process = Mock()
        popen.return_value = process

        with tempfile.TemporaryDirectory() as td:
            m1 = Path(td) / "mesh1.exo"
            m2 = Path(td) / "mesh2.exo"
            m1.write_bytes(b"1")
            m2.write_bytes(b"2")

            pvmod.open_hierarchical_exodus_in_paraview(m1, m2, wait=True)
            process.wait.assert_called_once_with()

    def test_open_file_list_requires_at_least_one_file(self):
        with self.assertRaises(ValueError):
            pvmod.open_exodus_files_in_paraview([])


if __name__ == "__main__":
    unittest.main()
