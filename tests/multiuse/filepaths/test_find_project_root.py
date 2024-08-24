import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiuse.filepaths.find_project_root import FindProjectRoot


class TestFindProjectRoot(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "project"
        self.project_root.mkdir()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    def test_find_project_root_with_git(self):
        (self.project_root / ".git").mkdir()
        result = FindProjectRoot.find_project_root(str(self.project_root))
        self.assertEqual(result.resolve(), self.project_root.resolve())

    def test_find_project_root_with_pyproject_toml(self):
        (self.project_root / "pyproject.toml").touch()
        result = FindProjectRoot.find_project_root(str(self.project_root))
        self.assertEqual(result.resolve(), self.project_root.resolve())

    def test_find_project_root_with_project_root_file(self):
        (self.project_root / ".project_root").touch()
        result = FindProjectRoot.find_project_root(str(self.project_root))
        self.assertEqual(result.resolve(), self.project_root.resolve())

    def test_find_project_root_from_subdirectory(self):
        (self.project_root / ".git").mkdir()
        subdirectory = self.project_root / "src" / "module"
        subdirectory.mkdir(parents=True)
        result = FindProjectRoot.find_project_root(str(subdirectory))
        self.assertEqual(result.resolve(), self.project_root.resolve())

    def test_find_project_root_not_found(self):
        with self.assertRaises(FileNotFoundError):
            FindProjectRoot.find_project_root(self.temp_dir)

    @patch.dict(os.environ, {"MULTIUSE_PROJECT_ROOT": "/fake/project/root"})
    def test_find_project_root_from_env(self):
        result = FindProjectRoot.find_project_root()
        self.assertEqual(result, Path("/fake/project/root"))

    def test_verify_project_root(self):
        (self.project_root / ".git").mkdir()
        (self.project_root / "pyproject.toml").touch()
        instance = FindProjectRoot()
        self.assertTrue(instance._verify_project_root(self.project_root))

    def test_verify_project_root_insufficient_indicators(self):
        (self.project_root / "random_file.txt").touch()
        instance = FindProjectRoot()
        self.assertFalse(instance._verify_project_root(self.project_root))

    @patch("multiuse.filepaths.find_project_root.print")
    def test_debug_output(self, mock_print):
        (self.project_root / ".git").mkdir()
        (self.project_root / "pyproject.toml").touch()
        FindProjectRoot.find_project_root(str(self.project_root), debug=True)
        mock_print.assert_called_with(f"Project root found at: {self.project_root.resolve()}")


if __name__ == "__main__":
    unittest.main()
