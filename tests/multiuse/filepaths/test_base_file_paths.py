import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from multiuse.file_io.parquet_io import ParquetIO
from multiuse.filepaths.base_file_paths import BaseFilePaths


class TestBaseFilePaths(unittest.TestCase):
    def setUp(self):
        self.base_file_paths = BaseFilePaths()
        self.base_file_paths.data_dict = {
            "test_key": Path("/test/path/file.parquet"),
            "csv_key": Path("/test/path/file.csv"),
        }

    def test_repr(self):
        result = repr(self.base_file_paths)
        self.assertEqual(result, "")  # __repr__ returns an empty string

    @patch("os.getcwd")
    def test_return_project_base_directory(self, mock_getcwd):
        mock_getcwd.return_value = "/test/project"
        result = self.base_file_paths._return_project_base_directory()
        self.assertEqual(result, Path("/test/project"))

    def test_get_fpath_existing_key(self):
        result = self.base_file_paths.get_fpath("test_key")
        self.assertEqual(result, Path("/test/path/file.parquet"))

    def test_get_fpath_non_existing_key(self):
        with self.assertRaises(KeyError):
            self.base_file_paths.get_fpath("non_existing_key")

    @patch.object(Path, "exists")
    @patch.object(ParquetIO, "read_parquet")
    def test_return_df_existing_parquet(self, mock_read_parquet, mock_exists):
        mock_exists.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_parquet.return_value = mock_df

        result = self.base_file_paths.return_df("test_key")

        mock_read_parquet.assert_called_once_with(Path("/test/path/file.parquet"))
        pd.testing.assert_frame_equal(result, mock_df)

    @patch.object(Path, "exists")
    def test_return_df_non_existing_file(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.base_file_paths.return_df("test_key")

    @patch.object(Path, "exists")
    def test_return_df_non_existing_file_return_empty(self, mock_exists):
        mock_exists.return_value = False
        result = self.base_file_paths.return_df(
            "test_key",
            return_empty_df_if_missing=True,
            columns_for_empty_df=["col1", "col2"],
        )
        expected = pd.DataFrame(columns=["col1", "col2"])
        pd.testing.assert_frame_equal(result, expected)

    @patch.object(Path, "exists")
    def test_return_df_unsupported_file_type(self, mock_exists):
        mock_exists.return_value = True
        with self.assertRaises(ValueError):
            self.base_file_paths.return_df("csv_key")


if __name__ == "__main__":
    unittest.main()
