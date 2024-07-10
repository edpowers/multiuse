"""System utils for handling filepaths."""

import inspect
import os
from pathlib import Path
from typing import List, Optional, Union


class SystemUtils:
    """Staticmethods for system utils."""

    @staticmethod
    def make_file_dir(file_path: Union[str, Path]) -> None:
        """Make a file dir from a string."""
        # Create the directory if it does not exist
        os.makedirs(Path(file_path).parent, exist_ok=True, mode=0o777)

    @staticmethod
    def format_path(path: str) -> str:
        """Format a path."""
        return os.path.normpath(path)

    @staticmethod
    def format_path_for_import(path: Union[str, Path]) -> Union[str, Path]:
        """Format a path to posix.

        Args
        ----
        path : Union[str, Path]
            The path to format. If string, will return string. If path, will return path.
        """

        def format_path_str(path: str) -> str:
            return (
                path.replace("/", ".")
                .replace(".py", "")
                .strip(".")
                .replace(".__init__", "___init__")
            )

        if isinstance(path, Path):
            return Path(format_path_str(str(path)))

        return format_path_str(path)

    @staticmethod
    def get_class_file_path(cls: object) -> str:
        # Get the module of the class
        module = inspect.getmodule(cls)

        if module is None:
            raise ValueError(f"Could not find {cls=}")

        # Get the file path of the module
        file_path = inspect.getfile(module)

        return os.path.abspath(file_path)

    @staticmethod
    def clean_directory(
        generated_code_dir: Path, python_file_patterns: Optional[List] = None
    ) -> None:
        """Clean the generated code directory."""
        if not python_file_patterns:
            # Use a list to include both .py and .pyc files
            python_file_patterns = ["*.py", "*.pyc"]

        if generated_code_dir.exists():
            for pattern in python_file_patterns:
                for f in generated_code_dir.rglob(pattern):
                    try:
                        if f.is_file():
                            f.unlink()
                        elif f.is_dir():
                            # If it's a __pycache__ directory, remove it and its contents
                            if f.name == "__pycache__":
                                for cache_file in f.iterdir():
                                    cache_file.unlink()
                                f.rmdir()
                    except Exception as e:
                        print(f"Error deleting {f}: {e}")

            # Additionally, remove empty directories
            for root, dirs, files in os.walk(generated_code_dir, topdown=False):
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except OSError:
                        # Directory not empty, skip it
                        pass
