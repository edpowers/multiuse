"""System utils for handling filepaths."""

import contextlib
import inspect
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


class SystemUtils:
    """Staticmethods for system utils."""

    @staticmethod
    def make_file_dir(file_path: str | Path) -> None:
        """Make a file dir from a string."""
        # Create the directory if it does not exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True, mode=0o777)

    @staticmethod
    def format_path(path: str) -> str:
        """Format a path."""
        return os.path.normpath(path)

    @staticmethod
    def format_path_for_import(path: str | Path) -> str | Path:
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

        return str(Path(file_path).absolute())

    @staticmethod
    def clean_directory(
        generated_code_dir: Path, python_file_patterns: list | None = None
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
                        elif f.is_dir() and f.name == "__pycache__":
                            # If it's a __pycache__ directory, remove it and its contents
                            for cache_file in f.iterdir():
                                cache_file.unlink()
                            f.rmdir()
                    except Exception as e:
                        print(f"Error deleting {f}: {e}")

            # Additionally, remove empty directories
            for root, dirs, _ in os.walk(generated_code_dir, topdown=False):
                for child_dir in dirs:
                    with contextlib.suppress(OSError):
                        Path(root).joinpath(child_dir).rmdir()

    @staticmethod
    def is_file_older_than_days(file_path: Path, days: int) -> bool:
        """
        Check if a file's last modified date is older than the specified number of days.

        Args:
            file_path: Path to the file to check
            days: Number of days to compare against

        Returns:
            True if the file is older than the specified number of days or doesn't exist,
            False otherwise
        """
        if not Path(file_path).exists():
            return True

        file_time = datetime.fromtimestamp(
            Path(file_path).stat().st_mtime, tz=timezone.utc
        )
        cutoff_time = datetime.now(tz=timezone.utc) - timedelta(days=days)

        return file_time < cutoff_time
