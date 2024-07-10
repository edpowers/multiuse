"""Class for reliably finding the project root."""

import os
from pathlib import Path


class FindProjectRoot:
    @classmethod
    def find_project_root(cls, start_path: str = "", debug: bool = False) -> Path:
        instance = cls()
        # Usage example
        try:
            project_root = instance._find_project_root(start_path)
            if debug:
                if instance._verify_project_root(project_root):
                    print(f"Project root found at: {project_root}")
                else:
                    print(
                        f"Found path {project_root}, but it may not be a valid project root."
                    )
            return project_root
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise e

    def _find_project_root(self, start_path: str = "") -> Path:
        """
        Find the project root directory.

        This function looks for common project root indicators like .git, pyproject.toml,
        or a custom .project_root file.

        Args:
        start_path (str): The directory to start searching from. Defaults to the current working directory.

        Returns:
        Path: The path to the project root directory.

        Raises:
        FileNotFoundError: If the project root cannot be found.
        """
        if not start_path:
            start_path = os.getcwd()

        current_path = Path(start_path).resolve()

        while True:
            # Check for common project root indicators
            if (current_path / ".git").exists():
                return current_path
            if (current_path / "pyproject.toml").exists():
                return current_path
            if (current_path / ".project_root").exists():
                return current_path

            # Move up one directory
            parent_path = current_path.parent

            # If we've reached the root directory and haven't found anything, raise an error
            if parent_path == current_path:
                raise FileNotFoundError("Project root not found.")

            current_path = parent_path

    def _verify_project_root(self, root_path: Path) -> bool:
        """
        Verify that the found path is indeed the project root.

        Args:
        root_path (Path): The path to verify.

        Returns:
        bool: True if the path seems to be a valid project root, False otherwise.
        """
        # Check for the presence of key project files/directories
        key_indicators = [
            ".git",
            "pyproject.toml",
            ".project_root",
            "setup.py",
            "requirements.txt",
            "src",
            "tests",
        ]

        indicator_count = sum(
            (root_path / indicator).exists() for indicator in key_indicators
        )

        # If at least two indicators are present, we consider it a valid project root
        return indicator_count >= 2
