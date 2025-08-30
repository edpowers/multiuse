"""Class for reliably finding the project root."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class FindProjectRoot:
    """Class for reliably finding the project root."""

    @classmethod
    def find_project_root(
        cls,
        start_path: str = "",
        debug: bool = False,
        raise_error_if_no_env_file: bool = False,
        raise_if_not_path: bool = False,
    ) -> Path | None:
        """
        Find the project root directory.

        This function looks for common project root indicators like .git, pyproject.toml,
        or a custom .project_root file. It first checks for an environment variable,
        then searches for indicators if the variable is not set.

        Args:
            start_path (str): The directory to start searching from. Defaults to the current working directory.
            debug (bool): If True, print debug information.
            raise_error_if_no_env_file (bool): If True, raise an error if no .env file is found.

        Returns:
            Path: The path to the project root directory.

        Raises:
            FileNotFoundError: If the project root cannot be found.
        """
        cls._load_environment(raise_error_if_no_env_file)

        if project_root := cls._get_project_root_from_env():
            if not isinstance(project_root, Path):
                raise ValueError("Project root is not a valid Path object.")
            return project_root

        instance = cls()
        try:
            project_root = instance._find_project_root(start_path)
            cls._handle_debug_output(debug, instance, project_root)
            if not isinstance(project_root, Path):
                raise ValueError("Project root is not a valid Path object.")
            return project_root
        except FileNotFoundError as e:
            cls._handle_error(e)

        if raise_if_not_path:
            raise FileNotFoundError("Project root not found.")

        return None

    @staticmethod
    def _load_environment(raise_error_if_no_env_file: bool) -> None:
        load_dotenv(find_dotenv(raise_error_if_not_found=raise_error_if_no_env_file))

    @staticmethod
    def _get_project_root_from_env() -> Path | None:
        if project_root := os.environ.get("MULTIUSE_PROJECT_ROOT"):
            return Path(project_root)
        return None

    @classmethod
    def _handle_debug_output(
        cls, debug: bool, instance: "FindProjectRoot", project_root: Path
    ) -> None:
        if debug:
            if instance._verify_project_root(project_root):
                print(f"Project root found at: {project_root}")
            else:
                print(
                    f"Found path {project_root}, but it may not be a valid project root."
                )

    @staticmethod
    def _handle_error(error: Exception) -> None:
        print(f"Error: {error}")
        raise error

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
