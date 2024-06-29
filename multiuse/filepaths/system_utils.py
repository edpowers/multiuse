"""System utils for handling filepaths."""

import ast
import asyncio
import hashlib
import inspect
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar

import chardet

T = TypeVar("T")


def persistent_cache(
    *args: Any,
    cache_dir: str = "cache",
    **kwargs: Any,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get the function's module and name
            module: Optional[Any] = inspect.getmodule(func)
            module_path: str = module.__file__ if module else ""
            func_name: str = func.__name__

            # Create a unique cache key
            key: str = f"{module_path}:{func_name}"
            for arg in args:
                key += f":{str(arg)}"
            for k, v in kwargs.items():
                key += f":{k}:{str(v)}"

            # Hash the key to create a filename
            filename: str = f"{hashlib.md5(key.encode()).hexdigest()}.pickle"
            cache_file: str = os.path.join(cache_dir, filename)

            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Check if cached result exists
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            # If not cached, call the function
            result: T = await func(*args, **kwargs)

            # Cache the result
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            # Check if the function is a coroutine function
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class SystemUtils:
    """Staticmethods for system utils."""

    @staticmethod
    def get_class_file_path(cls: object) -> str:
        # Get the module of the class
        module = inspect.getmodule(cls)

        assert module is not None, f"Could not find {cls=}."
        # assert isinstance(module, Module), f"Expected {module=} to be a Module

        # Get the file path of the module
        file_path = inspect.getfile(module)

        return os.path.abspath(file_path)

    @staticmethod
    def find_project_root(start_path: str) -> Path:
        """Find the project root by looking for a .git directory or pyproject.toml file."""
        path = Path(start_path).resolve()
        for parent in [path, *path.parents]:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("Could not find project root.")

    @staticmethod
    def find_classes_in_dir(directory: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Traverse a directory and find all classes with their import paths.

        Args:
        directory (str): The root directory to start the search from.

        Returns:
        Dict[str, List[Tuple[str, str]]]: A dictionary where keys are file paths
        and values are lists of tuples containing (class_name, import_path).
        """
        class_info = {}

        directory_name = str(directory).rsplit("/", maxsplit=1)[-1]

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    module_path = os.path.splitext(relative_path)[0].replace(
                        os.path.sep, "."
                    )

                    try:
                        # Detect the file encoding
                        with open(file_path, "rb") as f:
                            raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result["encoding"]

                        # Read the file with the detected encoding
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()

                        tree = ast.parse(content)
                    except (UnicodeDecodeError, SyntaxError) as e:
                        print(f"Error in file {file_path}: {str(e)}")
                        continue

                    classes = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            import_path = f"{directory_name}.{module_path}.{class_name}"
                            classes.append((class_name, import_path))

                    if classes:
                        class_info[file_path] = classes

        return class_info
