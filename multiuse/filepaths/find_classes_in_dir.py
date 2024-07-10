"""Find the classes in the directory."""

import ast
import importlib
import inspect
import os
from typing import Dict, List, Tuple

import chardet


class FindClassesInDir:
    @staticmethod
    def find_classes_in_dir(
        directory: str, include_parent_methods: bool = False
    ) -> Dict[str, List[Tuple[str, str]]]:
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

                    if classes := FindClassesInDir.get_class_methods(
                        tree,
                        directory_name,
                        module_path,
                        include_parent_methods=include_parent_methods,
                    ):
                        class_info[file_path] = classes

        return class_info

    @staticmethod
    def get_class_methods(
        tree: ast.Module,
        directory_name: str,
        module_path: str,
        include_parent_methods: bool = False,
    ) -> list:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                import_path = f"{directory_name}.{module_path}.{class_name}"

                methods = [
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                # If we want to include parent methods, we need to import the class
                if include_parent_methods:
                    try:
                        module = importlib.import_module(
                            f"{directory_name}.{module_path}"
                        )
                        class_obj = getattr(module, class_name)

                        # Get all methods (including private, async ones) from the class and its parents
                        all_methods = [
                            name
                            for name, func in inspect.getmembers(
                                class_obj,
                                predicate=lambda x: inspect.isfunction(x)
                                or inspect.iscoroutinefunction(x),
                            )
                        ]

                        # Combine methods from AST and inspect, removing duplicates
                        methods = list(set(methods + all_methods))
                    except ImportError:
                        print(f"Warning: Unable to import {import_path}")

                classes.append((class_name, import_path, methods))

        return classes
