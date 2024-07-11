"""Model the class data getting passed into generate code."""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import Field

from multiuse.filepaths.system_utils import SystemUtils
from multiuse.log_methods.setup_logger import SetupLogger
from multiuse.model.pretty_print_base_model import PrettyPrintBaseModel


class ClassDataModel(PrettyPrintBaseModel):
    """Class data."""

    class_object: object  # The actual class object
    class_name: str  # The name of the class
    class_methods: List[str]  # The methods (functions) of the class
    class_attributes: List[str]  # The attributes of the class
    init_params: List[str]  # The parameters of the __init__ method
    base_classes: List[str]  # The base classes of the class
    absolute_path: str  # The absolute path to the class file - full path on host system
    coroutine_methods: List[str]  # The coroutine methods of the class

    module_absolute_path: Path = Field(
        None, description="Absolute path to the module file"
    )
    module_relative_path: Path = Field(
        None, description="Relative path from the project root to the module file"
    )

    # Formatted field for pytest coverage executing in subprocess.
    # Creating here rather than at runtime to minimize ambiguity.
    coverage_file_path: str = Field(
        None, description="Path to the coverage file for the class"
    )

    log_path: Path = Field(None, description="Path to the log file for the class")
    logger_name: str = Field(None, description="Name of the logger for the class")
    class_logger: logging.Logger = Field(None, description="Logger for the class")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class ClassDataModelFactory:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.class_data_models: List[ClassDataModel] = []

    def find_class_info(self, class_name: str) -> ClassDataModel:
        """Helper method to find a class in the class data models."""
        for class_data in self.class_data_models:
            if class_data.class_name == class_name:
                return class_data

        raise KeyError(f"Class '{class_name}' not found in class data models")

    def create_from_class_info(
        self, class_info: Dict[str, List[Tuple[str, str, List[str]]]]
    ) -> List[ClassDataModel]:
        class_data_models = []

        for file_path, classes in class_info.items():
            for class_name, import_path, function_names in classes:
                print(f"Generating code for class: {class_name} from {file_path}")

                try:
                    class_to_process = self._import_class(import_path)

                    # Get the absolute path of the function's module
                    module_path = Path(
                        SystemUtils.get_class_file_path(class_to_process)
                    )
                    # Create the relative path from the project root
                    relative_path = module_path.relative_to(self.project_root)

                    generated_log_path = (
                        self.project_root.joinpath("generated_code_logs")
                        .joinpath(relative_path)
                        .joinpath(f"{class_name}.log")
                    )

                    coverage_file_path = import_path.rsplit(".", maxsplit=1)[0]

                    logger_name = f"{class_name}_logger"

                    class_logger = SetupLogger.setup_logger(
                        logger_name,
                        logging_path=generated_log_path,
                        log_level=logging.DEBUG,
                    )

                    class_model = ClassDataModel(
                        class_object=class_to_process,
                        class_name=class_name,
                        class_methods=function_names,
                        class_attributes=ClassDataModelFactory._get_class_attributes(
                            class_to_process
                        ),
                        init_params=ClassDataModelFactory._get_init_parameters(
                            class_to_process
                        ),
                        base_classes=ClassDataModelFactory._get_base_classes(
                            class_to_process
                        ),
                        absolute_path=import_path,
                        coroutine_methods=ClassDataModelFactory._get_coroutine_methods(
                            class_to_process
                        ),
                        module_absolute_path=module_path,
                        module_relative_path=relative_path,
                        coverage_file_path=coverage_file_path,
                        log_path=generated_log_path,
                        logger_name=logger_name,
                        class_logger=class_logger,
                    )

                    class_data_models.append(class_model)

                except Exception as e:
                    print(f"Error processing class {class_name}: {str(e)}")

        self.class_data_models = class_data_models

        return class_data_models

    @staticmethod
    def _import_class(import_path: str) -> Any:
        module_name, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def _get_base_classes(class_to_process: Any) -> List[str]:
        return [base.__name__ for base in class_to_process.__bases__]

    @staticmethod
    def _get_class_attributes(class_to_process: Any) -> List[str]:
        return [
            name
            for name in dir(class_to_process)
            if (
                not name.startswith("__")
                and not name.startswith("model")
                and name not in {"_abc_impl"}
            )
            and not callable(getattr(class_to_process, name))
        ]

    @staticmethod
    def _get_init_parameters(class_to_process: Any) -> List[str]:
        init_signature = inspect.signature(class_to_process.__init__)
        return [
            param.name
            for param in init_signature.parameters.values()
            if param.name != "self"
        ]

    @staticmethod
    def _get_coroutine_methods(class_to_process: Any) -> List[str]:
        return [
            name
            for name, method in inspect.getmembers(class_to_process)
            if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
        ]
