import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from multiuse.model.class_data_model import ClassDataModel, ClassDataModelFactory
from multiuse.model.pretty_print_base_model import PrettyPrintBaseModel


class TestClassDataModel(unittest.TestCase):
    def setUp(self):
        self.class_data = ClassDataModel(
            class_object=TestClassDataModel,
            class_name="TestClass",
            class_methods=["method1", "method2"],
            class_attributes=["attr1", "attr2"],
            init_params=["param1", "param2"],
            base_classes=["BaseClass"],
            absolute_path="/path/to/class",
            coroutine_methods=["async_method"],
            module_absolute_path=Path("/abs/path/to/module.py"),
            module_relative_path=Path("relative/path/to/module.py"),
            coverage_file_path="/path/to/coverage",
            log_path=Path("/path/to/log"),
            logger_name="test_logger",
            class_logger=logging.getLogger("test_logger"),
        )

    def test_import_path(self):
        with patch(
            "multiuse.model.class_data_model.SystemUtils.format_path_for_import"
        ) as mock_format:
            mock_format.return_value = "formatted.import.path"
            self.assertEqual(self.class_data.import_path, "formatted.import.path")

    def test_import_path_error(self):
        with patch(
            "multiuse.model.class_data_model.SystemUtils.format_path_for_import"
        ) as mock_format:
            mock_format.return_value = Path("not_a_string")
            with self.assertRaises(ValueError):
                _ = self.class_data.import_path

    def test_import_statement(self):
        with patch.object(ClassDataModel, "import_path", "module.path"):
            self.assertEqual(
                self.class_data.import_statement, "from module.path import TestClass"
            )

    def test_raise_if_class_name_not_in_code(self):
        self.class_data.raise_if_class_name_not_in_code("TestClass is here")
        with self.assertRaises(ValueError):
            self.class_data.raise_if_class_name_not_in_code("No class here")

    def test_raise_if_import_path_not_in_code(self):
        with patch.object(ClassDataModel, "import_path", "module.path"):
            self.class_data.raise_if_import_path_not_in_code(
                "from module.path import TestClass"
            )
            with self.assertRaises(ImportError):
                self.class_data.raise_if_import_path_not_in_code("No import here")

    def test_raise_if_no_test_in_code(self):
        self.class_data.raise_if_no_test_in_code("def test_method1():")
        self.class_data.raise_if_no_test_in_code("def test_TestClass():")
        with self.assertRaises(ValueError):
            self.class_data.raise_if_no_test_in_code("No test here")


class TestClassDataModelFactory(unittest.TestCase):
    def setUp(self):
        self.factory = ClassDataModelFactory(Path("/project/root"))

    def test_find_class_info(self):
        mock_class_data = MagicMock(class_name="TestClass")
        self.factory.class_data_models = [mock_class_data]
        self.assertEqual(self.factory.find_class_info("TestClass"), mock_class_data)
        with self.assertRaises(KeyError):
            self.factory.find_class_info("NonexistentClass")

    def test_sort_by_hierarchy(self):
        class_data1 = MagicMock(class_name="Class1")
        class_data2 = MagicMock(class_name="Class2")
        class_data3 = MagicMock(class_name="Class3")
        self.factory.class_data_models = [class_data2, class_data3, class_data1]
        sorted_models = self.factory.sort_by_hierarchy(["Class1", "Class2", "Class3"])
        self.assertEqual(sorted_models, [class_data1, class_data2, class_data3])

    def test_import_class(self):
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = "TestClass"
            mock_import.return_value = mock_module
            result = ClassDataModelFactory._import_class("module.TestClass")
            self.assertEqual(result, "TestClass")

    def test_get_base_classes(self):
        class BaseClass:
            pass

        class TestClass(BaseClass):
            pass

        base_classes = ClassDataModelFactory._get_base_classes(TestClass)
        self.assertEqual(base_classes, ["BaseClass"])

    def test_get_class_attributes(self):
        class TestClass:
            attr1 = 1
            attr2 = "test"

            def method(self):
                pass

        attributes = ClassDataModelFactory._get_class_attributes(TestClass)
        self.assertEqual(set(attributes), {"attr1", "attr2"})

    def test_get_init_parameters(self):
        class TestClass:
            def __init__(self, param1, param2):
                pass

        init_params = ClassDataModelFactory._get_init_parameters(TestClass)
        self.assertEqual(init_params, ["param1", "param2"])

    def test_get_coroutine_methods(self):
        class TestClass:
            async def async_method(self):
                pass

            def normal_method(self):
                pass

        coroutine_methods = ClassDataModelFactory._get_coroutine_methods(TestClass)
        self.assertEqual(coroutine_methods, ["async_method"])


if __name__ == "__main__":
    unittest.main()
