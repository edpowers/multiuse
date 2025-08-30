"""Setup logger with optional handlers."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from multiuse.filepaths import FindProjectRoot


class SetupLogger:
    """Setup logger with optional handlers."""

    @classmethod
    def setup_logger(
        cls,
        logger_name: str,
        logging_path: Path | None = None,
        log_level: int = logging.DEBUG,
        clear_logs: bool = False,
        force_create_new_logger: bool = False,
    ) -> logging.Logger:
        """Setup logger with optional handlers.

        Args
        ----
        logger_name (str):
            The name of the logger.
        logging_path (Path | None, default = None):
            The path to the logging file.
        log_level (int, default = logging.DEBUG):
            The log level.
        clear_logs (bool, default = False):
            Whether to clear the logs.
        """
        if logging_path is None:
            logging_path = cls.make_path_given_name(logger_name)

        if clear_logs:
            for f in logging_path.parent.glob("*.log"):
                f.unlink()

        # Get the logger by name
        logger = logging.getLogger(logger_name)

        # Check if logger already has handlers and we're not forcing a new one
        if logger.handlers and not force_create_new_logger:
            # Logger already exists with handlers, just return it
            return logger

        # Clear any existing handlers if we're creating a new logger
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        # Set the log level
        logger.setLevel(log_level)
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s \n %(message)s"
        )

        # Add the handlers to the logger
        logger.addHandler(
            cls._add_file_handler(logger, logging_path, log_level, formatter)
        )
        logger.addHandler(cls._add_console_handler(logger, log_level, formatter))

        logger.file = logging_path

        return logger

    @staticmethod
    def make_path_given_name(name: str) -> Path:
        """Make a path given a name."""
        project_root = FindProjectRoot.find_project_root()
        if project_root is None:
            raise ValueError("Project root not found")

        # Get the last part of the name
        name_parts = name.split("/")
        last_part = name_parts[-1]

        # Create the path
        logging_path = project_root.joinpath("logs", *name_parts, f"{last_part}.log")

        if not logging_path.exists():
            logging_path.parent.mkdir(parents=True, exist_ok=True)

        return logging_path

    @staticmethod
    def _add_file_handler(
        logger: logging.Logger,
        logging_path: Path,
        log_level: int = logging.DEBUG,
        formatter: logging.Formatter | None = None,
    ) -> logging.FileHandler:
        """Add a file handler to the logger."""
        # Create a rotating file handler (1MB max size, 5 backup files)
        file_handler = RotatingFileHandler(
            logging_path,
            maxBytes=1_000_000,  # 1MB
            backupCount=5,
            delay=True,  # Only create the file when the first record is emitted
        )
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        if formatter:
            file_handler.setFormatter(formatter)

        return file_handler

    @staticmethod
    def _add_console_handler(
        logger: logging.Logger,
        log_level: int = logging.INFO,
        formatter: logging.Formatter | None = None,
    ) -> logging.StreamHandler:
        """Add a console handler to the logger."""
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(log_level)
        # logger.addHandler(console_handler)
        # if formatter:
        #     console_handler.setFormatter(formatter)

        # Add a Rich handler to the logger
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)

        if formatter:
            rich_handler.setFormatter(formatter)

        return rich_handler

    @staticmethod
    def _get_file_from_logger(logger: logging.Logger) -> str:
        """Get the file from the logger."""
        first = next(
            filter(
                lambda x: isinstance(x, logging.handlers.RotatingFileHandler)
                or hasattr(x, "baseFilename"),
                logger.handlers,
            )
        )

        if first is None:
            raise ValueError("No file handler found")
        if not hasattr(first, "baseFilename"):
            raise ValueError("File handler does not have a baseFilename")

        return first.baseFilename
