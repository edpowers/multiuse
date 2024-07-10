"""Setup logger with optional handlers."""

import logging
from pathlib import Path


class SetupLogger:
    @classmethod
    def setup_logger(
        cls,
        logger_name: str,
        logging_path: Path,
        log_level: int = logging.DEBUG,
    ) -> logging.Logger:
        """Setup logger with optional handlers."""
        # Ensure the directory exists
        logging_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a logger with the class name
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        # Create a file handler
        file_handler = logging.FileHandler(logging_path)
        file_handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # You can adjust this level as needed

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
