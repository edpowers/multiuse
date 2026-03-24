"""Class for reliably finding the project root."""

from functools import lru_cache
from pathlib import Path


@lru_cache
def find_project_root() -> Path:
    current_dir = Path.cwd()
    while not (current_dir / ".git").exists():
        current_dir = current_dir.parent
    return current_dir
