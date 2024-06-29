"""Save output to file."""

from pathlib import Path

from typing import Union


class SeleniumSaveToFile:
    """Save the selenium output to a file."""

    def __init__(self, fpath: str = "") -> None:
        self.fpath = fpath

    # Split url at query string
    def split_url(self, url: str) -> str:
        return url.split("?")[0]

    def get_url_path(self, url: str) -> str:
        return self.split_url(url).replace("https://", "").replace("/", "_")

    def make_parent_path(self, fpath: Union[Path, str]) -> None:
        """Make the parent path."""
        Path(fpath).parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    def write_html_to_file(self, html: str, fpath: Union[Path, str]) -> None:
        """Write the html to a file."""
        with open(fpath, "w") as f:
            f.write(html)
