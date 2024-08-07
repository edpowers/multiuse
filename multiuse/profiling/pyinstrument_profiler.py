""""""

import contextlib
from pathlib import Path
from typing import Optional

from pyinstrument import Profiler


class PyInstrumentProfiler:
    """Implementation of pyinstrument profiler.

    Example
    -------
    >>> from scripts.dev.multiuse.profilers import PyInstrumentProfiler
    >>> profiler = PyInstrumentProfiler()
    >>> profiler.start()
    >>> # Do stuff
    >>> profiler.stop()
    """

    def __init__(self, profiling_dir: Optional[Path] = None) -> None:
        self.profiler = Profiler()

        # Reset the profiler
        if self.profiler.is_running:
            self.profiler.reset()

        self.profiler_root: Optional[Path] = profiling_dir

        # Make the directories if they don't already exist.
        if isinstance(self.profiler_root, Path):
            self.profiler_root.mkdir(exist_ok=True)

    def start(self) -> None:
        """Start profiling."""
        # Reset the profiler
        if self.profiler.is_running:
            self.profiler.reset()
        else:
            with contextlib.suppress(RuntimeError):
                self.profiler.start()

    def stop(self) -> None:
        """Stop profiling."""
        if not self.profiler.is_running:
            print("Profiler is not running.")
        else:
            self.profiler.stop()

    def render(self) -> None:
        """Render profile."""
        self.profiler.output_text()

    def render_in_browser(self) -> None:
        """Render profile in browser."""
        # if not self.profiler.is_running:
        #    print("Profiler is not running.")
        # else:
        self.profiler.open_in_browser()

    def write_profile(self, name: str) -> None:
        """Write profile to file."""
        if not isinstance(self.profiler_root, Path):
            raise ValueError("profiling_dir must be a Path object.")

        results_file = self.profiler_root.joinpath(f"{name}.html")
        with open(results_file, "w", encoding="utf-8") as f_html:
            f_html.write(self.profiler.output_html())
