"""Custom logging funcs."""

import contextlib
import sys
import traceback
from typing import Dict, Optional


class CustomLoggingFuncs:
    @classmethod
    def show_code_lines(cls, e: Optional[BaseException] = None) -> str:
        """Show the code lines of the exception."""
        instance = cls()
        func_n_d = instance.get_parent_function_names()
        argp = ".".join(
            list(
                map(instance.reduce_if_current_name, reversed(list(func_n_d.values())))
            )
        )

        if e:
            argp += f"{instance.return_traceback_filepath_string(e)}"

        return argp

    @staticmethod
    def get_parent_function_names() -> dict:
        """Get the parent function names as dict."""
        func_n_d: Dict[str, str] = {}
        func_n_l = [
            "func_name",
            "func_parent",
            "func_gparent",
            "func_ggparent",
            "func_gggparent",
            "func_gggparent",
        ]
        for i, name in zip(range(2, 6), func_n_l):
            with contextlib.suppress(ValueError):
                frame_val = sys._getframe(i)
                func_name = frame_val.f_code.co_name
                func_line = frame_val.f_lineno

                func_val_str = f"{func_name}:_{func_line}_"

                if func_val_str in func_n_d.values():
                    continue
                else:
                    func_n_d[name] = (
                        f"{func_name}:_{func_line}_\n"
                        if name == "func_parent"
                        else f"{func_name}:_{func_line}_"
                    )

        return func_n_d

    @staticmethod
    def return_traceback_filepath_string(e: BaseException) -> str:
        """Find and return the traceback filepath string."""
        # Extract the traceback details
        tb = traceback.extract_tb(e.__traceback__)

        # Get the last frame
        frame = tb[-1]

        filepath = frame.filename
        line_number = frame.lineno
        function_name = frame.name
        code_line = frame.line

        return f"{filepath}-{line_number}-{function_name}-{code_line}"

    @staticmethod
    def reduce_if_current_name(func_n: str) -> str:
        """Function name reduction."""
        reduce_t = ("help_print_arg", "print_time_ts", "hpa")
        return "" if func_n in reduce_t else func_n
