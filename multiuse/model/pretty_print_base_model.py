import textwrap
from typing import Any, List

from pydantic import BaseModel


class PrettyPrintBaseModel(BaseModel):
    def __repr__(self) -> str:
        attributes = []
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                formatted_value = self._format_list(value)
            else:
                formatted_value = repr(value)
            attributes.append(f"{key}: {formatted_value}")

        attributes_str = "\n    ".join(attributes)
        return textwrap.dedent(
            f"""
        {self.__class__.__name__}(
            {attributes_str}
        )
        """
        ).strip()

    def _format_list(self, items: List[Any]) -> str:
        if not items:
            return "[]"
        elif len(items) == 1:
            return f"[{repr(items[0])}]"
        else:
            formatted_items = ",\n                ".join(repr(item) for item in items)
            return f"[\n                {formatted_items}\n            ]"
