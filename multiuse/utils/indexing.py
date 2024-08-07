"""Indexing builtins for dictionary."""

from numbers import Number
from typing import Any, Dict, List, Union


class Indexing:
    @staticmethod
    def get_item_at_index(
        params: Union[Dict, List, tuple, Number],
        index: Union[int, slice],
        return_original_structure: bool = False,
    ) -> Any:
        """Get item at index from dictionary, list, tuple, or number.

        Example
        -------
        >>> for dist_name, params in fitted_params[state].items():
            >>> predicted_dist = DIST_NAME_MAP[dist_name](
                actual_dist.index,
                Indexing.get_item_at_index(params, 0),
                **Indexing.get_item_at_index(
                    params, slice(1, None), return_original_structure=True
                ),
            )

        Parameters
        ----------
        params : Union[Dict, List, tuple, Number]
            Dictionary, list, tuple, or number to get item from.
        index : Union[int, slice]
            Index or slice to get item from.
        return_original_structure : bool, default False
            Whether to return the original structure of the input.

        Returns
        -------
        Any
            Item at index from dictionary, list, tuple, or number.

        Raises
        ------
        IndexError
            If index is out of range.
        TypeError
            If params is not a dict, list, tuple, or number.
        """

        if isinstance(params, dict):
            keys = list(params.keys())
            values = list(params.values())
            if isinstance(index, slice) and return_original_structure:
                return {k: params[k] for k in keys[index]}
            elif (
                isinstance(index, slice)
                or 0 <= index < len(values)
                and not return_original_structure
            ):
                return values[index]
            elif 0 <= index < len(values):
                return {keys[index]: values[index]}
            else:
                raise IndexError(
                    f"Index {index} out of range for dictionary with {len(values)} items"
                )
        elif isinstance(params, (list, tuple)):
            selected = params[index]
            if return_original_structure:
                return type(params)(selected) if isinstance(index, slice) else selected
            return selected
        elif isinstance(params, Number):
            if index == 0 or (isinstance(index, slice) and index.start in (0, None)):
                return params
            else:
                raise IndexError(f"Index {index} out of range for single number")
        else:
            raise TypeError(
                f"Unsupported type: {type(params)}. Expected dict, list, tuple, or number."
            )
