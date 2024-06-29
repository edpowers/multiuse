"""Staticmethods for handling null values."""

import copy
import pandas as pd
import numpy as np


class NullValuesUtils:
    """Staticmethods for handling null values."""

    

    @staticmethod
    def replace_inf_with_null(df: pd.DataFrame) -> pd.DataFrame:
        """Replace nulls with 0."""
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def arr_find_null_elements(arr: np.ndarray) -> np.ndarray:
        """
        Find all null elements in a nested numpy array.

        Parameters:
        arr (numpy.ndarray): The input nested numpy array

        Returns:
        numpy.ndarray: Boolean array where True indicates a null element
        """

        # Convert the array to a flattened version that can handle nested structures
        flattened = np.array(arr).ravel()

        # Find null elements (includes None, np.nan, and pd.NaT)
        is_null = np.frompyfunc(
            lambda x: x is None or (isinstance(x, float) and np.isnan(x)), 1, 1
        )(flattened).astype(bool)

        # Reshape the result to match the original array shape
        return is_null.reshape(arr.shape)

    @staticmethod
    def find_null_elements(nested_list: list) -> list[tuple[int, ...]]:
        """
        Find all null elements in a nested list structure.

        Parameters:
        nested_list (list): The input nested list

        Returns:
        list: A list of tuples, each containing the index path to a null element
        """

        def recurse(item, index_path=[]):
            if isinstance(item, list):
                for i, sub_item in enumerate(item):
                    yield from recurse(sub_item, index_path + [i])
            elif pd.isna(item) or item is None:
                yield tuple(index_path)

        return list(recurse(nested_list))

    @staticmethod
    def remove_null_elements(nested_list: list, null_indices: list) -> list:
        """
        Remove null elements from a nested list based on provided indices.

        Parameters:
        nested_list (list): The input nested list
        null_indices (list): List of tuples containing indices of null elements

        Returns:
        list: A new nested list with null elements removed
        """

        def remove_at_index(lst, index):
            if len(index) == 1:
                if isinstance(lst, list) and 0 <= index[0] < len(lst):
                    del lst[index[0]]
            elif isinstance(lst, list) and 0 <= index[0] < len(lst):
                remove_at_index(lst[index[0]], index[1:])
            return lst

        # Sort indices in reverse order to avoid shifting problems
        sorted_indices = sorted(null_indices, key=lambda x: x[0], reverse=True)

        # Create a deep copy of the original list to avoid modifying it
        result = copy.deepcopy(nested_list)

        for index in sorted_indices:
            remove_at_index(result, index)

        return result
