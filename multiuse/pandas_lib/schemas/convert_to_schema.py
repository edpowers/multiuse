"""Convert DF to Pydantic Schema."""

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import pandas as pd
from IPython.display import display
from pydantic import BaseModel


def convert_to_float(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def convert_to_string(series: pd.Series) -> pd.Series:
    return series.astype(str)


def convert_to_datetime(value: Any) -> datetime.datetime:
    return pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")


@dataclass
class NoValidFieldType(Exception):
    field_name: str
    field_info: Any

    def __str__(self) -> str:
        return f"No valid field type for '{self.field_name}': {self.field_info}"


class ConvertToSchema:
    type_converters: Dict[Type, Callable] = {
        float: convert_to_float,
        str: convert_to_string,
        datetime.datetime: convert_to_datetime,
    }

    def __init__(self, df: pd.DataFrame, schema: Type[BaseModel]):
        self.df = df
        self.schema = schema

    def convert_to_schema(self) -> pd.DataFrame:
        converted_df = self.df.copy()

        self._log_all_extra_columns()

        for field_name, field_info in self.schema.model_fields.items():
            if field_name not in converted_df.columns:
                raise KeyError(f"{field_name} not in DataFrame columns.")

            # Raise if no valid field type annotation.
            field_type = field_info.annotation

            if not field_type:
                raise NoValidFieldType(field_name, field_info)

            converter = self.type_converters.get(field_type, convert_to_string)

            converted_df[field_name] = converter(converted_df[field_name])

        return converted_df

    def _log_all_extra_columns(self) -> None:
        extra_columns = set(self.df.columns) - set(self.schema.model_fields)
        # Log all of the columns that are not in the schema
        for column in self.df.columns:
            if column not in self.schema.model_fields:
                print(f"{column} not in schema.")

        if extra_columns:
            display(self.df[list(extra_columns)])

    def create_schema_instances(self) -> list[BaseModel]:
        converted_df = self.convert_to_schema()
        return [self.schema(**row) for row in converted_df.to_dict(orient="records")]
