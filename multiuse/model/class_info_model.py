"""Class Info Models."""

from typing import Dict, List, Tuple

from pydantic import BaseModel


class ClassInfoTuple(BaseModel):
    """Class Info Model."""

    class_name: str
    module_path: str
    methods: List[str]


class ClassInfoModel(BaseModel):
    """Class Info Model."""

    class_info: Tuple[ClassInfoTuple]


class AllClassInfo(BaseModel):
    """All Class Info Model."""

    class_info: Dict[str, List[ClassInfoModel]]
