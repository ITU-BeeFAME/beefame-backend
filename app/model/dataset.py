# app/model/dataset.py

from pydantic import BaseModel, HttpUrl
from typing import List

class DatasetInfo(BaseModel):
    name: str
    url: HttpUrl
    instances: int
    description: str

class DatasetSelectionRequest(BaseModel):
    names: List[str]