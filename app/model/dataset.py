# app/model/dataset.py

from pydantic import BaseModel, HttpUrl
from typing import List

class DatasetInfo(BaseModel):
    Name: str
    Url: HttpUrl
    Instances: int
    Description: str

class DatasetSelectionRequest(BaseModel):
    Names: List[str]