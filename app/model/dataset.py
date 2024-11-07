# app/model/dataset.py

from pydantic import BaseModel, HttpUrl

class DatasetInfo(BaseModel):
    Name: str
    Url: HttpUrl
    Instances: int
    Description: str
