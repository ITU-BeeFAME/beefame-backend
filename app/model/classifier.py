# app/model/classifier.py

from pydantic import BaseModel, HttpUrl

class ClassifierInfo(BaseModel):
    id: str
    name: str
    url: HttpUrl
