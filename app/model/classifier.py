# app/model/classifier.py

from pydantic import BaseModel, HttpUrl

class ClassifierInfo(BaseModel):
    name: str
    url: HttpUrl
