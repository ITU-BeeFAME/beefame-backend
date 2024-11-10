# app/model/classifier.py

from pydantic import BaseModel, HttpUrl

class ClassifierInfo(BaseModel):
    Name: str
    Url: HttpUrl
