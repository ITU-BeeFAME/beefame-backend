# app/model/method.py

from pydantic import BaseModel, HttpUrl

class MethodInfo(BaseModel):
    Name: str
    Type: str
    Url: HttpUrl
    Description: str
