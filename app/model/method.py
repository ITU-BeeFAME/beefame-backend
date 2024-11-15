# app/model/method.py

from pydantic import BaseModel, HttpUrl

class MethodInfo(BaseModel):
    name: str
    type: str
    url: HttpUrl
    description: str
