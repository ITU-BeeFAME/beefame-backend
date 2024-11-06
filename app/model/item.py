# app/model/item.py
from pydantic import BaseModel, Field
from typing import Optional

class ItemCreate(BaseModel):
    name: str = Field(..., example="Laptop")
    description: Optional[str] = Field(None, example="A high-performance laptop")
    price: float = Field(..., gt=0, example=999.99)
    quantity: int = Field(0, ge=0, example=10)

class ItemUpdate(BaseModel):
    name: Optional[str] = Field(None, example="Laptop")
    description: Optional[str] = Field(None, example="An updated description")
    price: Optional[float] = Field(None, gt=0, example=899.99)
    quantity: Optional[int] = Field(None, ge=0, example=15)

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    quantity: int

    class Config:
        orm_mode = True
