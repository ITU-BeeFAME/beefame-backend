# app/controller/item_controller.py
from app.model.response import SuccessResponse
from app.service.method_service import MethodService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.method import MethodInfo

router = APIRouter(
    prefix="/methods",
    tags=["Methods"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=SuccessResponse)
def get_methods(db: Session = Depends(get_db)):
    service = MethodService()
    methods = service.get_methods()
    return SuccessResponse(data=methods)