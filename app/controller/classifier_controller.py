# app/controller/item_controller.py
from app.model.response import SuccessResponse
from app.service.classifier_service import ClassifierService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.classifier import ClassifierInfo

router = APIRouter(
    prefix="/classifiers",
    tags=["Classifiers"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=SuccessResponse)
def get_classifiers(db: Session = Depends(get_db)):
    service = ClassifierService()
    classifiers = service.get_classifiers()
    return SuccessResponse(data=classifiers)