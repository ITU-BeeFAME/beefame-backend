# app/controller/item_controller.py
from app.model.response import SuccessResponse
from app.service.dataset_service import DatasetService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.dataset import DatasetInfo, DatasetSelectionRequest

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=SuccessResponse)
def get_datasets(db: Session = Depends(get_db)):
    service = DatasetService()
    datasets = service.get_datasets()
    return SuccessResponse(data=datasets)
