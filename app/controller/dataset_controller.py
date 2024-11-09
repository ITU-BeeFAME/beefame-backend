# app/controller/item_controller.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.dataset import DatasetInfo, DatasetSelectionRequest

router = APIRouter(
    prefix="/dataset",
    tags=["datasets"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[DatasetInfo])
def get_datasets(db: Session = Depends(get_db)):
    # Example dataset - replace this with actual data fetching or processing logic
    datasets = [
        DatasetInfo(
            Name="Statlog (German Credit Data)",
            Url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            Instances=1000,
            Description="This dataset classifies people described by a set of attributes as good or bad credit risks. Comes in two formats (one all numeric). Also comes with a cost matrix."
        ),
        DatasetInfo(
            Name="Census Income",
            Url="https://archive.ics.uci.edu/dataset/20/census+income",
            Instances=48842,
            Description="Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset."
        )
    ]
    return datasets

@router.post("/", response_model=List[str])
def receive_selected_datasets(selection: DatasetSelectionRequest, db: Session = Depends(get_db)):
    
    # todo: business logic
    print(selection)
    response = ["test"]
    
    if not response:
        raise HTTPException(status_code=404, detail="No datasets found")
    
    return response