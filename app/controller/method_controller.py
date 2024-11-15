# app/controller/item_controller.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.method import MethodInfo

router = APIRouter(
    prefix="/method",
    tags=["methods"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[MethodInfo])
def get_methods(db: Session = Depends(get_db)):
    methods = [
        MethodInfo(
            Name="Data Repaierer",
            Type="Preprocessing",
            Url="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/data_repairer.py",
            Description="Transforms the data distribution so that a given feature distribution is marginally independent of the sensitive attribute, s."
        ),
        MethodInfo(
            Name="Prevalence Sampling",
            Type="Preprocessing",
            Url="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/prevalence_sample.py",
            Description="Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset."
        ),
        MethodInfo(
            Name="Relabeller",
            Type="Preprocessing",
            Url="https://github.com/cosmicBboy/themis-ml/blob/master/themis_ml/preprocessing/relabelling.py",
            Description="Relabels target variables using a function that can compute a decision boundary in input data space using heuristic."
        )
    ]
    return methods