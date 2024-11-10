# app/controller/item_controller.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.classifier import ClassifierInfo

router = APIRouter(
    prefix="/classifier",
    tags=["classifiers"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[ClassifierInfo])
def get_classifiers(db: Session = Depends(get_db)):
    classifiers = [
        ClassifierInfo(
            Name="Support Vector Classification (SVC)",
            Url="https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html"
        ),
        ClassifierInfo(
            Name="Random Forest Classifier",
            Url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
        ),
        ClassifierInfo(
            Name="Logistic Regression",
            Url="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html"
        ),
        ClassifierInfo(
            Name="XGBClassifier",
            Url="https://xgboost.readthedocs.io/en/stable/get_started.html"
        )
    ]
    return classifiers