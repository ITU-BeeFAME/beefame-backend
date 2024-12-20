# app/controller/item_controller.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.response import FailureResponse, SuccessResponse
from app.service.bias_metric_service import BiasMetricService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import SessionLocal
from app.model.method import MethodInfo

router = APIRouter(
    prefix="/bias-metrics",
    tags=["Bias Metrics"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def check_bias_metrics(datasetList: BiasMetricRequest, db: Session = Depends(get_db)):
    service = BiasMetricService()
    
    bias_metrics = service.fetch_all_bias_metrics(datasetList)

    if bias_metrics is None:
        return FailureResponse(status=404, message="No datasets found")
    
    # example usage of adding a bias metric to db
    # metric_added = service.add_bias_metric("abc", "def", "xyz", 75, "abc", ["aa"])

    return SuccessResponse(data=bias_metrics)