# app/controller/item_controller.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.response import FailureResponse, SuccessResponse
from app.service.bias_metric_service import BiasMetricService
from fastapi import APIRouter, HTTPException, status
from typing import List
from app.model.method import MethodInfo

router = APIRouter(
    prefix="/bias-metrics",
    tags=["Bias Metrics"],
)

""" @router.post("/")
def check_bias_metrics(datasetList: BiasMetricRequest, db: Session = Depends(get_db)):
    service = BiasMetricService()
    
    bias_metrics = service.fetch_all_bias_metrics(datasetList)

    if bias_metrics is None:
        return FailureResponse(status=404, message="No datasets found")

    return SuccessResponse(data=bias_metrics) """

@router.post("/", response_model=SuccessResponse)
def get_bias_metrics_for_selected_datasets(datasetList: BiasMetricRequest):
    service = BiasMetricService()

    # datasetList must be something like ["id of dataset", "id of dataset"]
    # datasetList = ["gM5tmlniAYWYjnIVKQZ6"]
    bias_metrics = service.fetch_bias_metrics(datasetList)
    
    # example usage of adding a bias metric to db
    # metric_added = service.add_bias_metric("Age", "Old", "Young", 75, "With default thresholds, bias against unprivileged group detected in 4 out of 5 metrics", ["Equal Opportunity Difference","Average Odds Difference", "Disparate Impact","Theil Index"])

    return SuccessResponse(data=bias_metrics)