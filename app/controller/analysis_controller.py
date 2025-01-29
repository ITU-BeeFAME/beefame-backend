# app/controller/analysis_controller.py
from model.analysis import AnalyseRequest
from model.classifier import ClassifierName
from model.response import SuccessResponse
from service.analysis_service import AnalysisService
from fastapi import APIRouter, HTTPException, status
from typing import List
from model.dataset import DatasetName 

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


@router.post("/", response_model=SuccessResponse)
def analyse_dataset(request: AnalyseRequest):
    service = AnalysisService()
    analysis_result = service.analyse(request.dataset_name, request.classifier_name)

    return SuccessResponse(data=analysis_result)
