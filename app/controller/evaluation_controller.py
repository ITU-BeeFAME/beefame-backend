# app/controller/evaluation_controller.py
from app.model.response import SuccessResponse
from app.service.evaluation_service import EvaluationService
from fastapi import APIRouter, HTTPException, status
from typing import List

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)

@router.get("/", response_model=SuccessResponse)
def get_evaluation():
    service = EvaluationService()

    evaluation_results = service.get_evaluation()
    print(evaluation_results)

    return SuccessResponse(data=evaluation_results)